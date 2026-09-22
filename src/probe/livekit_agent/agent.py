"""Versa LiveKit voice-agent worker.

Run as its own process (it is NOT the web server):

    uv run --no-sync python -m probe.livekit_agent.agent dev     # local dev
    uv run --no-sync python -m probe.livekit_agent.agent start   # production

What it does per room (one student voice session):

1. LiveKit delivers the student's microphone audio.
2. STT (LiveKit Inference) transcribes each user turn; VAD + turn detection
   decide when the student has finished, and enable natural barge-in.
3. `VersaAgent.llm_node` is overridden so the *answer* is NOT a generic LLM
   reply — it POSTs the transcript to the existing Versa pipeline
   (`/api/voice/turn`), which runs ambiguity handling, learner-scoped Moss
   retrieval, and Gemini. One brain, shared with text mode.
4. TTS (LiveKit Inference) streams the answer back through LiveKit as audio.

Interruption/barge-in is handled natively by `AgentSession`.

Config (env / .env):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET   (required)
    VERSA_API_URL        base URL of the running Versa web server
                         (default http://127.0.0.1:8000)
    LIVEKIT_STT_MODEL    inference STT model  (default deepgram/nova-3)
    LIVEKIT_TTS_MODEL    inference TTS model  (default cartesia/sonic-2)
"""

from __future__ import annotations

import json
import logging
import os

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, inference
from livekit.plugins import silero

# Browser listens for this data topic to render the live Moss retrieval panel.
_TELEMETRY_TOPIC = "versa.telemetry"

load_dotenv()
logger = logging.getLogger("versa-agent")

VERSA_API_URL = os.getenv("VERSA_API_URL", "http://127.0.0.1:8000").rstrip("/")
STT_MODEL = os.getenv("LIVEKIT_STT_MODEL", "deepgram/nova-3")
TTS_MODEL = os.getenv("LIVEKIT_TTS_MODEL", "cartesia/sonic-2")

# Room name convention set by the token endpoint: "versa__<learner>__<rand>".
_ROOM_PREFIX = "versa"
_ROOM_SEP = "__"


def learner_from_room(room_name: str) -> str:
    """Extract the learner id/label the browser encoded into the room name.
    Falls back to a demo learner so a manually-created room still works."""
    parts = (room_name or "").split(_ROOM_SEP)
    if len(parts) >= 3 and parts[0] == _ROOM_PREFIX and parts[1]:
        return parts[1]
    return "demo-learner-a"


class VersaAgent(Agent):
    """A LiveKit Agent whose answers come from the existing Versa pipeline.

    Only `llm_node` is customized — STT, TTS, VAD, turn detection and
    interruption are the framework's. `llm_node` pulls the latest user
    transcript and asks Versa over HTTP, so learner memory (Moss) and
    reasoning (Gemini) are never re-implemented here."""

    def __init__(self, learner_id: str, room: rtc.Room | None = None) -> None:
        super().__init__(instructions="You are Versa, a friendly, concise voice tutor.")
        self._learner = learner_id
        self._room = room
        self._session_id: str | None = None
        self._http: aiohttp.ClientSession | None = None

    async def _publish_telemetry(self, data: dict) -> None:
        """Send the Moss retrieval telemetry to the browser so the voice page
        can visibly show engine/latency/memories (VERSA voice context §9)."""
        if self._room is None:
            return
        payload = {
            "type": "moss",
            "retrieval": data.get("retrieval"),
            "branched": data.get("branched"),
        }
        try:
            await self._room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"),
                reliable=True,
                topic=_TELEMETRY_TOPIC,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry is best-effort
            logger.debug("telemetry publish failed: %s", exc)

    async def _http_session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    async def _ask_versa(self, user_text: str) -> str:
        """POST the transcript to the existing Versa tutor turn and return the
        spoken answer. Reuses the Versa session across turns (multi-turn
        context) and speaks the clarification + options on an ambiguous turn."""
        http = await self._http_session()
        payload = {
            "learner": self._learner,
            "text": user_text,
            "session_id": self._session_id,
        }
        try:
            async with http.post(
                f"{VERSA_API_URL}/api/voice/turn",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                data = await resp.json()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to speech
            logger.warning("Versa call failed: %s", exc)
            return "Sorry, I couldn't reach the tutor just now. Could you say that again?"

        self._session_id = data.get("session_id") or self._session_id
        await self._publish_telemetry(data)
        message = (data.get("message") or "").strip()
        if data.get("branched") and data.get("pending_options"):
            opts = ", or ".join(o["text"] for o in data["pending_options"])
            return f"{message} You can say: {opts}"
        return message or "Okay."

    async def llm_node(self, chat_ctx, tools, model_settings):  # type: ignore[override]
        # The most recent user turn's transcript.
        user_text = ""
        for item in reversed(chat_ctx.items):
            if getattr(item, "role", None) == "user":
                user_text = (item.text_content or "").strip()
                break
        if not user_text:
            yield "Sorry, I didn't catch that — could you repeat it?"
            return
        yield await self._ask_versa(user_text)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    learner = learner_from_room(ctx.room.name)
    logger.info("Versa agent joined room %s (learner=%s)", ctx.room.name, learner)

    session = AgentSession(
        stt=inference.STT(model=STT_MODEL),
        tts=inference.TTS(model=TTS_MODEL),
        vad=silero.VAD.load(),
        # No LLM: VersaAgent.llm_node fully overrides response generation.
    )
    await session.start(agent=VersaAgent(learner, ctx.room), room=ctx.room)
    # Greet without going through llm_node (which would have no user turn yet).
    await session.say("Hi! I'm Versa. What would you like to learn today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
