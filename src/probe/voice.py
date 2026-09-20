"""Voice output layer — text-to-speech only (VERSA remaining-build §21).

A thin, backend-only wrapper so the ElevenLabs secret never reaches the
browser (§6.4/§41): the browser posts a transcript, the server runs the
existing Versa tutor, and this service turns the answer text into audio the
browser plays back. Nothing here touches the tutoring pipeline, learner
memory, or Moss — voice is strictly an input/output surface (§19).

Two implementations behind one `VoiceService` interface:

- `ElevenLabsVoiceService` — real TTS via the ElevenLabs REST API using the
  already-available `httpx` (no new dependency). Low-latency model by
  default; voice/model overridable by env.
- `StubVoiceService` — returns a tiny fixed byte string, for tests and for
  offline runs, so nothing calls the paid API in CI (§42).

`build_voice_service` picks the real service when `ELEVENLABS_API_KEY` is
set, else the stub — mirroring the `--stub` convention used everywhere else.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

import httpx

# A public ElevenLabs prebuilt voice ("Rachel") and a low-latency model, both
# overridable by env. These are not secrets.
_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
_DEFAULT_MODEL_ID = "eleven_turbo_v2_5"
_ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Spoken answers are kept short to conserve the ElevenLabs allowance (§54)
# and to feel responsive (§23) — the full answer text is still shown in the
# transcript; only the SPOKEN version is trimmed.
_MAX_SPOKEN_SENTENCES = 4
_MAX_SPOKEN_CHARS = 600


def speakable_text(answer: str) -> str:
    """Turn a tutor answer into concise, plain speech: strip markdown
    emphasis/headers/list markers and code fences, then cap to a few
    sentences. Never speaks JSON, IDs, or markup (§23)."""
    text = answer.strip()
    # Drop fenced code blocks entirely — they do not read aloud well.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    # Strip common markdown markers.
    text = re.sub(r"[`*_#>]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    # Cap to a few sentences.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    spoken = " ".join(sentences[:_MAX_SPOKEN_SENTENCES]).strip()
    if len(spoken) > _MAX_SPOKEN_CHARS:
        spoken = spoken[:_MAX_SPOKEN_CHARS].rsplit(" ", 1)[0] + "…"
    return spoken or text[:_MAX_SPOKEN_CHARS]


class VoiceService(Protocol):
    engine_name: str
    mime_type: str

    async def speak(self, text: str) -> bytes: ...


class ElevenLabsVoiceService:
    engine_name = "elevenlabs"
    mime_type = "audio/mpeg"

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str = _DEFAULT_VOICE_ID,
        model_id: str = _DEFAULT_MODEL_ID,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._timeout = timeout_seconds

    async def speak(self, text: str) -> bytes:
        url = _ELEVENLABS_TTS_URL.format(voice_id=self._voice_id)
        headers = {
            "xi-api-key": self._api_key,  # server-side only, never sent to browser
            "accept": self.mime_type,
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                url, headers=headers, json=payload,
                params={"output_format": "mp3_44100_128"},
            )
            resp.raise_for_status()
            return resp.content


class StubVoiceService:
    """Offline TTS stand-in — returns a fixed tiny payload, makes no network
    call, costs nothing. Used in tests and whenever no ElevenLabs key is
    configured, so voice mode still returns a well-formed (silent) response
    and the UI degrades gracefully (§35)."""

    engine_name = "stub-voice"
    mime_type = "audio/mpeg"

    async def speak(self, text: str) -> bytes:
        # A minimal, valid-enough MP3-ish marker; the frontend treats an
        # empty/short clip as "no audio" and shows the text answer.
        return b"\x00"


def build_voice_service(env: dict[str, str] | None = None) -> VoiceService:
    """Real ElevenLabs service when a key is present; the stub otherwise."""
    e = env if env is not None else os.environ
    api_key = e.get("ELEVENLABS_API_KEY")
    if not api_key:
        return StubVoiceService()
    return ElevenLabsVoiceService(
        api_key,
        voice_id=e.get("ELEVENLABS_VOICE_ID") or _DEFAULT_VOICE_ID,
        model_id=e.get("ELEVENLABS_MODEL_ID") or _DEFAULT_MODEL_ID,
    )
