"""Regression test for the incident recorded in session_builder.py's
own docstring: before 2026-09-15, `probe serve` (webserver.py) built
its `SessionLoop` by hand, with none of the interaction pipeline wired
in -- no claims, no stated preferences, no reference bindings, no
history block. Every session run through the actual browser-facing web
UI ran fully un-personalized, silently, with no error and no test
catching it (the existing webserver test suite drove `SessionLoop.
handle_turn` directly, never the real `/api/session/*/turn` HTTP route
with a real interaction pipeline attached).

This test drives the REAL HTTP routes (`httpx.AsyncClient` over
`webserver.create_app()`'s actual ASGI app, not a shortcut around it)
through one full session and confirms -- by reading `node_calls` back,
not by checking a store has rows -- that all four blocks actually reach
`FinalAnswer`'s own prompt input: `claim_constraints_block`,
`structural_requirement`, `reference_bindings_block`, and
`learner_history_block`. That distinction (reaches the prompt, not
just "the store has a row") is the whole lesson from the incident."""
import json

import httpx
import pytest

from uuid import UUID

from probe import webserver
from probe.audit import NodeCallStore
from probe.demo_fixture import CONCRETE_PORTRAIT_LABEL, seed_demo_fixture
from probe.interactions import InteractionStore, ReferenceBindingStore
from probe.llm import StubLLMClient

from tests.conftest import DATABASE_URL

QUESTION = "Explain how binary search works, including the swap step."

_STATED_PREFERENCE_CANNED = json.dumps({
    "has_preference": True,
    "stated_preference": "always wants real-world analogies",
    "label": "wants_analogies",
})


def _last_sse_event(response_text: str) -> dict:
    lines = [line for line in response_text.splitlines() if line.startswith("data: ")]
    return json.loads(lines[-1][len("data: "):])


@pytest.mark.asyncio(loop_scope="session")
async def test_probe_serve_wires_claims_stated_preferences_reference_bindings_and_history(
    clean_pool, monkeypatch,
):
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    # A custom canned StubLLMClient so ClassifyStatedPreference reports a
    # real preference -- webserver's own JSON API has no way to inject
    # canned responses, so this is the one piece of test-only wiring:
    # everything downstream of it (interaction recording, background
    # classification, the actual HTTP routes) runs for real.
    monkeypatch.setattr(
        webserver, "StubLLMClient",
        lambda: StubLLMClient(canned={"CLASSIFY:STATED_PREFERENCE": _STATED_PREFERENCE_CANNED}),
    )

    portraits = await seed_demo_fixture(clean_pool)
    concrete = portraits["concrete"]
    webserver._state.pool = clean_pool
    created_sessions: list[str] = []
    try:
        transport = httpx.ASGITransport(app=webserver.create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Session A: establishes the stated preference and the
            # interaction that a reference binding will point at.
            made_a = await client.post(
                "/api/session", json={"learner": CONCRETE_PORTRAIT_LABEL, "stub": True}
            )
            assert made_a.status_code == 200
            session_a = made_a.json()["session_id"]
            created_sessions.append(session_a)

            turn_a = await client.post(f"/api/session/{session_a}/turn", json={"text": QUESTION})
            assert turn_a.status_code == 200
            event_a = _last_sse_event(turn_a.text)
            assert event_a["phase"] == "done"

            # The stated-preference classification runs as a detached
            # background task (loop.py's _fire_background) -- exactly
            # like a real browser turn, nothing here awaits it by
            # default. Wait for it deterministically rather than
            # guessing at a sleep duration (SessionLoop.
            # wait_for_background_tasks's own documented purpose).
            await webserver._SESSIONS[session_a].loop.wait_for_background_tasks()

            interactions = await InteractionStore(clean_pool).get_recent_for_learner(concrete.id, limit=5)
            assert len(interactions) == 1
            interaction_a = interactions[0]

            # Seed a reference binding pointing at that real interaction
            # -- same "hand-authored, direct construction" precedent
            # demo_fixture.py already uses for claims: the classifier
            # that WRITES bindings (ClassifyReferenceResolution) is a
            # separate, already-tested mechanism; what's under test here
            # is only whether the READ side (assemble_reference_bindings_block)
            # reaches FinalAnswer through the real web path.
            await ReferenceBindingStore(clean_pool).record_resolution(
                learner_id=concrete.id, reference_text="the swap",
                resolved_to="the two-element exchange that repositions a value during binary search",
                evidence_interaction_id=interaction_a.id,
                classifier_version="test",
            )

            # Session B: the SAME question text (StubEmbeddingClient
            # embeds identical text identically, so retrieval finds
            # session A's interaction for real, not by construction)
            # and it contains "the swap", the reference binding's exact-
            # match phrase.
            made_b = await client.post(
                "/api/session", json={"learner": CONCRETE_PORTRAIT_LABEL, "stub": True}
            )
            assert made_b.status_code == 200
            session_b = made_b.json()["session_id"]
            created_sessions.append(session_b)

            turn_b = await client.post(f"/api/session/{session_b}/turn", json={"text": QUESTION})
            assert turn_b.status_code == 200
            event_b = _last_sse_event(turn_b.text)
            assert event_b["phase"] == "done"

            node_calls = NodeCallStore(clean_pool)
            call = await node_calls.get_call_for_turn(UUID(session_b), 0, "FinalAnswer")
            assert call is not None
            inputs = call.input_json

            # Not "the store has a row" -- these are FinalAnswer's own
            # recorded prompt inputs for this turn (CLAUDE.md invariant
            # 2's audit trail), read back after the fact.
            assert "concrete worked example" in inputs["claim_constraints_block"]
            assert "real-world analogy" in inputs["claim_constraints_block"]
            assert "Include a real-world analogy" in inputs["structural_requirement"]
            assert "the swap" in inputs["reference_bindings_block"]
            assert "two-element exchange" in inputs["reference_bindings_block"]
            assert inputs["learner_history_block"]  # non-empty: session A's turn was retrieved
    finally:
        for sid in created_sessions:
            webserver._SESSIONS.pop(sid, None)
        webserver._state.pool = None
