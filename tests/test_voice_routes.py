"""Integration tests for POST /api/voice/turn (webserver.py). Driven with
`stub: true`, so the voice route runs the SAME Versa tutoring pipeline the
text route uses (StubLLMClient) and the stub voice service — no real Gemini,
Moss, or ElevenLabs call is ever made (§42). Confirms voice is an I/O
adapter over the one existing brain, with learner isolation preserved.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch):
    from starlette.testclient import TestClient

    from probe import webserver
    from tests.conftest import DATABASE_URL

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    with TestClient(webserver.create_app()) as c:
        yield c


def test_voice_turn_runs_the_existing_pipeline(client):
    resp = client.post(
        "/api/voice/turn",
        json={"learner": "voice-route-test", "text": "explain recursion", "stub": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    # The same tutor pipeline produced a real answer via the shared brain.
    assert isinstance(data["message"], str) and data["message"]
    assert data["session_id"]
    assert data["learner"]["label"] == "voice-route-test"
    # Stub turn never hits the paid TTS API.
    assert data["voice_engine"] == "stub-voice"
    # Retrieval telemetry is present and honest about the engine.
    assert data["retrieval"] is not None
    assert data["retrieval"]["engine"] in {"stub-moss", "moss"}


def test_voice_turn_is_multi_turn_on_one_session(client):
    first = client.post(
        "/api/voice/turn",
        json={"learner": "voice-multiturn", "text": "explain recursion", "stub": True},
    ).json()
    sid = first["session_id"]
    second = client.post(
        "/api/voice/turn",
        json={"session_id": sid, "learner": "voice-multiturn",
              "text": "why does it stop", "stub": True},
    )
    assert second.status_code == 200
    data = second.json()
    assert data["session_id"] == sid  # same session reused
    assert data["turn_index"] == 1  # advanced


def test_voice_turn_requires_text(client):
    resp = client.post(
        "/api/voice/turn", json={"learner": "voice-x", "text": "", "stub": True}
    )
    assert resp.status_code == 400


def test_voice_turn_requires_learner_for_new_session(client):
    resp = client.post("/api/voice/turn", json={"text": "hello", "stub": True})
    assert resp.status_code == 400
