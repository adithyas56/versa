"""Unit tests for the LiveKit token/route wiring (webserver.py) and the
voice-agent room parsing (livekit_agent.agent). Pure/offline — no DB, no
LiveKit connection, no audio.
"""

from probe import webserver
from probe.livekit_agent import agent


def test_sanitize_learner_keeps_safe_chars_and_strips_others():
    assert webserver._sanitize_learner("demo-learner-a") == "demo-learner-a"
    assert webserver._sanitize_learner("weird name!!") == "weird-name"
    assert webserver._sanitize_learner("a__b") == "a--b"  # no "__" leaks into room name
    assert webserver._sanitize_learner("") == "student"
    assert webserver._sanitize_learner("!!!") == "student"


def test_create_app_registers_voice_and_token_routes():
    app = webserver.create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/voice" in paths
    assert "/api/livekit/token" in paths


def test_agent_parses_learner_from_room_name():
    assert agent.learner_from_room("versa__demo-learner-a__ab12cd") == "demo-learner-a"
    assert agent.learner_from_room("versa__student-b__ff00") == "student-b"
    # malformed / manual rooms fall back to a demo learner, never crash
    assert agent.learner_from_room("some-random-room") == "demo-learner-a"
    assert agent.learner_from_room("") == "demo-learner-a"


def test_room_name_roundtrips_through_sanitize():
    safe = webserver._sanitize_learner("Priya Kumar")
    room = f"versa__{safe}__deadbeef"
    assert agent.learner_from_room(room) == safe
