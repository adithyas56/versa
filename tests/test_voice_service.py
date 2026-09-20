"""Unit tests for the voice TTS layer (voice.py). Pure/offline — the stub
service makes no network call, and `build_voice_service` selection is
checked without ever invoking the real ElevenLabs API.
"""

import pytest

from probe.voice import (
    ElevenLabsVoiceService,
    StubVoiceService,
    build_voice_service,
    speakable_text,
)


def test_speakable_strips_markdown_and_code():
    raw = "# Heading\nHere is **bold** and `code`.\n```python\nprint(1)\n```\n- a bullet"
    out = speakable_text(raw)
    assert "#" not in out
    assert "**" not in out
    assert "`" not in out
    assert "print(1)" not in out  # fenced code dropped
    assert "bold" in out


def test_speakable_caps_length_to_a_few_sentences():
    raw = " ".join(f"Sentence number {i} explains a point." for i in range(20))
    out = speakable_text(raw)
    # Far shorter than the input; capped near the sentence/char limit.
    assert len(out) <= 620
    assert out.count(".") <= 5


def test_speakable_never_empty_for_real_text():
    assert speakable_text("Recursion is a function calling itself.")


@pytest.mark.asyncio
async def test_stub_voice_makes_no_network_call_and_returns_bytes():
    svc = StubVoiceService()
    audio = await svc.speak("Hello, I am Versa.")
    assert isinstance(audio, bytes)
    assert svc.engine_name == "stub-voice"


def test_build_voice_service_returns_stub_without_key():
    svc = build_voice_service({})
    assert isinstance(svc, StubVoiceService)


def test_build_voice_service_returns_real_with_key():
    svc = build_voice_service({"ELEVENLABS_API_KEY": "xi-fake"})
    assert isinstance(svc, ElevenLabsVoiceService)
    assert svc.engine_name == "elevenlabs"
    assert svc.mime_type == "audio/mpeg"


def test_real_voice_service_honors_env_overrides():
    svc = build_voice_service(
        {
            "ELEVENLABS_API_KEY": "xi-fake",
            "ELEVENLABS_VOICE_ID": "custom-voice",
            "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5",
        }
    )
    assert svc._voice_id == "custom-voice"
    assert svc._model_id == "eleven_flash_v2_5"
