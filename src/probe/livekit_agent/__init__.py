"""LiveKit real-time voice agent for Versa.

This package is the real-time voice transport/agent layer (VERSA voice-
conversation context). It is deliberately thin: LiveKit handles the audio
room, STT, TTS, VAD, turn detection, and barge-in; the *answer* on every
turn comes from the existing Versa HTTP pipeline (`POST /api/voice/turn`),
so Moss retrieval + Gemini + ambiguity all run through the one existing
tutor brain. See `agent.py`.
"""
