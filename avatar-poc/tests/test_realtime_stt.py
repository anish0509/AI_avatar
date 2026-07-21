import base64
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings
from app.services.realtime_stt import RealtimeApiTranscriber, build_session_update


class FakeWebSocket:
    """Stands in for a real websockets connection: records what we send,
    and replays a canned queue of incoming events on iteration. Mirrors
    test_realtime_tts.py's FakeWebSocket."""

    def __init__(self, incoming: list[dict]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        if not self._incoming:
            raise StopAsyncIteration
        return json.dumps(self._incoming.pop(0))

    async def close(self) -> None:
        self.closed = True


def _delta_event(text: str) -> dict:
    return {"type": "conversation.item.input_audio_transcription.delta", "delta": text}


def _completed_event(text: str) -> dict:
    return {"type": "conversation.item.input_audio_transcription.completed", "transcript": text}


def _patched_connect(fake_ws: FakeWebSocket):
    return patch("app.services.realtime_stt.websockets.connect", AsyncMock(return_value=fake_ws))


SESSION_SETUP_EVENTS = [{"type": "session.created"}, {"type": "session.updated"}]


@pytest.mark.asyncio
async def test_connect_sends_session_update_and_waits_for_confirmation():
    fake_ws = FakeWebSocket(incoming=list(SESSION_SETUP_EVENTS))
    with _patched_connect(fake_ws):
        async with RealtimeApiTranscriber():
            pass

    sent = fake_ws.sent[0]
    assert sent["type"] == "session.update"
    assert sent["session"]["type"] == "transcription"
    audio_input = sent["session"]["audio"]["input"]
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["turn_detection"] == {"type": "server_vad"}
    # language is pinned by default so the model doesn't mis-detect short
    # segments into other scripts (the JavaScript -> Urdu regression).
    assert audio_input["transcription"]["language"] == settings.realtime_transcribe_language
    assert fake_ws.closed is True




def test_build_session_update_omits_all_knobs_by_default(monkeypatch):
    # With every mitigation knob at its empty/None default, the payload must
    # be exactly the pre-Bug-4 default: plain server_vad, no noise_reduction,
    # no prompt, no VAD overrides.
    monkeypatch.setattr(settings, "realtime_noise_reduction", "")
    monkeypatch.setattr(settings, "realtime_vad_threshold", None)
    monkeypatch.setattr(settings, "realtime_prefix_padding_ms", None)
    monkeypatch.setattr(settings, "realtime_silence_duration_ms", None)
    monkeypatch.setattr(settings, "realtime_transcribe_prompt", "")

    audio_input = build_session_update("gpt-4o-transcribe")["session"]["audio"]["input"]

    assert audio_input["turn_detection"] == {"type": "server_vad"}
    assert "noise_reduction" not in audio_input
    assert "prompt" not in audio_input["transcription"]


def test_build_session_update_includes_knobs_when_set(monkeypatch):
    monkeypatch.setattr(settings, "realtime_noise_reduction", "far_field")
    monkeypatch.setattr(settings, "realtime_vad_threshold", 0.6)
    monkeypatch.setattr(settings, "realtime_prefix_padding_ms", 300)
    monkeypatch.setattr(settings, "realtime_silence_duration_ms", 500)
    monkeypatch.setattr(settings, "realtime_transcribe_prompt", "Short English replies.")

    audio_input = build_session_update("gpt-4o-transcribe")["session"]["audio"]["input"]

    assert audio_input["noise_reduction"] == {"type": "far_field"}
    assert audio_input["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.6,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
    }
    assert audio_input["transcription"]["prompt"] == "Short English replies."


def test_noise_reduction_defaults_to_far_field():
    from app.core.config import Settings

    assert Settings.model_fields["realtime_noise_reduction"].default == "far_field"


def test_empty_env_string_coerced_to_none_for_numeric_knobs():
    from app.core.config import Settings

    # A blank line in .env (VAR=) must be treated as "unset" (field omitted),
    # not a validation error on parsing "" to float/int.
    s = Settings(
        realtime_vad_threshold="",
        realtime_prefix_padding_ms="",
        realtime_silence_duration_ms="",
    )
    assert s.realtime_vad_threshold is None
    assert s.realtime_prefix_padding_ms is None
    assert s.realtime_silence_duration_ms is None


@pytest.mark.asyncio
async def test_send_audio_sends_base64_encoded_append():
    fake_ws = FakeWebSocket(incoming=list(SESSION_SETUP_EVENTS))
    with _patched_connect(fake_ws):
        async with RealtimeApiTranscriber() as transcriber:
            await transcriber.send_audio(b"raw-pcm-bytes")

    append_msgs = [m for m in fake_ws.sent if m["type"] == "input_audio_buffer.append"]
    assert len(append_msgs) == 1
    assert append_msgs[0]["audio"] == base64.b64encode(b"raw-pcm-bytes").decode()


@pytest.mark.asyncio
async def test_events_yields_delta_then_final_in_order():
    fake_ws = FakeWebSocket(
        incoming=[
            *SESSION_SETUP_EVENTS,
            _delta_event("Hello"),
            _delta_event(" world"),
            _completed_event("Hello world"),
        ]
    )
    with _patched_connect(fake_ws):
        async with RealtimeApiTranscriber() as transcriber:
            events = [e async for e in transcriber.events()]

    assert [(e.kind, e.text) for e in events] == [
        ("delta", "Hello"),
        ("delta", " world"),
        ("final", "Hello world"),
    ]


@pytest.mark.asyncio
async def test_events_raises_on_error_event():
    fake_ws = FakeWebSocket(incoming=[*SESSION_SETUP_EVENTS, {"type": "error", "error": {"message": "boom"}}])
    with _patched_connect(fake_ws):
        async with RealtimeApiTranscriber() as transcriber:
            with pytest.raises(RuntimeError, match="boom"):
                async for _ in transcriber.events():
                    pass


@pytest.mark.asyncio
async def test_connect_raises_on_error_event_during_session_setup():
    fake_ws = FakeWebSocket(incoming=[{"type": "error", "error": {"message": "bad session"}}])
    with _patched_connect(fake_ws):
        with pytest.raises(RuntimeError, match="bad session"):
            async with RealtimeApiTranscriber():
                pass


@pytest.mark.asyncio
async def test_send_audio_outside_context_manager_raises():
    transcriber = RealtimeApiTranscriber()
    with pytest.raises(RuntimeError, match="outside"):
        await transcriber.send_audio(b"x")


@pytest.mark.asyncio
async def test_events_outside_context_manager_raises():
    transcriber = RealtimeApiTranscriber()
    with pytest.raises(RuntimeError, match="outside"):
        async for _ in transcriber.events():
            pass


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr("app.services.realtime_stt.settings.openai_api_key", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        async with RealtimeApiTranscriber():
            pass
