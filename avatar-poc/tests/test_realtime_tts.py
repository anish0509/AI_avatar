import base64
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.realtime_tts import RealtimeApiSpeaker


class FakeWebSocket:
    """Stands in for a real websockets connection: records what we send,
    and replays a canned queue of incoming events on iteration."""

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


def _audio_event(payload: bytes) -> dict:
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(payload).decode()}


def _patched_connect(fake_ws: FakeWebSocket):
    return patch("app.services.realtime_tts.websockets.connect", AsyncMock(return_value=fake_ws))


@pytest.mark.asyncio
async def test_connect_sends_session_update_and_waits_for_confirmation():
    fake_ws = FakeWebSocket(incoming=[{"type": "session.updated"}])
    with _patched_connect(fake_ws):
        async with RealtimeApiSpeaker():
            pass

    assert fake_ws.sent[0]["type"] == "session.update"
    assert fake_ws.sent[0]["session"]["output_modalities"] == ["audio"]
    assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_speak_yields_audio_chunks_in_order():
    fake_ws = FakeWebSocket(
        incoming=[
            {"type": "session.updated"},
            _audio_event(b"chunk-one"),
            _audio_event(b"chunk-two"),
            {"type": "response.done"},
        ]
    )
    with _patched_connect(fake_ws):
        async with RealtimeApiSpeaker() as speaker:
            chunks = [c async for c in speaker.speak("Hello there.")]

    assert chunks == [b"chunk-one", b"chunk-two"]
    sent_types = [m["type"] for m in fake_ws.sent]
    assert sent_types == ["session.update", "conversation.item.create", "response.create"]
    assert fake_ws.sent[1]["item"]["content"][0]["text"] == "Hello there."


@pytest.mark.asyncio
async def test_speak_can_be_called_multiple_times_on_one_connection():
    fake_ws = FakeWebSocket(
        incoming=[
            {"type": "session.updated"},
            _audio_event(b"first"),
            {"type": "response.done"},
            _audio_event(b"second"),
            {"type": "response.done"},
        ]
    )
    with _patched_connect(fake_ws) as mock_connect:
        async with RealtimeApiSpeaker() as speaker:
            first_chunks = [c async for c in speaker.speak("First sentence.")]
            second_chunks = [c async for c in speaker.speak("Second sentence.")]

    assert first_chunks == [b"first"]
    assert second_chunks == [b"second"]
    mock_connect.assert_awaited_once()  # only ONE connection for both sentences

    item_creates = [m for m in fake_ws.sent if m["type"] == "conversation.item.create"]
    assert len(item_creates) == 2
    assert item_creates[0]["item"]["content"][0]["text"] == "First sentence."
    assert item_creates[1]["item"]["content"][0]["text"] == "Second sentence."


@pytest.mark.asyncio
async def test_speak_raises_on_error_event():
    fake_ws = FakeWebSocket(
        incoming=[
            {"type": "session.updated"},
            {"type": "error", "error": {"message": "boom"}},
        ]
    )
    with _patched_connect(fake_ws):
        async with RealtimeApiSpeaker() as speaker:
            with pytest.raises(RuntimeError, match="boom"):
                async for _ in speaker.speak("This will fail."):
                    pass


@pytest.mark.asyncio
async def test_connect_raises_on_error_event_during_session_setup():
    fake_ws = FakeWebSocket(incoming=[{"type": "error", "error": {"message": "bad session"}}])
    with _patched_connect(fake_ws):
        with pytest.raises(RuntimeError, match="bad session"):
            async with RealtimeApiSpeaker():
                pass


@pytest.mark.asyncio
async def test_speak_outside_context_manager_raises():
    speaker = RealtimeApiSpeaker()
    with pytest.raises(RuntimeError, match="outside"):
        async for _ in speaker.speak("Hello."):
            pass


@pytest.mark.asyncio
async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr("app.services.realtime_tts.settings.openai_api_key", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        async with RealtimeApiSpeaker():
            pass
