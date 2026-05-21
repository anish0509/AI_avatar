import asyncio
import json
from unittest.mock import patch

import pytest

from app.api.transcription_routes import transcribe_ws
from app.services.realtime_stt import StreamingTranscriber, TranscriptEvent


class FakeBrowserSocket:
    """Stands in for Starlette's WebSocket, browser side. Mirrors
    test_orchestration.py's FakeBrowserSocket, adapted for a socket that
    carries binary audio frames plus one JSON control message rather than
    one JSON prompt plus binary audio out."""

    def __init__(
        self,
        messages: list[dict] | None = None,
        disconnect_event: "asyncio.Event | None" = None,
    ) -> None:
        self._messages = list(messages or [])
        self._disconnect_event = disconnect_event
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self) -> None:
        return None

    async def receive(self) -> dict:
        if self._messages:
            return self._messages.pop(0)
        if self._disconnect_event is not None:
            await self._disconnect_event.wait()
            return {"type": "websocket.disconnect", "code": 1000}
        await asyncio.Event().wait()  # browser still connected, nothing more sent

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class FakeTranscriber(StreamingTranscriber):
    """Concrete lightweight StreamingTranscriber stub (not a Mock of
    RealtimeApiTranscriber), mirrors test_orchestration.py's FakeSpeaker."""

    def __init__(
        self,
        canned_events: list[TranscriptEvent] | None = None,
        fail_after_events: bool = False,
        hang_after_events: bool = False,
        signal_on_hang: "asyncio.Event | None" = None,
    ) -> None:
        self.sent_audio: list[bytes] = []
        self._canned_events = canned_events or []
        self._fail_after_events = fail_after_events
        self._hang_after_events = hang_after_events
        self._signal_on_hang = signal_on_hang

    async def send_audio(self, pcm16: bytes) -> None:
        self.sent_audio.append(pcm16)

    async def events(self):
        for event in self._canned_events:
            yield event
        if self._fail_after_events:
            raise RuntimeError("transcriber boom")
        if self._hang_after_events:
            if self._signal_on_hang is not None:
                self._signal_on_hang.set()
            await asyncio.Event().wait()  # simulates "still connected" until cancelled


def _audio_message(data: bytes) -> dict:
    return {"type": "websocket.receive", "bytes": data}


def _stop_message() -> dict:
    return {"type": "websocket.receive", "text": json.dumps({"type": "stop"})}


def _patched_transcriber(transcriber: FakeTranscriber):
    return patch("app.api.transcription_routes.RealtimeApiTranscriber", lambda *a, **kw: transcriber)


@pytest.mark.asyncio
async def test_relays_audio_then_streams_deltas_and_final():
    fake_ws = FakeBrowserSocket(messages=[_audio_message(b"chunk1"), _audio_message(b"chunk2"), _stop_message()])
    transcriber = FakeTranscriber(
        canned_events=[
            TranscriptEvent(kind="delta", text="Hello"),
            TranscriptEvent(kind="delta", text=" world"),
            TranscriptEvent(kind="final", text="Hello world"),
        ]
    )

    with _patched_transcriber(transcriber):
        await asyncio.wait_for(transcribe_ws(fake_ws), timeout=2)

    assert transcriber.sent_audio == [b"chunk1", b"chunk2"]
    assert fake_ws.sent == [
        {"type": "delta", "text": "Hello"},
        {"type": "delta", "text": " world"},
        {"type": "final", "text": "Hello world"},
        {"type": "done"},
    ]
    assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_transcriber_error_sends_error_frame_and_cancels_uplink():
    fake_ws = FakeBrowserSocket(messages=[_audio_message(b"chunk1")], disconnect_event=asyncio.Event())
    transcriber = FakeTranscriber(canned_events=[TranscriptEvent(kind="delta", text="Hi")], fail_after_events=True)

    with _patched_transcriber(transcriber):
        await asyncio.wait_for(transcribe_ws(fake_ws), timeout=2)

    assert fake_ws.sent == [
        {"type": "delta", "text": "Hi"},
        {"type": "error", "detail": "transcriber boom"},
    ]
    assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_disconnect_before_any_audio_is_handled_cleanly():
    disconnect_event = asyncio.Event()
    disconnect_event.set()
    fake_ws = FakeBrowserSocket(disconnect_event=disconnect_event)
    transcriber = FakeTranscriber(hang_after_events=True)

    with _patched_transcriber(transcriber):
        await asyncio.wait_for(transcribe_ws(fake_ws), timeout=2)

    assert fake_ws.sent == []  # no "done" -- session never completed normally
    assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_stop_with_no_audio_ends_session_cleanly():
    fake_ws = FakeBrowserSocket(messages=[_stop_message()])
    transcriber = FakeTranscriber(hang_after_events=True)

    with _patched_transcriber(transcriber):
        await asyncio.wait_for(transcribe_ws(fake_ws), timeout=2)

    assert transcriber.sent_audio == []
    assert fake_ws.sent == [{"type": "done"}]
    assert fake_ws.closed is True
