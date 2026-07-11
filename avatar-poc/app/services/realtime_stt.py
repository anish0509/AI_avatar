"""Transcribes user speech to text AS IT ARRIVES via OpenAI's Realtime API,
behind a StreamingTranscriber interface (mirrors the TtsSpeaker pattern in
realtime_tts.py, which itself mirrors the ingestion project's ASRProvider)
so a different streaming backend (e.g. a local VOSK recognizer) is a
one-file swap if ever needed.

This is the streaming counterpart to transcription.py's transcribe_audio()
(POST /transcribe, whole-clip-then-transcribe) -- that endpoint is left
untouched as a non-streaming fallback. This module is consumed by the new
WebSocket route (POST /ws/transcribe) so the browser can show text while
the user is still speaking, instead of only after they stop.

One connection is opened once (in __aenter__) and audio is streamed to it
continuously via send_audio(); OpenAI's server-side VAD decides where
utterance boundaries are and emits delta (partial) and completed (final)
transcript events, consumed via events().
"""

import base64
import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

import websockets

from app.core.config import settings
from app.core.logger import get_logger
from app.core.timing import step_timer

logger = get_logger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
INPUT_SAMPLE_RATE = 24000


@dataclass
class TranscriptEvent:
    kind: Literal["delta", "final"]
    text: str


def build_session_update(model: str) -> dict:
    """Builds the transcription `session.update` payload from settings.

    Kept as a module-level pure function (reads `settings`, no I/O) so the
    Bug-4 mitigation knobs can be unit-tested without opening a websocket.
    Every optional knob is OMITTED when empty/None, so the payload is
    byte-for-byte the previous default (server_vad, API defaults) until a
    value is explicitly configured -- see bug-report-streaming-stt.md.
    """
    transcription: dict = {"model": model}
    # Empty language -> omit the field and let the model auto-detect
    # (needed for Hindi/Hinglish); a set language biases decoding and
    # curbs spurious language switches on short segments.
    if settings.realtime_transcribe_language:
        transcription["language"] = settings.realtime_transcribe_language
    if settings.realtime_transcribe_prompt:
        transcription["prompt"] = settings.realtime_transcribe_prompt

    turn_detection: dict = {"type": "server_vad"}
    if settings.realtime_vad_threshold is not None:
        turn_detection["threshold"] = settings.realtime_vad_threshold
    if settings.realtime_prefix_padding_ms is not None:
        turn_detection["prefix_padding_ms"] = settings.realtime_prefix_padding_ms
    if settings.realtime_silence_duration_ms is not None:
        turn_detection["silence_duration_ms"] = settings.realtime_silence_duration_ms

    audio_input: dict = {
        "format": {"type": "audio/pcm", "rate": INPUT_SAMPLE_RATE},
        "transcription": transcription,
        "turn_detection": turn_detection,
    }
    if settings.realtime_noise_reduction:
        audio_input["noise_reduction"] = {"type": settings.realtime_noise_reduction}

    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {"input": audio_input},
        },
    }


class StreamingTranscriber(ABC):
    async def __aenter__(self) -> "StreamingTranscriber":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    @abstractmethod
    async def send_audio(self, pcm16: bytes) -> None:
        """Push one chunk of raw PCM16 mono audio (at INPUT_SAMPLE_RATE)."""

    @abstractmethod
    def events(self) -> AsyncIterator[TranscriptEvent]:
        """Yields transcript events as they arrive: 'delta' for incremental
        partials, 'final' once an utterance boundary is reached."""


class RealtimeApiTranscriber(StreamingTranscriber):
    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.realtime_transcribe_model
        self._ws = None

    async def __aenter__(self) -> "RealtimeApiTranscriber":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

        url = f"{REALTIME_URL}?intent=transcription"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        # Timed for the same reason as the TTS socket: a new connection per
        # recording, on the path between pressing the mic button and the first
        # transcript delta appearing.
        try:
            with step_timer("realtime stt session setup", "realtime_stt") as timer:
                self._ws = await websockets.connect(url, additional_headers=headers)
                timer.mark("ws_connect")

                await self._ws.send(json.dumps(build_session_update(self._model)))
                # Wait for confirmation before any send_audio() call streams bytes,
                # so we never race the session configuration (same discipline as
                # RealtimeApiSpeaker.__aenter__). Note: despite the session object
                # itself being typed "transcription", the confirmation event name
                # is still the generic "session.updated" (verified against the
                # real API -- NOT "transcription_session.updated"); an initial
                # "session.created" arrives first and is skipped by this loop.
                await self._wait_for_event("session.updated")
                timer.mark("session_configure")
                timer.log()
        except BaseException:
            # __aexit__ is not called when __aenter__ fails or is cancelled.
            with contextlib.suppress(Exception):
                await self.__aexit__()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def send_audio(self, pcm16: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("RealtimeApiTranscriber used outside 'async with' -- no open connection")

        await self._ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16).decode(),
                }
            )
        )

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        if self._ws is None:
            raise RuntimeError("RealtimeApiTranscriber used outside 'async with' -- no open connection")

        async for raw in self._ws:
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "conversation.item.input_audio_transcription.delta":
                yield TranscriptEvent(kind="delta", text=event.get("delta", ""))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                yield TranscriptEvent(kind="final", text=event.get("transcript", ""))
            elif event_type == "error":
                raise RuntimeError(f"Realtime API error: {event}")

    async def _wait_for_event(self, expected_type: str) -> dict:
        async for raw in self._ws:
            event = json.loads(raw)
            if event.get("type") == "error":
                raise RuntimeError(f"Realtime API error: {event}")
            if event.get("type") == expected_type:
                return event
        raise RuntimeError(f"Connection closed before receiving '{expected_type}'")
