"""Speaks text aloud via OpenAI's Realtime API, behind a TtsSpeaker
interface (mirrors the ASRProvider pattern from the ingestion project) so a
different backend (e.g. gpt-4o-mini-tts) is a one-file swap if ever needed
-- see the Decision Log in implementation.md.

One connection is opened once (in __aenter__) and reused across multiple
speak() calls, sentence by sentence, rather than reconnecting per sentence
-- a single LLM reply is usually several sentences in a row, and
reconnecting for each one would add real per-sentence latency.
"""

import base64
import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import websockets

from app.core.config import settings
from app.core.logger import get_logger
from app.core.timing import step_timer

logger = get_logger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
SAMPLE_RATE = 24000

VERBATIM_INSTRUCTIONS = (
    "You are a text-to-speech engine. Read the user's message aloud, "
    "verbatim, exactly as written. Do not respond, comment, add anything, "
    "or ask questions -- just read the text aloud."
)


class TtsSpeaker(ABC):
    async def __aenter__(self) -> "TtsSpeaker":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    @abstractmethod
    def speak(self, text: str) -> AsyncIterator[bytes]:
        """Speak one piece of text; yields raw PCM16 audio chunks as they arrive."""


class RealtimeApiSpeaker(TtsSpeaker):
    def __init__(self, model: str | None = None, voice: str | None = None) -> None:
        self._model = model or settings.realtime_model
        self._voice = voice or settings.realtime_voice
        self._ws = None

    async def __aenter__(self) -> "RealtimeApiSpeaker":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

        url = f"{REALTIME_URL}?model={self._model}"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        # Timed because a NEW connection is opened per turn (see P2) and this
        # sits directly on the critical path, in front of generation. Measured
        # at ~1.4s, previously visible only as an unexplained gap between the
        # retrieval and LLM spans in a Logfire trace.
        try:
            with step_timer("realtime tts session setup", "realtime_tts") as timer:
                self._ws = await websockets.connect(url, additional_headers=headers)
                timer.mark("ws_connect")

                await self._ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "model": self._model,
                                "output_modalities": ["audio"],
                                "instructions": VERBATIM_INSTRUCTIONS,
                                "audio": {
                                    "output": {
                                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                        "voice": self._voice,
                                    }
                                },
                            },
                        }
                    )
                )
                # Wait for confirmation before any speak() call sends text, so we
                # never race the session configuration.
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

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        if self._ws is None:
            raise RuntimeError("RealtimeApiSpeaker used outside 'async with' -- no open connection")

        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
        await self._ws.send(json.dumps({"type": "response.create", "response": {"output_modalities": ["audio"]}}))

        async for raw in self._ws:
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "response.output_audio.delta":
                yield base64.b64decode(event["delta"])
            elif event_type == "error":
                raise RuntimeError(f"Realtime API error: {event}")
            elif event_type == "response.done":
                return

    async def _wait_for_event(self, expected_type: str) -> dict:
        async for raw in self._ws:
            event = json.loads(raw)
            if event.get("type") == "error":
                raise RuntimeError(f"Realtime API error: {event}")
            if event.get("type") == expected_type:
                return event
        raise RuntimeError(f"Connection closed before receiving '{expected_type}'")
