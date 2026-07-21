"""Stream answers through sentence chunking and TTS to the browser. Agent turns use thread-based memory."""

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import logfire
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logger import get_logger
from app.core.task_bridge import wait_and_cancel_rest
from app.services.agent import plan_turn, stream_agent_answer
from app.services.llm_stream import stream_llm_response
from app.services.rag_stream import retrieve_context, stream_grounded_answer
from app.services.realtime_tts import RealtimeApiSpeaker, TtsSpeaker
from app.services.speech_sanitizer import sanitize_for_speech
from app.services.text_chunker import TextChunker

logger = get_logger(__name__)
router = APIRouter()

# A zero-arg factory returning the answer's text-delta stream. Built once per
# turn (already bound to the prompt + any retrieved context/decision) so the
# pipeline below is agnostic to which answer source produced it.
StreamFactory = Callable[[], AsyncIterator[str]]


@dataclass
class TurnTiming:
    """Record first-text and first-audio latency from receipt of a prompt. Each metric is emitted once per turn."""

    start: float
    log_extra: dict
    _first_text_done: bool = False
    _first_audio_done: bool = False

    def _elapsed_ms(self) -> float:
        return round((time.monotonic() - self.start) * 1000, 1)

    def record_first_text(self) -> float | None:
        if self._first_text_done:
            return None
        self._first_text_done = True
        ttft_ms = self._elapsed_ms()
        # logfire captures ttft_ms as a structured, chartable attribute; the
        # JSON logger only serializes CORRELATION_FIELDS, so the number is put
        # in the message itself to stay visible there too (see logger.py).
        logfire.info("time to first text: {ttft_ms}ms", ttft_ms=ttft_ms)
        logger.info("time to first text: %sms", ttft_ms, extra=self.log_extra)
        return ttft_ms

    def record_first_audio(self) -> float | None:
        if self._first_audio_done:
            return None
        self._first_audio_done = True
        ttfa_ms = self._elapsed_ms()
        logfire.info("time to first audio: {ttfa_ms}ms", ttfa_ms=ttfa_ms)
        logger.info("time to first audio: %sms", ttfa_ms, extra=self.log_extra)
        return ttfa_ms


async def _plan_answer(prompt: str, thread_id: str, log_extra: dict) -> tuple[dict, StreamFactory]:
    """Choose the answer source for this turn and return (meta, stream_factory):
    `meta` is the browser/log-facing description of what will happen, and
    `stream_factory` produces the actual text-delta stream when called.

    For "agent"/"rag", the planning/retrieval decision happens HERE (once),
    so the meta can report the real decision and the same decision is reused
    for generation -- no double search, no double planner call.
    """
    if settings.answer_source == "agent":
        decision = await plan_turn(prompt, thread_id)
        retrieval = decision.retrieval
        meta = {
            "type": "meta",
            "answer_source": "agent",
            "thread_id": thread_id,
            "is_conversational": decision.planner.is_conversational,
            "planner_query": decision.planner.query,
            "grounded": retrieval.grounded if retrieval else None,
            "top_score": round(retrieval.top_score, 4) if retrieval else None,
            "score_floor": settings.retrieval_score_floor if retrieval else None,
            "sources": retrieval.sources if retrieval else [],
        }
        logger.info(
            "answer source: AGENT (planner + retrieval + memory)",
            extra={
                **log_extra,
                "answer_source": "agent",
                "thread_id": thread_id,
                "is_conversational": decision.planner.is_conversational,
                "grounded": retrieval.grounded if retrieval else None,
            },
        )
        return meta, lambda: stream_agent_answer(prompt, thread_id, decision)

    if settings.answer_source == "rag":
        context = await retrieve_context(prompt)
        meta = {
            "type": "meta",
            "answer_source": "rag",
            "grounded": context.grounded,
            "top_score": round(context.top_score, 4),
            "score_floor": settings.retrieval_score_floor,
            "sources": context.sources,
        }
        logger.info(
            "answer source: RAG (retrieval-grounded)",
            extra={
                **log_extra,
                "answer_source": "rag",
                "grounded": context.grounded,
                "top_score": round(context.top_score, 4),
                "score_floor": settings.retrieval_score_floor,
                "retrieved_sources": context.sources,
            },
        )
        return meta, lambda: stream_grounded_answer(prompt, context)

    meta = {"type": "meta", "answer_source": "gpt"}
    logger.info(
        "answer source: GPT (direct LLM, NO retrieval)",
        extra={**log_extra, "answer_source": "gpt"},
    )
    return meta, lambda: stream_llm_response(prompt)


async def _produce(stream_factory: StreamFactory, queue: "asyncio.Queue[str | None]") -> None:
    chunker = TextChunker()
    # aclosing() so the answer stream's finally block runs PROMPTLY when this
    # task is cancelled (Stop / disconnect), rather than whenever GC happens to
    # collect the abandoned generator. That finally is what writes a partial
    # answer to conversation memory -- see stream_agent_answer in agent.py.
    async with contextlib.aclosing(stream_factory()) as stream:
        async for delta in stream:
            for sentence in chunker.feed(delta):
                await queue.put(sentence)
    leftover = chunker.flush()
    if leftover:
        await queue.put(leftover)
    await queue.put(None)


async def _consume(
    queue: "asyncio.Queue[str | None]", speaker: TtsSpeaker, websocket: WebSocket, timing: TurnTiming
) -> None:
    while True:
        sentence = await queue.get()
        if sentence is None:
            return
        # Strip markdown/LaTeX before the sentence reaches either the
        # transcript or the TTS engine -- otherwise the TTS's "read verbatim"
        # instruction speaks raw '**'/'$'/'\frac' characters aloud.
        sentence = sanitize_for_speech(sentence)
        await websocket.send_json({"type": "text", "text": sentence})
        timing.record_first_text()  # no-op after the first sentence
        async for pcm_chunk in speaker.speak(sentence):
            timing.record_first_audio()  # no-op after the first audio byte
            await websocket.send_bytes(pcm_chunk)


async def _run_producer_consumer(
    stream_factory: StreamFactory, speaker: TtsSpeaker, websocket: WebSocket, session_id: str, timing: TurnTiming
) -> None:
    queue: "asyncio.Queue[str | None]" = asyncio.Queue()
    producer = asyncio.create_task(_produce(stream_factory, queue), name=f"voice-producer-{session_id}")
    consumer = asyncio.create_task(_consume(queue, speaker, websocket, timing), name=f"voice-consumer-{session_id}")
    await wait_and_cancel_rest((producer, consumer), asyncio.FIRST_EXCEPTION)


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    """Concurrently waits for the client to close the connection (e.g. the
    Stop button) so the pipeline can be cancelled immediately, instead of
    only being noticed the next time the server tries to send a message."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get("code", 1000))


async def _run_pipeline(
    stream_factory: StreamFactory, speaker: TtsSpeaker, websocket: WebSocket, session_id: str, timing: TurnTiming
) -> None:
    pipeline = asyncio.create_task(
        _run_producer_consumer(stream_factory, speaker, websocket, session_id, timing),
        name=f"voice-pipeline-{session_id}",
    )
    watcher = asyncio.create_task(_watch_for_disconnect(websocket), name=f"voice-watcher-{session_id}")
    await wait_and_cancel_rest((pipeline, watcher), asyncio.FIRST_COMPLETED)


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid.uuid4().hex
    log_extra = {"node_name": "voice_ws", "session_id": session_id}

    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        logger.info("client disconnected before sending a prompt", extra=log_extra)
        return

    except (json.JSONDecodeError, KeyError):
        await _safe_send_error(websocket, 'expected {"prompt": "<text>"}')
        await _safe_close(websocket)
        return

    raw_prompt = payload.get("prompt") if isinstance(payload, dict) else None
    prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
    if not prompt:
        await _safe_send_error(websocket, 'expected {"prompt": "<text>"}')
        await _safe_close(websocket)
        return

    # Caller-supplied thread continuity (see module docstring) -- a fresh
    # thread_id if the browser didn't send one (first turn of a session).
    raw_thread_id = payload.get("thread_id") if isinstance(payload, dict) else None
    thread_id = raw_thread_id.strip() if isinstance(raw_thread_id, str) and raw_thread_id.strip() else uuid.uuid4().hex

    logger.info("session started", extra={**log_extra, "thread_id": thread_id})
    timing = TurnTiming(start=time.monotonic(), log_extra=log_extra)
    # Wraps the whole turn (planning through the producer/consumer pipeline)
    # in one named span, so a Logfire trace shows "everything that happened
    # answering this question" grouped together, not just the individual
    # service-level spans nested inside app/services/*.py in isolation.
    with logfire.span(
        "voice turn", session_id=session_id, thread_id=thread_id, answer_source=settings.answer_source
    ):
        try:
            # Pick + announce the answer source before any answer text streams,
            # so which path ran (and its decision) is visible in the browser and
            # logs rather than being invisible mid-stream.
            meta, stream_factory = await _plan_answer(prompt, thread_id, log_extra)
            await websocket.send_json(meta)

            async with RealtimeApiSpeaker() as speaker:
                await _run_pipeline(stream_factory, speaker, websocket, session_id, timing)
            await websocket.send_json({"type": "done"})
            logger.info("session completed", extra=log_extra)
        except WebSocketDisconnect:
            logger.info("client disconnected mid-session", extra=log_extra)
        except Exception as exc:
            logger.error("session failed: %s", exc, extra=log_extra, exc_info=True)
            await _safe_send_error(websocket, str(exc))
        finally:
            await _safe_close(websocket)


async def _safe_send_error(websocket: WebSocket, detail: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "detail": detail})


async def _safe_close(websocket: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await websocket.close()
