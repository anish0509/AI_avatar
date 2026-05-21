"""HeyGen LiveAvatar route: one persistent WebSocket per browser tab that
answers however many questions are asked over it, reusing the same HeyGen
session and TTS connection across them instead of rebuilding both per turn.

Previously (through 2026-07-27) this was one WebSocket PER QUESTION: HeyGen
session setup (~4.4s, measured) and the TTS socket (~2.3s, measured) were both
paid on every single Ask, even though nothing about either one is specific to
one question. Reused here: `/ws/avatar` now stays open across turns, building
the `LiveAvatarRenderer` and `RealtimeApiSpeaker` once and holding them in
`AvatarConnectionState` for as long as the connection lives. HeyGen setup is
still started concurrently with answer generation (see `_start_session_setup`)
exactly as before -- that concurrency is unaffected, it just now only has to
happen once per connection instead of once per question.

This requires `HEYGEN_IS_SANDBOX=false`: a sandbox session is hard-capped at
60 seconds (from creation, not from first speech), which makes reuse across
more than one turn structurally impossible there. Production billing is 1
credit/min, and now runs for as long as the connection is open -- including
idle time between questions, not just while the avatar is speaking. See
`IDLE_RELEASE_TIMEOUT_S` below for how that's bounded.

Because the browser can send another question at any time, ONE reader task
(`_read_client_messages`) owns `websocket.receive()` for the connection's
whole lifetime -- not just one turn, as it used to -- and dispatches parsed
messages onto two queues that the turn-processing loop and the session-join
handshake read from. Turns are processed strictly one at a time (no barge-in,
same limitation as always); only the setup-and-idle bookkeeping is new.
"""

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import logfire
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.voice_routes import StreamFactory, TurnTiming, _plan_answer
from app.core.logger import get_logger
from app.core.task_bridge import wait_and_cancel_rest
from app.services.heygen_streaming import AvatarRenderer, LiveAvatarRenderer
from app.services.realtime_tts import RealtimeApiSpeaker
from app.services.speech_sanitizer import sanitize_for_speech
from app.services.text_chunker import TextChunker

logger = get_logger(__name__)
router = APIRouter()

# How long to wait for the browser to confirm it has joined the LiveKit room
# before speaking anyway. Generous, because losing the opening words is worse
# than a late start -- but bounded, so a browser that never reports ready
# (old cached avatar.js, blocked WebRTC) degrades instead of hanging.
CLIENT_READY_TIMEOUT_S = 15.0

# How long the HeyGen session + TTS socket are kept alive with no question in
# flight before being released. Set to 2 minutes per explicit request (shorter
# than the 5 minutes first suggested) because production billing is 1
# credit/min and runs the whole time a session sits idle, not just while the
# avatar speaks -- unlike a forgotten browser tab, an unbounded idle session
# is an unbounded, silent cost. The next question after a release simply pays
# the setup cost again and continues; nothing else about the connection ends.
IDLE_RELEASE_TIMEOUT_S = 120.0


@dataclass
class AvatarConnectionState:
    """Everything that's built once per connection and reused across turns,
    plus the bookkeeping needed to release and rebuild it. `renderer` and
    `speaker` being non-None is the single source of truth for "is a session
    currently live" -- checked by `_start_session_setup` before doing any
    work, so calling it every turn is always safe."""

    renderer: AvatarRenderer | None = None
    speaker: RealtimeApiSpeaker | None = None
    session_ready_task: "asyncio.Task | None" = None
    speaker_ready_task: "asyncio.Task | None" = None
    # True once the browser has joined LiveKit for the CURRENT renderer/speaker
    # pair. Reset to False on release, so a rebuilt session sends a fresh
    # "session" frame and waits for "ready" again -- the old room is gone.
    session_joined: bool = False
    idle_release_task: "asyncio.Task | None" = None


def _verbatim_stream_factory(text: str) -> StreamFactory:
    """A StreamFactory that bypasses planning/retrieval/the LLM entirely --
    used for the avatar's scripted per-persona intro (Start Session), which
    must speak the frontend's exact hand-written script, not whatever an LLM
    would generate for it. _produce()'s TextChunker still splits it into
    natural sentence-sized pieces for TTS/HeyGen pacing, same as any other
    turn -- feed() correctly extracts every sentence from one big string in
    a single call, not just from incremental deltas.

    Deliberately NOT persisted to conversation memory: nothing in this path
    touches plan_turn/stream_agent_answer, so there is no append_message()
    call for it -- a canned greeting isn't a real turn in the conversation
    and shouldn't be replayed back to the LLM as one."""

    async def _stream() -> AsyncIterator[str]:
        yield text

    return _stream


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


async def _timed_audio(pcm_chunks: AsyncIterator[bytes], timing: TurnTiming) -> AsyncIterator[bytes]:
    """Passes PCM chunks through untouched, recording the turn's first-audio
    marker as the first one goes by. Needed because _consume() hands the
    audio stream to the renderer instead of iterating it itself, so there is
    no inline place to call record_first_audio() the way voice_routes.py's
    _consume() does.

    Note this marks when audio was first sent TO HeyGen, not when the viewer
    heard anything -- HeyGen's render plus LiveKit delivery still follow. The
    honest user-facing number is measured browser-side in avatar.js."""
    async for chunk in pcm_chunks:
        timing.record_first_audio()  # no-op after the first chunk
        yield chunk


async def _await_client_ready(ready_queue: "asyncio.Queue[None]", log_extra: dict) -> None:
    """Wait for the browser to report it has joined the LiveKit room, and time
    that wait.

    Timed separately because this gap sits between the session frame and the
    first sentence, so without its own number it gets silently absorbed into
    whatever the next measurement covers -- exactly how it ended up
    misattributed to HeyGen session setup once already (P17 in
    architecture-and-query-flow.md)."""
    started = time.monotonic()
    try:
        async with asyncio.timeout(CLIENT_READY_TIMEOUT_S):
            await ready_queue.get()
    except TimeoutError:
        logger.warning(
            "browser never confirmed it joined the LiveKit room after %ss -- speaking anyway; "
            "the first words may not be seen",
            CLIENT_READY_TIMEOUT_S,
            extra=log_extra,
        )
        return
    join_ms = round((time.monotonic() - started) * 1000, 1)
    logfire.info("browser livekit join: {join_ms}ms", join_ms=join_ms)
    logger.info("browser livekit join: %sms", join_ms, extra=log_extra)


async def _consume(
    queue: "asyncio.Queue[str | None]",
    speaker: RealtimeApiSpeaker,
    renderer: AvatarRenderer,
    websocket: WebSocket,
    timing: TurnTiming,
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
        await websocket.send_json({"type": "speaking"})
        # Drives HeyGen's lip-sync directly from our own TTS audio -- the
        # browser never receives this audio itself, only the rendered video
        # (over LiveKit) and these status frames.
        await renderer.speak_audio(_timed_audio(speaker.speak(sentence), timing))


async def _run_producer_consumer(
    stream_factory: StreamFactory,
    speaker: RealtimeApiSpeaker,
    renderer: AvatarRenderer,
    websocket: WebSocket,
    session_id: str,
    timing: TurnTiming,
) -> None:
    queue: "asyncio.Queue[str | None]" = asyncio.Queue()
    producer = asyncio.create_task(_produce(stream_factory, queue), name=f"avatar-producer-{session_id}")
    consumer = asyncio.create_task(
        _consume(queue, speaker, renderer, websocket, timing), name=f"avatar-consumer-{session_id}"
    )
    await wait_and_cancel_rest((producer, consumer), asyncio.FIRST_EXCEPTION)
    # Every sentence has been PUSHED, but HeyGen renders in real time and is
    # almost certainly still speaking. Wait for it to drain before the turn
    # reports "done" -- not to protect teardown any more (the session survives
    # past this turn now), but so the next question doesn't start pushing
    # audio for a NEW turn while this one is still mid-sentence.
    await renderer.wait_until_done()


def _start_session_setup(state: AvatarConnectionState, ws_session_id: str) -> None:
    """Kick off HeyGen + TTS setup as background tasks, unless a session is
    already live or already being built. Idempotent -- safe to call at the
    start of every turn; it only does real work the first time a connection
    needs a session, or again after an idle release rebuilds from scratch."""
    if state.renderer is not None or state.session_ready_task is not None:
        return
    state.renderer = LiveAvatarRenderer()
    state.session_ready_task = asyncio.create_task(
        state.renderer.__aenter__(), name=f"avatar-session-{ws_session_id}"
    )
    state.speaker = RealtimeApiSpeaker()
    state.speaker_ready_task = asyncio.create_task(
        state.speaker.__aenter__(), name=f"avatar-speaker-{ws_session_id}"
    )


async def _ensure_session_joined(
    state: AvatarConnectionState, websocket: WebSocket, ready_queue: "asyncio.Queue[None]", log_extra: dict
) -> None:
    """Wait for setup (already running concurrently with _plan_answer, same
    as before this rework) to finish, and -- only the first time since the
    session was (re)built -- send the session frame and wait for the browser
    to join LiveKit. A no-op wait on turns 2+, since both tasks are already
    done and session_joined is already True."""
    await state.session_ready_task
    await state.speaker_ready_task
    if state.session_joined:
        return
    session = state.renderer.session
    await websocket.send_json(
        {
            "type": "session",
            "session_id": session.session_id,
            "livekit_url": session.livekit_url,
            "livekit_client_token": session.livekit_client_token,
            "max_session_duration": session.max_session_duration,
        }
    )
    await _await_client_ready(ready_queue, log_extra)
    state.session_joined = True


async def _release_avatar_resources(state: AvatarConnectionState, log_extra: dict) -> None:
    """Tear down whatever is currently live, however far setup got, and reset
    state so the next question rebuilds from scratch. Used both for the
    idle-timeout release and final connection teardown -- both are the same
    operation: dispose of what exists, safely, regardless of whether it's
    fully live, still connecting, or already gone."""
    renderer, speaker = state.renderer, state.speaker
    session_ready_task, speaker_ready_task = state.session_ready_task, state.speaker_ready_task
    state.renderer = None
    state.speaker = None
    state.session_ready_task = None
    state.speaker_ready_task = None
    state.session_joined = False

    for task in (session_ready_task, speaker_ready_task):
        if task is not None and not task.done():
            task.cancel()
    for task in (session_ready_task, speaker_ready_task):
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    if renderer is not None:
        with contextlib.suppress(Exception):
            await renderer.__aexit__(None, None, None)
    if speaker is not None:
        with contextlib.suppress(Exception):
            await speaker.__aexit__(None, None, None)
    logger.info("avatar session resources released", extra=log_extra)


async def _release_after_idle(state: AvatarConnectionState, log_extra: dict) -> None:
    await asyncio.sleep(IDLE_RELEASE_TIMEOUT_S)
    if state.renderer is None and state.session_ready_task is None:
        return  # already released some other way (shouldn't happen -- defensive)
    logger.info(
        "releasing avatar session after %ss with no question -- will rebuild on the next one",
        IDLE_RELEASE_TIMEOUT_S,
        extra=log_extra,
    )
    await _release_avatar_resources(state, log_extra)


def _schedule_idle_release(state: AvatarConnectionState, ws_session_id: str, log_extra: dict) -> None:
    state.idle_release_task = asyncio.create_task(
        _release_after_idle(state, log_extra), name=f"avatar-idle-release-{ws_session_id}"
    )


async def _cancel_idle_release(state: AvatarConnectionState) -> None:
    if state.idle_release_task is None:
        return
    state.idle_release_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await state.idle_release_task
    state.idle_release_task = None


@dataclass
class Question:
    """One question off the wire, plus the conversation thread it belongs to.

    thread_id is carried alongside the text (rather than minted per turn)
    because conversation memory is keyed on it -- see _run_turns. Browser-
    supplied and optional; None means "mint me one", same contract
    /ws/voice already uses.

    verbatim marks a scripted turn (currently: the per-persona Start Session
    intro) that must be spoken exactly as sent rather than planned/generated
    -- see _verbatim_stream_factory. False for every ordinary question."""

    text: str
    thread_id: str | None
    verbatim: bool = False


async def _read_client_messages(
    websocket: WebSocket,
    question_queue: "asyncio.Queue[Question]",
    ready_queue: "asyncio.Queue[None]",
) -> None:
    """Sole reader of the browser socket for the WHOLE connection's lifetime,
    not just one turn -- two concurrent receive() calls on one WebSocket are
    not safe. Dispatches question text onto question_queue for _run_turns to
    pick up whenever it's ready for the next one, and readiness pings onto
    ready_queue for whichever _ensure_session_joined call is currently
    waiting on one. Raises on disconnect, which -- via the FIRST_COMPLETED
    race in avatar_ws -- cancels whatever turn is in flight immediately, the
    same guarantee the old per-turn watcher gave, now covering every turn on
    the connection instead of just one."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get("code", 1000))
        text = message.get("text")
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "ready":
            await ready_queue.put(None)
            continue
        if payload.get("type") == "intro":
            intro_text = payload.get("text")
            if isinstance(intro_text, str) and intro_text.strip():
                raw_thread_id = payload.get("thread_id")
                thread_id = (
                    raw_thread_id.strip()
                    if isinstance(raw_thread_id, str) and raw_thread_id.strip()
                    else None
                )
                await question_queue.put(Question(text=intro_text.strip(), thread_id=thread_id, verbatim=True))
            else:
                await _safe_send_error(websocket, 'expected {"type": "intro", "text": "<text>"}')
            continue
        question = payload.get("question")
        if isinstance(question, str) and question.strip():
            raw_thread_id = payload.get("thread_id")
            thread_id = (
                raw_thread_id.strip()
                if isinstance(raw_thread_id, str) and raw_thread_id.strip()
                else None
            )
            await question_queue.put(Question(text=question.strip(), thread_id=thread_id))
        else:
            await _safe_send_error(websocket, 'expected {"question": "<text>"}')


async def _run_turns(
    websocket: WebSocket,
    question_queue: "asyncio.Queue[Question]",
    ready_queue: "asyncio.Queue[None]",
    state: AvatarConnectionState,
    ws_session_id: str,
    log_extra: dict,
) -> None:
    """Processes questions one at a time for the life of the connection.
    A failed turn is logged and reported to the browser without ending the
    connection -- one bad question must not stop the next one from working,
    same principle as the ingestion pipeline's per-video error isolation."""
    # Falls back to a server-minted thread when the browser sends none, so
    # memory still works for a client that doesn't track one (the check
    # scripts, for instance) -- it just won't survive a reconnect. Same
    # contract as /ws/voice.
    fallback_thread_id = uuid.uuid4().hex

    while True:
        question_msg = await question_queue.get()
        await _cancel_idle_release(state)

        question = question_msg.text
        # NOT minted per turn any more. It used to be, which meant
        # get_history() looked up a brand-new key every question and the
        # avatar could never remember anything -- correct when each Ask was
        # its own session, wrong once the connection went multi-turn.
        thread_id = question_msg.thread_id or fallback_thread_id
        logger.info("avatar turn started", extra={**log_extra, "thread_id": thread_id})
        # Same time-to-first-response instrumentation as /ws/voice. Started
        # fresh per turn -- unlike setup, this measures per-question latency,
        # which stays meaningful even though the session itself is reused.
        timing = TurnTiming(start=time.monotonic(), log_extra=log_extra)

        with logfire.span("avatar turn", session_id=ws_session_id, thread_id=thread_id):
            try:
                # Idempotent: only does real work the first turn, or after an
                # idle release. Runs concurrently with _plan_answer below,
                # same as the original single-turn design.
                _start_session_setup(state, ws_session_id)

                if question_msg.verbatim:
                    meta = {"type": "meta", "answer_source": "intro"}
                    stream_factory = _verbatim_stream_factory(question)
                    logger.info(
                        "answer source: INTRO (verbatim script, no LLM)",
                        extra={**log_extra, "answer_source": "intro", "thread_id": thread_id},
                    )
                else:
                    meta, stream_factory = await _plan_answer(question, thread_id, log_extra)
                # Echoed back so the browser can keep reusing this thread on
                # later questions and across reconnects. _plan_answer only
                # sets thread_id itself on the "agent" source; setting it here
                # means the browser gets it on every source.
                meta["thread_id"] = thread_id
                await websocket.send_json(meta)

                await _ensure_session_joined(state, websocket, ready_queue, log_extra)
                await _run_producer_consumer(
                    stream_factory, state.speaker, state.renderer, websocket, ws_session_id, timing
                )
                await websocket.send_json({"type": "done"})
                logger.info("avatar turn completed", extra=log_extra)
            except Exception as exc:
                logger.error("avatar turn failed: %s", exc, extra=log_extra, exc_info=True)
                await _safe_send_error(websocket, str(exc))
                # Don't leave a half-built or now-suspect session around for
                # the next question to (mis)trust -- rebuild clean instead.
                await _release_avatar_resources(state, log_extra)

        _schedule_idle_release(state, ws_session_id, log_extra)


@router.websocket("/ws/avatar")
async def avatar_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    ws_session_id = uuid.uuid4().hex
    log_extra = {"node_name": "avatar_ws", "session_id": ws_session_id}
    logger.info("avatar connection opened", extra=log_extra)

    question_queue: "asyncio.Queue[Question]" = asyncio.Queue()
    ready_queue: "asyncio.Queue[None]" = asyncio.Queue()
    state = AvatarConnectionState()

    reader = asyncio.create_task(
        _read_client_messages(websocket, question_queue, ready_queue), name=f"avatar-reader-{ws_session_id}"
    )
    turns = asyncio.create_task(
        _run_turns(websocket, question_queue, ready_queue, state, ws_session_id, log_extra),
        name=f"avatar-turns-{ws_session_id}",
    )

    try:
        await wait_and_cancel_rest((reader, turns), asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        logger.info("client disconnected", extra=log_extra)
    except Exception as exc:
        logger.error("avatar connection failed: %s", exc, extra=log_extra, exc_info=True)
        await _safe_send_error(websocket, str(exc))
    finally:
        await _cancel_idle_release(state)
        await _release_avatar_resources(state, log_extra)
        await _safe_close(websocket)
        logger.info("avatar connection closed", extra=log_extra)


async def _safe_send_error(websocket: WebSocket, detail: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "detail": detail})


async def _safe_close(websocket: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await websocket.close()
