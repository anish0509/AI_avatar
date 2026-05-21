import asyncio
import contextlib
import json
import time

import pytest
from fastapi import WebSocketDisconnect

from app.api.avatar_routes import (
    AvatarConnectionState,
    Question,
    _await_client_ready,
    _cancel_idle_release,
    _ensure_session_joined,
    _read_client_messages,
    _release_after_idle,
    _release_avatar_resources,
    _schedule_idle_release,
    _start_session_setup,
    _timed_audio,
    avatar_ws,
)
from app.api.voice_routes import TurnTiming
from app.services.heygen_streaming import AvatarSession


async def _achunks(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


def _timing() -> TurnTiming:
    return TurnTiming(start=time.monotonic(), log_extra={"node_name": "test"})


def _text_frame(payload: dict) -> dict:
    return {"type": "websocket.receive", "text": json.dumps(payload)}


class FakeAvatarSocket:
    """Stands in for Starlette's WebSocket for the whole connection's
    lifetime (not just one turn, unlike the old per-turn fakes) -- an
    internal queue lets a test push() more messages after the connection is
    already running, e.g. to simulate a second question arriving later."""

    def __init__(self, initial: "list[dict] | None" = None) -> None:
        self._incoming: "asyncio.Queue[dict]" = asyncio.Queue()
        for message in initial or []:
            self._incoming.put_nowait(message)
        self.sent: list[dict] = []
        self.closed = False

    def push(self, message: dict) -> None:
        self._incoming.put_nowait(message)

    async def accept(self) -> None:
        return None

    async def receive(self) -> dict:
        return await self._incoming.get()

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


class FakeRenderer:
    """Generic "has __aexit__" stub, reused for both the renderer and speaker
    slots in tests that only care about teardown, not the full AvatarRenderer
    interface."""

    def __init__(self) -> None:
        self.exited = False

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited = True


class FakeAvatarRenderer:
    """Full AvatarRenderer stand-in for the multi-turn reuse tests --
    `instances_created` is a class-level counter so a test can assert setup
    happened exactly once across several turns despite `_start_session_setup`
    being called before every one of them."""

    instances_created = 0

    def __init__(self) -> None:
        type(self).instances_created += 1
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "FakeAvatarRenderer":
        self.entered = True
        self._session = AvatarSession(
            session_id="fake-session",
            ws_url="wss://fake",
            livekit_url="wss://fake-livekit",
            livekit_client_token="fake-token",
            max_session_duration=60,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited = True

    @property
    def session(self) -> AvatarSession:
        return self._session

    async def speak_audio(self, pcm_chunks) -> None:
        async for _ in pcm_chunks:
            pass

    async def wait_until_done(self) -> None:
        return None

    async def interrupt(self) -> None:
        return None


class FakeTtsSpeaker:
    """RealtimeApiSpeaker stand-in, same counting purpose as FakeAvatarRenderer."""

    instances_created = 0

    def __init__(self) -> None:
        type(self).instances_created += 1
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "FakeTtsSpeaker":
        self.entered = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited = True

    async def speak(self, text: str):
        yield b"\x00\x01"


async def _fake_plan_answer(question: str, thread_id: str, log_extra: dict):
    async def stream():
        yield f"Answer to {question}. "

    return {"type": "meta", "answer_source": "gpt"}, stream


@pytest.mark.asyncio
async def test_timed_audio_passes_chunks_through_unchanged() -> None:
    timing = _timing()
    received = [chunk async for chunk in _timed_audio(_achunks([b"one", b"two"]), timing)]
    assert received == [b"one", b"two"]


@pytest.mark.asyncio
async def test_timed_audio_records_first_audio() -> None:
    timing = _timing()
    async for _ in _timed_audio(_achunks([b"one", b"two"]), timing):
        pass
    # record_first_audio() returns the measured ms only on its FIRST call, so
    # None here proves the wrapper already recorded it while streaming.
    assert timing.record_first_audio() is None


@pytest.mark.asyncio
async def test_timed_audio_with_no_chunks_records_nothing() -> None:
    timing = _timing()
    received = [chunk async for chunk in _timed_audio(_achunks([]), timing)]
    assert received == []
    assert timing.record_first_audio() is not None  # still unrecorded


@pytest.mark.asyncio
async def test_await_client_ready_returns_as_soon_as_the_browser_reports_in() -> None:
    ready_queue: "asyncio.Queue[None]" = asyncio.Queue()
    ready_queue.put_nowait(None)
    async with asyncio.timeout(2.0):
        await _await_client_ready(ready_queue, {"node_name": "test"})


@pytest.mark.asyncio
async def test_await_client_ready_gives_up_instead_of_hanging(monkeypatch) -> None:
    """A browser running cached JS will never send "ready". Speaking late is
    better than never answering at all."""
    monkeypatch.setattr("app.api.avatar_routes.CLIENT_READY_TIMEOUT_S", 0.05)
    async with asyncio.timeout(2.0):
        await _await_client_ready(asyncio.Queue(), {"node_name": "test"})  # never filled


@pytest.mark.asyncio
async def test_read_client_messages_dispatches_question_to_the_queue() -> None:
    ws = FakeAvatarSocket([_text_frame({"question": "hello"})])
    question_queue: "asyncio.Queue[Question]" = asyncio.Queue()
    ready_queue: "asyncio.Queue[None]" = asyncio.Queue()
    reader = asyncio.create_task(_read_client_messages(ws, question_queue, ready_queue))

    async with asyncio.timeout(2.0):
        question = await question_queue.get()
    assert question == Question(text="hello", thread_id=None)
    reader.cancel()


@pytest.mark.asyncio
async def test_read_client_messages_carries_the_browsers_thread_id() -> None:
    ws = FakeAvatarSocket([_text_frame({"question": "hello", "thread_id": "abc-123"})])
    question_queue: "asyncio.Queue[Question]" = asyncio.Queue()
    reader = asyncio.create_task(_read_client_messages(ws, question_queue, asyncio.Queue()))

    async with asyncio.timeout(2.0):
        question = await question_queue.get()
    assert question == Question(text="hello", thread_id="abc-123")
    reader.cancel()


@pytest.mark.asyncio
async def test_read_client_messages_dispatches_intro_as_a_verbatim_question() -> None:
    ws = FakeAvatarSocket([_text_frame({"type": "intro", "text": "Welcome!", "thread_id": "t1"})])
    question_queue: "asyncio.Queue[Question]" = asyncio.Queue()
    reader = asyncio.create_task(_read_client_messages(ws, question_queue, asyncio.Queue()))

    async with asyncio.timeout(2.0):
        question = await question_queue.get()
    assert question == Question(text="Welcome!", thread_id="t1", verbatim=True)
    reader.cancel()


@pytest.mark.asyncio
async def test_read_client_messages_errors_on_an_intro_with_no_text() -> None:
    ws = FakeAvatarSocket([_text_frame({"type": "intro"}), _text_frame({"question": "hi"})])
    question_queue: "asyncio.Queue[Question]" = asyncio.Queue()
    reader = asyncio.create_task(_read_client_messages(ws, question_queue, asyncio.Queue()))

    async with asyncio.timeout(2.0):
        question = await question_queue.get()  # the bad frame didn't stop the good one after it
    assert question == Question(text="hi", thread_id=None)
    reader.cancel()
    assert ws.sent == [{"type": "error", "detail": 'expected {"type": "intro", "text": "<text>"}'}]


@pytest.mark.asyncio
async def test_read_client_messages_dispatches_ready_to_the_queue() -> None:
    ws = FakeAvatarSocket([_text_frame({"type": "ready"})])
    question_queue: "asyncio.Queue[str]" = asyncio.Queue()
    ready_queue: "asyncio.Queue[None]" = asyncio.Queue()
    reader = asyncio.create_task(_read_client_messages(ws, question_queue, ready_queue))

    async with asyncio.timeout(2.0):
        await ready_queue.get()
    reader.cancel()


@pytest.mark.asyncio
async def test_read_client_messages_raises_on_disconnect() -> None:
    ws = FakeAvatarSocket([{"type": "websocket.disconnect", "code": 1000}])
    with pytest.raises(WebSocketDisconnect):
        await _read_client_messages(ws, asyncio.Queue(), asyncio.Queue())


@pytest.mark.asyncio
async def test_read_client_messages_ignores_frames_it_cannot_parse() -> None:
    # A stray non-JSON frame must not kill the reader -- it is also the only
    # thing detecting disconnects for the WHOLE connection now, not just one
    # turn, so dying here would break every future question on this tab.
    ws = FakeAvatarSocket([{"type": "websocket.receive", "text": "not json"}, _text_frame({"question": "hi"})])
    question_queue: "asyncio.Queue[Question]" = asyncio.Queue()
    reader = asyncio.create_task(_read_client_messages(ws, question_queue, asyncio.Queue()))

    async with asyncio.timeout(2.0):
        question = await question_queue.get()
    assert question == Question(text="hi", thread_id=None)
    reader.cancel()


@pytest.mark.asyncio
async def test_read_client_messages_errors_on_a_missing_question_without_disconnecting() -> None:
    ws = FakeAvatarSocket([_text_frame({"not_a_question": True}), _text_frame({"question": "hi"})])
    question_queue: "asyncio.Queue[Question]" = asyncio.Queue()
    reader = asyncio.create_task(_read_client_messages(ws, question_queue, asyncio.Queue()))

    async with asyncio.timeout(2.0):
        question = await question_queue.get()  # the bad frame didn't stop the good one after it
    assert question == Question(text="hi", thread_id=None)
    reader.cancel()
    assert ws.sent == [{"type": "error", "detail": 'expected {"question": "<text>"}'}]


@pytest.mark.asyncio
async def test_start_session_setup_only_builds_once_across_turns(monkeypatch) -> None:
    FakeAvatarRenderer.instances_created = 0
    FakeTtsSpeaker.instances_created = 0
    monkeypatch.setattr("app.api.avatar_routes.LiveAvatarRenderer", FakeAvatarRenderer)
    monkeypatch.setattr("app.api.avatar_routes.RealtimeApiSpeaker", FakeTtsSpeaker)

    state = AvatarConnectionState()
    _start_session_setup(state, "test-session")
    first_renderer = state.renderer
    _start_session_setup(state, "test-session")  # simulates a second turn on the same connection

    assert state.renderer is first_renderer
    assert FakeAvatarRenderer.instances_created == 1
    assert FakeTtsSpeaker.instances_created == 1

    await state.session_ready_task
    await state.speaker_ready_task


@pytest.mark.asyncio
async def test_ensure_session_joined_sends_the_session_frame_only_once(monkeypatch) -> None:
    monkeypatch.setattr("app.api.avatar_routes.LiveAvatarRenderer", FakeAvatarRenderer)
    monkeypatch.setattr("app.api.avatar_routes.RealtimeApiSpeaker", FakeTtsSpeaker)
    state = AvatarConnectionState()
    _start_session_setup(state, "test-session")

    ws = FakeAvatarSocket()
    ready_queue: "asyncio.Queue[None]" = asyncio.Queue()
    ready_queue.put_nowait(None)

    await _ensure_session_joined(state, ws, ready_queue, {"node_name": "test"})
    await _ensure_session_joined(state, ws, ready_queue, {"node_name": "test"})  # second turn -- no-op

    session_frames = [f for f in ws.sent if f.get("type") == "session"]
    assert len(session_frames) == 1
    assert state.session_joined is True


@pytest.mark.asyncio
async def test_release_avatar_resources_cancels_still_connecting_tasks() -> None:
    """Session/speaker setup runs concurrently with generation, so an error in
    planning can reach release while either is mid-connect."""
    state = AvatarConnectionState()
    renderer, speaker = FakeRenderer(), FakeRenderer()
    started_renderer, started_speaker = asyncio.Event(), asyncio.Event()

    async def never_finishes(started: asyncio.Event) -> None:
        started.set()
        await asyncio.Event().wait()

    state.renderer = renderer
    state.speaker = speaker
    state.session_ready_task = asyncio.create_task(never_finishes(started_renderer))
    state.speaker_ready_task = asyncio.create_task(never_finishes(started_speaker))
    await started_renderer.wait()
    await started_speaker.wait()

    await _release_avatar_resources(state, {"node_name": "test"})

    assert renderer.exited is True
    assert speaker.exited is True
    assert state.renderer is None
    assert state.speaker is None
    assert state.session_ready_task is None
    assert state.speaker_ready_task is None
    assert state.session_joined is False


@pytest.mark.asyncio
async def test_release_avatar_resources_survives_a_failed_setup_task() -> None:
    """The real case: HeyGen answers 403 while the LLM is already streaming.
    Release must still run and must not raise out of the finally block."""

    async def boom() -> None:
        raise RuntimeError("HeyGen 403")

    state = AvatarConnectionState()
    state.renderer = FakeRenderer()
    state.speaker = FakeRenderer()
    state.session_ready_task = asyncio.create_task(boom())
    state.speaker_ready_task = asyncio.create_task(boom())

    await _release_avatar_resources(state, {"node_name": "test"})

    assert state.renderer is None


@pytest.mark.asyncio
async def test_release_avatar_resources_is_a_no_op_when_nothing_is_live() -> None:
    state = AvatarConnectionState()
    await _release_avatar_resources(state, {"node_name": "test"})  # must not raise
    assert state.renderer is None


@pytest.mark.asyncio
async def test_schedule_and_cancel_idle_release() -> None:
    state = AvatarConnectionState()
    _schedule_idle_release(state, "test-session", {"node_name": "test"})
    assert state.idle_release_task is not None
    await _cancel_idle_release(state)
    assert state.idle_release_task is None


@pytest.mark.asyncio
async def test_idle_release_after_timeout_releases_live_resources(monkeypatch) -> None:
    monkeypatch.setattr("app.api.avatar_routes.IDLE_RELEASE_TIMEOUT_S", 0.01)
    state = AvatarConnectionState()
    renderer = FakeRenderer()
    state.renderer = renderer

    await _release_after_idle(state, {"node_name": "test"})

    assert renderer.exited is True
    assert state.renderer is None


@pytest.mark.asyncio
async def test_second_question_over_one_connection_reuses_renderer_and_speaker(monkeypatch) -> None:
    """The actual point of this rework: two questions on ONE /ws/avatar
    connection build the HeyGen session and TTS socket exactly once between
    them, not once each."""
    FakeAvatarRenderer.instances_created = 0
    FakeTtsSpeaker.instances_created = 0
    monkeypatch.setattr("app.api.avatar_routes.LiveAvatarRenderer", FakeAvatarRenderer)
    monkeypatch.setattr("app.api.avatar_routes.RealtimeApiSpeaker", FakeTtsSpeaker)
    monkeypatch.setattr("app.api.avatar_routes._plan_answer", _fake_plan_answer)

    ws = FakeAvatarSocket(
        [
            _text_frame({"question": "What is RAG?"}),
            _text_frame({"type": "ready"}),
            _text_frame({"question": "What is a neural network?"}),
        ]
    )

    task = asyncio.create_task(avatar_ws(ws))
    async with asyncio.timeout(2.0):
        while len([f for f in ws.sent if f.get("type") == "done"]) < 2:
            await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert FakeAvatarRenderer.instances_created == 1
    assert FakeTtsSpeaker.instances_created == 1
    session_frames = [f for f in ws.sent if f.get("type") == "session"]
    assert len(session_frames) == 1  # no re-join for the second question


@pytest.mark.asyncio
async def test_intro_turn_speaks_the_exact_script_without_calling_plan_answer(monkeypatch) -> None:
    """The Start Session intro must bypass planning/retrieval/the LLM
    entirely and speak the frontend's exact script -- proven here by making
    _plan_answer raise if it's ever called, not just by asserting the meta
    shape."""
    monkeypatch.setattr("app.api.avatar_routes.LiveAvatarRenderer", FakeAvatarRenderer)
    monkeypatch.setattr("app.api.avatar_routes.RealtimeApiSpeaker", FakeTtsSpeaker)

    async def _plan_answer_must_not_be_called(question, thread_id, log_extra):
        raise AssertionError("_plan_answer must not run for a verbatim intro turn")

    monkeypatch.setattr("app.api.avatar_routes._plan_answer", _plan_answer_must_not_be_called)

    ws = FakeAvatarSocket(
        [
            _text_frame({"type": "intro", "text": "Hello there. Welcome to the class!", "thread_id": "t1"}),
            _text_frame({"type": "ready"}),
        ]
    )

    task = asyncio.create_task(avatar_ws(ws))
    async with asyncio.timeout(2.0):
        while not any(f.get("type") == "done" for f in ws.sent):
            await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    meta_frames = [f for f in ws.sent if f.get("type") == "meta"]
    assert meta_frames == [{"type": "meta", "answer_source": "intro", "thread_id": "t1"}]
    spoken = [f["text"] for f in ws.sent if f.get("type") == "text"]
    assert spoken == ["Hello there.", "Welcome to the class!"]


@pytest.mark.asyncio
async def test_thread_id_persists_across_two_questions_without_browser_supplied_id(monkeypatch) -> None:
    """Regression: thread_id used to be minted fresh INSIDE the turn loop, so
    get_history() always looked up a brand-new key and the avatar could never
    remember anything -- correct when each Ask was its own connection, wrong
    once the connection became multi-turn. Two questions with no
    browser-supplied thread_id must still share ONE thread across the
    connection (the fallback minted once in _run_turns), and the server must
    echo it back in each meta frame so the browser can keep reusing it."""
    monkeypatch.setattr("app.api.avatar_routes.LiveAvatarRenderer", FakeAvatarRenderer)
    monkeypatch.setattr("app.api.avatar_routes.RealtimeApiSpeaker", FakeTtsSpeaker)

    seen_thread_ids: list[str] = []

    async def recording_plan_answer(question, thread_id, log_extra):
        seen_thread_ids.append(thread_id)

        async def stream():
            yield f"Answer to {question}. "

        return {"type": "meta", "answer_source": "gpt"}, stream

    monkeypatch.setattr("app.api.avatar_routes._plan_answer", recording_plan_answer)

    ws = FakeAvatarSocket(
        [
            _text_frame({"question": "What is RAG?"}),
            _text_frame({"type": "ready"}),
            _text_frame({"question": "What is a neural network?"}),
        ]
    )

    task = asyncio.create_task(avatar_ws(ws))
    async with asyncio.timeout(2.0):
        while len([f for f in ws.sent if f.get("type") == "done"]) < 2:
            await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert len(seen_thread_ids) == 2
    assert seen_thread_ids[0] == seen_thread_ids[1], "each question got a DIFFERENT thread -- memory is broken"

    meta_frames = [f for f in ws.sent if f.get("type") == "meta"]
    assert len(meta_frames) == 2
    assert meta_frames[0]["thread_id"] == meta_frames[1]["thread_id"] == seen_thread_ids[0]


@pytest.mark.asyncio
async def test_browser_supplied_thread_id_is_used_verbatim(monkeypatch) -> None:
    """The browser echoes back whatever thread_id the server sent in a
    previous meta frame (avatar.js) -- confirms the server actually uses it
    rather than silently minting its own regardless."""
    monkeypatch.setattr("app.api.avatar_routes.LiveAvatarRenderer", FakeAvatarRenderer)
    monkeypatch.setattr("app.api.avatar_routes.RealtimeApiSpeaker", FakeTtsSpeaker)

    seen_thread_ids: list[str] = []

    async def recording_plan_answer(question, thread_id, log_extra):
        seen_thread_ids.append(thread_id)

        async def stream():
            yield "ok "

        return {"type": "meta", "answer_source": "gpt"}, stream

    monkeypatch.setattr("app.api.avatar_routes._plan_answer", recording_plan_answer)

    ws = FakeAvatarSocket(
        [_text_frame({"question": "hi", "thread_id": "browser-thread-xyz"}), _text_frame({"type": "ready"})]
    )
    task = asyncio.create_task(avatar_ws(ws))
    async with asyncio.timeout(2.0):
        while not [f for f in ws.sent if f.get("type") == "done"]:
            await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert seen_thread_ids == ["browser-thread-xyz"]
