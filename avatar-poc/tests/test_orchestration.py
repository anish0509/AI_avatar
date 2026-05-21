import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocketDisconnect

from app.api.voice_routes import _produce, voice_ws
from app.core.config import settings
from app.services.agent import AgentDecision
from app.services.planner import PlannerDecision
from app.services.realtime_tts import TtsSpeaker
from app.services.rag_stream import RetrievalContext
from app.services.retrieval import RetrievedChunk

# Every turn now leads with a "meta" event announcing the answer source, so
# it's always visible (browser + logs) whether a reply is retrieval-grounded
# RAG or a direct LLM call. Tests assert it as the first sent frame.
META_GPT = {"type": "meta", "answer_source": "gpt"}


class FakeBrowserSocket:
    """Stands in for Starlette's WebSocket, browser side. Records every
    send_json/send_bytes call in arrival order; replays one canned prompt
    payload (or raises) from receive_json()."""

    def __init__(
        self,
        prompt_payload: object = None,
        fail_send_bytes_after: int | None = None,
        disconnect_event: "asyncio.Event | None" = None,
    ) -> None:
        self._prompt_payload = prompt_payload
        self._fail_send_bytes_after = fail_send_bytes_after
        self._send_bytes_calls = 0
        self._disconnect_event = disconnect_event
        self.sent: list[dict | bytes] = []
        self.closed = False

    async def accept(self) -> None:
        return None

    async def receive_json(self) -> dict:
        if isinstance(self._prompt_payload, BaseException):
            raise self._prompt_payload
        return self._prompt_payload

    async def receive(self) -> dict:
        """Stands in for the raw ASGI receive() the disconnect watcher polls.
        With no disconnect_event configured, hangs forever (browser still
        connected) so the watcher task just sits pending until cancelled."""
        if self._disconnect_event is not None:
            await self._disconnect_event.wait()
            return {"type": "websocket.disconnect", "code": 1000}
        await asyncio.Event().wait()

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self._send_bytes_calls += 1
        if self._fail_send_bytes_after is not None and self._send_bytes_calls > self._fail_send_bytes_after:
            raise WebSocketDisconnect(code=1006)
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class FakeSpeaker(TtsSpeaker):
    """Concrete lightweight TtsSpeaker stub (not a Mock of RealtimeApiSpeaker)."""

    def __init__(
        self,
        audio_by_sentence: dict[str, list[bytes]] | None = None,
        fail_on: str | None = None,
        hang_after_first_chunk_on: str | None = None,
        signal_on_hang: "asyncio.Event | None" = None,
    ) -> None:
        self.calls: list[str] = []
        self._audio_by_sentence = audio_by_sentence or {}
        self._fail_on = fail_on
        self._hang_on = hang_after_first_chunk_on
        self._signal_on_hang = signal_on_hang

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        if text == self._fail_on:
            raise RuntimeError("tts boom")
        for chunk in self._audio_by_sentence.get(text, [b"\x00\x01"]):
            yield chunk
        if text == self._hang_on:
            if self._signal_on_hang is not None:
                self._signal_on_hang.set()
            await asyncio.Event().wait()  # simulates "still speaking" until cancelled


def _fake_llm_stream(pieces: list[str]):
    async def _stream(prompt: str) -> AsyncIterator[str]:
        for piece in pieces:
            yield piece

    return _stream


def _failing_llm_stream(pieces: list[str], error: Exception):
    async def _stream(prompt: str) -> AsyncIterator[str]:
        for piece in pieces:
            yield piece
        raise error

    return _stream


def _fake_grounded_stream(pieces: list[str]):
    # stream_grounded_answer's real signature is (prompt, context).
    async def _stream(prompt: str, context: RetrievalContext) -> AsyncIterator[str]:
        for piece in pieces:
            yield piece

    return _stream


def _patched_gpt_pipeline(llm_stream, speaker: FakeSpeaker):
    # Pin answer_source explicitly rather than relying on the ambient .env
    # value -- these tests exercise the GPT (direct-LLM) path specifically.
    return (
        patch.object(settings, "answer_source", "gpt"),
        patch("app.api.voice_routes.stream_llm_response", llm_stream),
        patch("app.api.voice_routes.RealtimeApiSpeaker", lambda *a, **kw: speaker),
    )


@pytest.mark.asyncio
async def test_relays_meta_then_text_then_audio_per_sentence_then_done():
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi"})
    speaker = FakeSpeaker(audio_by_sentence={"Hello world.": [b"a1"], "Second sentence.": [b"a2"]})
    llm_stream = _fake_llm_stream(["Hello world. ", "Second sentence. "])

    p1, p2, p3 = _patched_gpt_pipeline(llm_stream, speaker)
    with p1, p2, p3:
        await voice_ws(fake_ws)

    assert fake_ws.sent == [
        META_GPT,
        {"type": "text", "text": "Hello world."},
        b"a1",
        {"type": "text", "text": "Second sentence."},
        b"a2",
        {"type": "done"},
    ]
    assert speaker.calls == ["Hello world.", "Second sentence."]
    assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_leftover_without_terminal_punctuation_is_still_relayed_via_flush():
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi"})
    speaker = FakeSpeaker(audio_by_sentence={"no punctuation here": [b"a1"]})
    llm_stream = _fake_llm_stream(["no punctuation here"])

    p1, p2, p3 = _patched_gpt_pipeline(llm_stream, speaker)
    with p1, p2, p3:
        await voice_ws(fake_ws)

    assert fake_ws.sent == [
        META_GPT,
        {"type": "text", "text": "no punctuation here"},
        b"a1",
        {"type": "done"},
    ]
    assert speaker.calls == ["no punctuation here"]


@pytest.mark.asyncio
async def test_llm_stream_failure_sends_error_frame():
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi"})
    speaker = FakeSpeaker()
    llm_stream = _failing_llm_stream(["no sentence end yet"], RuntimeError("llm boom"))

    p1, p2, p3 = _patched_gpt_pipeline(llm_stream, speaker)
    with p1, p2, p3:
        await voice_ws(fake_ws)

    assert fake_ws.sent == [META_GPT, {"type": "error", "detail": "llm boom"}]
    assert speaker.calls == []


@pytest.mark.asyncio
async def test_tts_failure_sends_error_frame():
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi"})
    speaker = FakeSpeaker(fail_on="Hello.")
    llm_stream = _fake_llm_stream(["Hello. "])

    p1, p2, p3 = _patched_gpt_pipeline(llm_stream, speaker)
    with p1, p2, p3:
        await voice_ws(fake_ws)

    assert fake_ws.sent == [
        META_GPT,
        {"type": "text", "text": "Hello."},
        {"type": "error", "detail": "tts boom"},
    ]
    assert speaker.calls == ["Hello."]


@pytest.mark.asyncio
async def test_disconnect_before_prompt_is_handled_cleanly():
    fake_ws = FakeBrowserSocket(prompt_payload=WebSocketDisconnect())
    speaker = FakeSpeaker()
    llm_stream = _fake_llm_stream([])

    p1, p2, p3 = _patched_gpt_pipeline(llm_stream, speaker)
    with p1, p2, p3:
        await voice_ws(fake_ws)

    # Disconnected before a prompt arrived -> not even the meta frame is sent.
    assert fake_ws.sent == []
    assert speaker.calls == []


@pytest.mark.asyncio
async def test_disconnect_mid_relay_cancels_producer_cleanly():
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi"}, fail_send_bytes_after=0)
    speaker = FakeSpeaker(audio_by_sentence={"Hello.": [b"a1"], "Second.": [b"a2"]})
    llm_stream = _fake_llm_stream(["Hello. ", "Second. "])

    p1, p2, p3 = _patched_gpt_pipeline(llm_stream, speaker)
    with p1, p2, p3:
        await voice_ws(fake_ws)

    assert fake_ws.sent == [META_GPT, {"type": "text", "text": "Hello."}]
    assert speaker.calls == ["Hello."]


@pytest.mark.asyncio
async def test_stop_disconnect_cancels_pipeline_mid_speech():
    """Simulates clicking the Stop button while the avatar is still
    mid-sentence: nothing ever fails a send (unlike the mid-relay test
    above) -- the browser just closes the connection, which the
    disconnect watcher must notice and use to cancel the still-running
    pipeline, rather than only noticing on a later send attempt."""
    disconnect_event = asyncio.Event()
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi"}, disconnect_event=disconnect_event)
    speaker = FakeSpeaker(
        audio_by_sentence={"Hello world.": [b"a1"]},
        hang_after_first_chunk_on="Hello world.",
        signal_on_hang=disconnect_event,
    )
    llm_stream = _fake_llm_stream(["Hello world. "])

    p1, p2, p3 = _patched_gpt_pipeline(llm_stream, speaker)
    with p1, p2, p3:
        await asyncio.wait_for(voice_ws(fake_ws), timeout=2)

    assert fake_ws.sent == [META_GPT, {"type": "text", "text": "Hello world."}, b"a1"]
    assert speaker.calls == ["Hello world."]


@pytest.mark.asyncio
async def test_rag_path_emits_rag_meta_and_uses_grounded_stream():
    """answer_source="rag" must (1) emit a rich RAG meta event reflecting the
    real retrieval decision and (2) stream via stream_grounded_answer (the
    retrieved-context path), NOT the plain-LLM path -- so the browser/logs can
    prove the answer came from RAG. Retrieval itself is mocked (no real
    Pinecone)."""
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "what is OLAP"})
    speaker = FakeSpeaker(audio_by_sentence={"Grounded answer.": [b"a1"]})

    context = RetrievalContext(
        chunks=[RetrievedChunk(text="OLAP is ...", source_file="olap.pdf", chunk_index=0, score=0.58)],
        top_score=0.58,
        grounded=True,
    )
    gpt_must_not_run = _failing_llm_stream([], RuntimeError("GPT path must not run for RAG"))

    with (
        patch.object(settings, "answer_source", "rag"),
        patch("app.api.voice_routes.retrieve_context", AsyncMock(return_value=context)),
        patch("app.api.voice_routes.stream_grounded_answer", _fake_grounded_stream(["Grounded answer. "])),
        patch("app.api.voice_routes.stream_llm_response", gpt_must_not_run),
        patch("app.api.voice_routes.RealtimeApiSpeaker", lambda *a, **kw: speaker),
    ):
        await voice_ws(fake_ws)

    assert fake_ws.sent == [
        {
            "type": "meta",
            "answer_source": "rag",
            "grounded": True,
            "top_score": 0.58,
            "score_floor": settings.retrieval_score_floor,
            "sources": ["olap.pdf"],
        },
        {"type": "text", "text": "Grounded answer."},
        b"a1",
        {"type": "done"},
    ]
    assert speaker.calls == ["Grounded answer."]


@pytest.mark.asyncio
async def test_rag_path_reports_blocked_when_not_grounded():
    """When retrieval isn't grounded, the meta must say so (grounded=False)
    -- this is the signal that lets the browser show 'abstaining' rather than
    the user wondering whether RAG even ran."""
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "what is sales"})
    speaker = FakeSpeaker(audio_by_sentence={"I don't have enough information in the knowledge base to answer that.": [b"a1"]})

    context = RetrievalContext(
        chunks=[RetrievedChunk(text="unrelated", source_file="olap.pdf", chunk_index=2, score=0.27)],
        top_score=0.27,
        grounded=False,
    )

    with (
        patch.object(settings, "answer_source", "rag"),
        patch("app.api.voice_routes.retrieve_context", AsyncMock(return_value=context)),
        patch(
            "app.api.voice_routes.stream_grounded_answer",
            _fake_grounded_stream(["I don't have enough information in the knowledge base to answer that."]),
        ),
        patch("app.api.voice_routes.RealtimeApiSpeaker", lambda *a, **kw: speaker),
    ):
        await voice_ws(fake_ws)

    assert fake_ws.sent[0] == {
        "type": "meta",
        "answer_source": "rag",
        "grounded": False,
        "top_score": 0.27,
        "score_floor": settings.retrieval_score_floor,
        "sources": ["olap.pdf"],
    }


@pytest.mark.asyncio
async def test_agent_path_conversational_meta_and_stream():
    """answer_source="agent", planner says CONVERSATIONAL: meta must report
    is_conversational=True with no retrieval fields, and it must stream via
    stream_agent_answer (the agent path), not the plain-LLM or plain-RAG
    paths -- proving the agent, not a fallback, actually ran."""
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi there", "thread_id": "thread-abc"})
    speaker = FakeSpeaker(audio_by_sentence={"Hello!": [b"a1"]})

    decision = AgentDecision(planner=PlannerDecision(is_conversational=True, query="CONVERSATIONAL"), retrieval=None)
    gpt_must_not_run = _failing_llm_stream([], RuntimeError("GPT path must not run for agent"))

    async def fake_stream_agent_answer(prompt, thread_id, passed_decision):
        assert thread_id == "thread-abc"
        assert passed_decision is decision
        yield "Hello!"

    with (
        patch.object(settings, "answer_source", "agent"),
        patch("app.api.voice_routes.plan_turn", AsyncMock(return_value=decision)),
        patch("app.api.voice_routes.stream_agent_answer", fake_stream_agent_answer),
        patch("app.api.voice_routes.stream_llm_response", gpt_must_not_run),
        patch("app.api.voice_routes.RealtimeApiSpeaker", lambda *a, **kw: speaker),
    ):
        await voice_ws(fake_ws)

    assert fake_ws.sent == [
        {
            "type": "meta",
            "answer_source": "agent",
            "thread_id": "thread-abc",
            "is_conversational": True,
            "planner_query": "CONVERSATIONAL",
            "grounded": None,
            "top_score": None,
            "score_floor": None,
            "sources": [],
        },
        {"type": "text", "text": "Hello!"},
        b"a1",
        {"type": "done"},
    ]


@pytest.mark.asyncio
async def test_agent_path_retrieval_meta_reflects_grounding_decision():
    """answer_source="agent", planner routes to a lookup: meta must report
    the real retrieval decision (grounded/top_score/sources), same rigor as
    the plain "rag" path -- observability shouldn't regress for the richer
    agent path."""
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "what is OLAP", "thread_id": "thread-xyz"})
    speaker = FakeSpeaker(audio_by_sentence={"OLAP is...": [b"a1"]})

    context = RetrievalContext(
        chunks=[RetrievedChunk(text="OLAP is ...", source_file="olap.pdf", chunk_index=0, score=0.58)],
        top_score=0.58,
        grounded=True,
    )
    decision = AgentDecision(planner=PlannerDecision(is_conversational=False, query="what is OLAP"), retrieval=context)

    async def fake_stream_agent_answer(prompt, thread_id, passed_decision):
        yield "OLAP is..."

    with (
        patch.object(settings, "answer_source", "agent"),
        patch("app.api.voice_routes.plan_turn", AsyncMock(return_value=decision)),
        patch("app.api.voice_routes.stream_agent_answer", fake_stream_agent_answer),
        patch("app.api.voice_routes.RealtimeApiSpeaker", lambda *a, **kw: speaker),
    ):
        await voice_ws(fake_ws)

    assert fake_ws.sent[0] == {
        "type": "meta",
        "answer_source": "agent",
        "thread_id": "thread-xyz",
        "is_conversational": False,
        "planner_query": "what is OLAP",
        "grounded": True,
        "top_score": 0.58,
        "score_floor": settings.retrieval_score_floor,
        "sources": ["olap.pdf"],
    }


@pytest.mark.asyncio
async def test_agent_path_generates_thread_id_when_none_sent():
    """First turn of a session: the browser sends no thread_id, so the
    server must mint one and echo it back -- the client persists whatever
    comes back in meta (see app.js), so a missing/empty thread_id must
    never surface as null/empty in the response."""
    fake_ws = FakeBrowserSocket(prompt_payload={"prompt": "hi"})  # no thread_id key at all
    speaker = FakeSpeaker(audio_by_sentence={"Hello!": [b"a1"]})

    decision = AgentDecision(planner=PlannerDecision(is_conversational=True, query="CONVERSATIONAL"), retrieval=None)

    async def fake_stream_agent_answer(prompt, thread_id, passed_decision):
        yield "Hello!"

    with (
        patch.object(settings, "answer_source", "agent"),
        patch("app.api.voice_routes.plan_turn", AsyncMock(return_value=decision)),
        patch("app.api.voice_routes.stream_agent_answer", fake_stream_agent_answer),
        patch("app.api.voice_routes.RealtimeApiSpeaker", lambda *a, **kw: speaker),
    ):
        await voice_ws(fake_ws)

    generated_thread_id = fake_ws.sent[0]["thread_id"]
    assert isinstance(generated_thread_id, str) and generated_thread_id


@pytest.mark.asyncio
async def test_produce_closes_the_answer_stream_when_cancelled():
    """_produce wraps the answer stream in contextlib.aclosing() specifically
    so a cancelled turn (Stop / disconnect) still runs that stream's finally.
    For the agent path that finally is what persists the partial answer to
    conversation memory -- without aclosing it is deferred to GC, i.e.
    possibly until after the NEXT question has already read the history.

    Guards the pairing documented in stream_agent_answer's docstring: the
    aclosing() here and the absence of @traceable there are both required,
    and removing either silently reintroduces the bug."""
    import contextlib

    closed = asyncio.Event()

    async def answer_stream():
        try:
            yield "First sentence. "
            await asyncio.Event().wait()  # hang so cancellation lands mid-stream
            yield "never reached"
        finally:
            closed.set()

    queue: "asyncio.Queue[str | None]" = asyncio.Queue()
    task = asyncio.create_task(_produce(lambda: answer_stream(), queue))

    assert await asyncio.wait_for(queue.get(), timeout=2) == "First sentence."
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert closed.is_set(), "answer stream was not closed -- is aclosing() still in _produce?"
