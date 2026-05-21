"""Unit tests for app/services/agent.py -- orchestration logic only.
planner.plan() and the real LLM calls inside _stream_conversational_answer /
stream_grounded_answer are mocked; conversation_memory is used for real
(pure, fast, no API calls) since testing the real memory integration is
more valuable than mocking it away. Verified manually against the real API
via scripts/check_agent.py, same convention every other paid-API path in
this codebase follows."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.agent import AgentDecision, plan_turn, stream_agent_answer, stream_agent_response
from app.services.conversation_memory import append_message, clear_thread, get_history
from app.services.planner import PlannerDecision
from app.services.rag_stream import RetrievalContext
from app.services.retrieval import RetrievedChunk


def _chunk(score=0.6) -> RetrievedChunk:
    return RetrievedChunk(text="chunk text", source_file="doc.pdf", chunk_index=0, score=score)


def _fake_stream(pieces: list[str]):
    async def _stream(*args, **kwargs):
        for piece in pieces:
            yield piece

    return _stream


# ── plan_turn ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_turn_conversational_skips_retrieval():
    thread = "agent-test-conversational"
    await clear_thread(thread)
    decision = PlannerDecision(is_conversational=True, query="CONVERSATIONAL")

    with (
        patch("app.services.agent.plan", AsyncMock(return_value=decision)),
        patch("app.services.agent.retrieve_context") as mock_retrieve,
    ):
        result = await plan_turn("hi there", thread)

    mock_retrieve.assert_not_called()
    assert result.planner.is_conversational is True
    assert result.retrieval is None
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_plan_turn_search_query_retrieves_using_rewritten_query():
    thread = "agent-test-rewrite"
    await clear_thread(thread)
    decision = PlannerDecision(is_conversational=False, query="how does OLAP differ from data mining")
    context = RetrievalContext(chunks=[_chunk()], top_score=0.6, grounded=True)

    with (
        patch("app.services.agent.plan", AsyncMock(return_value=decision)),
        patch("app.services.agent.retrieve_context", return_value=context) as mock_retrieve,
    ):
        result = await plan_turn("how is it different from that", thread)

    # The retriever must search on the planner's rewritten, self-contained
    # query -- not the raw (possibly pronoun-laden) user message.
    mock_retrieve.assert_called_once_with("how does OLAP differ from data mining")
    assert result.planner.is_conversational is False
    assert result.retrieval is context
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_plan_turn_passes_thread_history_to_planner():
    thread = "agent-test-history-in"
    await clear_thread(thread)
    await append_message(thread, "user", "earlier question")

    decision = PlannerDecision(is_conversational=True, query="CONVERSATIONAL")
    with patch("app.services.agent.plan", AsyncMock(return_value=decision)) as mock_plan:
        await plan_turn("follow-up", thread)

    called_history, called_message = mock_plan.call_args[0]
    assert called_history == [{"role": "user", "content": "earlier question"}]
    assert called_message == "follow-up"
    await clear_thread(thread)


# ── stream_agent_answer ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_agent_answer_conversational_path_skips_grounded_stream():
    thread = "agent-test-stream-conv"
    await clear_thread(thread)
    decision = AgentDecision(planner=PlannerDecision(is_conversational=True, query="CONVERSATIONAL"), retrieval=None)

    with (
        patch("app.services.agent._stream_conversational_answer", _fake_stream(["Hello", " there"])),
        patch("app.services.agent.stream_grounded_answer") as mock_grounded,
    ):
        pieces = [d async for d in stream_agent_answer("hi", thread, decision)]

    assert pieces == ["Hello", " there"]
    mock_grounded.assert_not_called()
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_stream_agent_answer_retrieval_path_passes_history_to_grounded_stream():
    thread = "agent-test-stream-rag"
    await clear_thread(thread)
    await append_message(thread, "user", "earlier turn")
    context = RetrievalContext(chunks=[_chunk()], top_score=0.6, grounded=True)
    decision = AgentDecision(planner=PlannerDecision(is_conversational=False, query="q"), retrieval=context)

    async def fake_grounded(prompt, ctx, history=None):
        assert ctx is context
        assert history == [{"role": "user", "content": "earlier turn"}]
        yield "Grounded"
        yield " answer"

    with patch("app.services.agent.stream_grounded_answer", fake_grounded):
        pieces = [d async for d in stream_agent_answer("what is OLAP", thread, decision)]

    assert pieces == ["Grounded", " answer"]
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_stream_agent_answer_persists_full_turn_to_memory():
    thread = "agent-test-persist"
    await clear_thread(thread)
    decision = AgentDecision(planner=PlannerDecision(is_conversational=True, query="CONVERSATIONAL"), retrieval=None)

    with patch("app.services.agent._stream_conversational_answer", _fake_stream(["Hello", " world"])):
        _ = [d async for d in stream_agent_answer("hi", thread, decision)]

    assert await get_history(thread) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello world"},
    ]
    await clear_thread(thread)


# ── stream_agent_response (one-call convenience wrapper) ─────────────────────


@pytest.mark.asyncio
async def test_stream_agent_response_composes_plan_and_stream():
    thread = "agent-test-response"
    await clear_thread(thread)
    decision = PlannerDecision(is_conversational=True, query="CONVERSATIONAL")

    with (
        patch("app.services.agent.plan", AsyncMock(return_value=decision)),
        patch("app.services.agent._stream_conversational_answer", _fake_stream(["Hi", "!"])),
    ):
        pieces = [d async for d in stream_agent_response("hello", thread)]

    assert pieces == ["Hi", "!"]
    assert (await get_history(thread))[-1] == {"role": "assistant", "content": "Hi!"}
    await clear_thread(thread)


# ── memory persist ordering (P3 fix) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_history_read_excludes_this_turns_own_prompt():
    """The user message is now appended BEFORE generation, so the history read
    has to happen first -- otherwise this turn's own question would be handed
    to the LLM twice, once as history and once as the question."""
    thread = "agent-test-read-before-append"
    await clear_thread(thread)
    await append_message(thread, "user", "earlier question")
    await append_message(thread, "assistant", "earlier answer")

    seen_history: list[list[dict]] = []

    async def capture_history(history, prompt):
        seen_history.append(list(history))
        yield "ok"

    decision = AgentDecision(
        planner=PlannerDecision(is_conversational=True, query="CONVERSATIONAL"), retrieval=None
    )
    with patch("app.services.agent._stream_conversational_answer", capture_history):
        async for _ in stream_agent_answer("new question", thread, decision):
            pass

    assert [m["content"] for m in seen_history[0]] == ["earlier question", "earlier answer"]
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_user_message_is_persisted_even_if_generation_fails():
    """Regression: both appends used to happen after the stream finished, so a
    failure mid-answer lost the whole turn and the next question had no idea
    this one was ever asked."""
    thread = "agent-test-persist-on-failure"
    await clear_thread(thread)

    async def blow_up(history, prompt):
        raise RuntimeError("LLM exploded")
        yield  # pragma: no cover -- makes this an async generator

    decision = AgentDecision(
        planner=PlannerDecision(is_conversational=True, query="CONVERSATIONAL"), retrieval=None
    )
    with patch("app.services.agent._stream_conversational_answer", blow_up):
        with pytest.raises(RuntimeError, match="LLM exploded"):
            async for _ in stream_agent_answer("was this remembered?", thread, decision):
                pass

    history = await get_history(thread)
    assert [m["role"] for m in history] == ["user"]
    assert history[0]["content"] == "was this remembered?"
    await clear_thread(thread)


@pytest.mark.asyncio
async def test_partial_answer_is_persisted_when_the_consumer_stops_early():
    """Stop mid-answer should record what the user actually heard, not
    silence. Relies on the generator's finally running when the consumer
    closes it -- which is exactly why _produce() wraps the stream in
    contextlib.aclosing()."""
    import contextlib

    thread = "agent-test-partial-on-stop"
    await clear_thread(thread)

    decision = AgentDecision(
        planner=PlannerDecision(is_conversational=True, query="CONVERSATIONAL"), retrieval=None
    )
    with patch(
        "app.services.agent._stream_conversational_answer",
        _fake_stream(["First part. ", "Second part. ", "Third part."]),
    ):
        async with contextlib.aclosing(stream_agent_answer("q", thread, decision)) as stream:
            async for _ in stream:
                break  # user pressed Stop after the first chunk

    history = await get_history(thread)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "First part. "  # partial, not the full answer
    await clear_thread(thread)
