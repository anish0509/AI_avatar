"""Unit tests for app/services/rag_stream.py. Everything here is pure logic
or mocked -- build_grounded_prompt(), is_grounded(), retrieve_context() (with
search mocked), and the ABSTAIN branch of stream_grounded_answer() (which
makes NO LLM call, so it's fully testable). The grounded branch of
stream_grounded_answer() wraps a real paid OpenAI streaming call and is
verified manually instead (scripts/check_rag_stream.py), same convention
llm_stream.py uses for external paid APIs."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.rag_stream import (
    ABSTAIN_MESSAGE,
    RetrievalContext,
    build_grounded_prompt,
    is_grounded,
    retrieve_context,
    stream_grounded_answer,
)
from app.services.retrieval import RetrievedChunk


def _chunk(text="...", source="doc.pdf", index=0, score=0.9) -> RetrievedChunk:
    return RetrievedChunk(text=text, source_file=source, chunk_index=index, score=score)


# ── build_grounded_prompt ──────────────────────────────────────────────────


def test_build_grounded_prompt_includes_chunk_text_and_question():
    result = build_grounded_prompt("what is OLAP", [_chunk(text="OLAP enables pre-aggregation.")])
    assert "OLAP enables pre-aggregation." in result
    assert "what is OLAP" in result
    assert "CONTEXT:" in result
    assert "QUESTION:" in result


def test_build_grounded_prompt_joins_multiple_chunks_with_separator():
    result = build_grounded_prompt("q", [_chunk(text="first chunk"), _chunk(text="second chunk", index=1)])
    assert "first chunk" in result
    assert "second chunk" in result
    assert result.index("first chunk") < result.index("second chunk")


def test_build_grounded_prompt_handles_no_chunks_without_crashing():
    result = build_grounded_prompt("what is OLAP", [])
    assert "what is OLAP" in result
    assert "No relevant context" in result


def test_build_grounded_prompt_omits_history_block_when_absent():
    # The plain single-turn RAG path (no agent, no history) must be
    # byte-for-byte unaffected by the history feature added for the agent.
    result = build_grounded_prompt("q", [_chunk(text="chunk")])
    assert "CONVERSATION HISTORY" not in result


def test_build_grounded_prompt_includes_history_when_present():
    history = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]
    result = build_grounded_prompt("follow-up", [_chunk(text="chunk")], history)

    assert "CONVERSATION HISTORY" in result
    assert "earlier question" in result
    assert "earlier answer" in result
    assert "follow-up" in result


# ── is_grounded ────────────────────────────────────────────────────────────


def test_is_grounded_false_when_no_chunks():
    assert is_grounded([]) is False


def test_is_grounded_false_when_top_score_below_floor():
    # Real off-topic score observed against the actual corpus ("what is sales").
    with patch("app.services.rag_stream.settings") as mock_settings:
        mock_settings.retrieval_score_floor = 0.35
        assert is_grounded([_chunk(score=0.27)]) is False


def test_is_grounded_true_when_top_score_above_floor():
    # Real on-topic score observed against the actual corpus ("what is OLAP").
    with patch("app.services.rag_stream.settings") as mock_settings:
        mock_settings.retrieval_score_floor = 0.35
        assert is_grounded([_chunk(score=0.58)]) is True


def test_is_grounded_true_at_exact_floor_boundary():
    with patch("app.services.rag_stream.settings") as mock_settings:
        mock_settings.retrieval_score_floor = 0.35
        assert is_grounded([_chunk(score=0.35)]) is True


def test_is_grounded_only_checks_top_chunk_score():
    with patch("app.services.rag_stream.settings") as mock_settings:
        mock_settings.retrieval_score_floor = 0.35
        assert is_grounded([_chunk(score=0.9), _chunk(score=0.05, index=1)]) is True


# ── retrieve_context ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_context_builds_grounded_context_from_search():
    chunks = [_chunk(source="olap.pdf", score=0.58), _chunk(source="olap.pdf", index=1, score=0.5)]
    with (
        patch("app.services.rag_stream.search", AsyncMock(return_value=chunks)),
        patch("app.services.rag_stream.settings") as mock_settings,
    ):
        mock_settings.retrieval_score_floor = 0.35
        ctx = await retrieve_context("what is OLAP")

    assert ctx.top_score == 0.58
    assert ctx.grounded is True
    assert ctx.sources == ["olap.pdf"]  # deduped


@pytest.mark.asyncio
async def test_retrieve_context_marks_not_grounded_when_below_floor():
    with (
        patch("app.services.rag_stream.search", AsyncMock(return_value=[_chunk(score=0.27)])),
        patch("app.services.rag_stream.settings") as mock_settings,
    ):
        mock_settings.retrieval_score_floor = 0.35
        ctx = await retrieve_context("what is sales")

    assert ctx.top_score == 0.27
    assert ctx.grounded is False


@pytest.mark.asyncio
async def test_retrieve_context_handles_empty_search():
    with (
        patch("app.services.rag_stream.search", AsyncMock(return_value=[])),
        patch("app.services.rag_stream.settings") as mock_settings,
    ):
        mock_settings.retrieval_score_floor = 0.35
        ctx = await retrieve_context("nonsense")

    assert ctx.top_score == 0.0
    assert ctx.grounded is False
    assert ctx.sources == []


# ── stream_grounded_answer (abstain branch only -- no LLM call) ─────────────


@pytest.mark.asyncio
async def test_stream_grounded_answer_abstains_without_llm_when_not_grounded():
    context = RetrievalContext(chunks=[_chunk(score=0.27)], top_score=0.27, grounded=False)

    pieces = [delta async for delta in stream_grounded_answer("what is sales", context)]

    # Exactly the abstain message, and nothing else -- if it had tried to call
    # the (unmocked) OpenAI client, this test would error instead.
    assert pieces == [ABSTAIN_MESSAGE]
