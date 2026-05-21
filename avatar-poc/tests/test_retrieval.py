"""Unit tests for app/services/retrieval.py -- Pinecone is mocked (paid,
cloud), matching the project's convention of never hitting real paid APIs
in the automated suite.

Uses create_autospec() against the real pinecone.index.Index class, not a
plain MagicMock -- the same discipline adopted after Stage 3's
upsert_records() signature bug slipped past a plain-MagicMock test."""

from unittest.mock import create_autospec, patch

import pytest
from pinecone.index import Index

from app.services.retrieval import TEXT_FIELD, search


def _make_hit(text: str, source_file: str, chunk_index: int, score: float):
    hit = type("FakeHit", (), {})()
    hit.score = score
    hit.fields = {TEXT_FIELD: text, "source_file": source_file, "chunk_index": chunk_index}
    return hit


@pytest.mark.asyncio
async def test_search_uses_configured_top_k_and_namespace():
    mock_index = create_autospec(Index, instance=True)
    mock_index.search.return_value = {"result": {"hits": []}}

    with (
        patch("app.services.retrieval.ensure_index", return_value=mock_index),
        patch("app.services.retrieval.settings") as mock_settings,
    ):
        mock_settings.pinecone_namespace = "avatar-poc"
        mock_settings.retrieval_top_k = 3

        await search("what is OLAP")

    _, kwargs = mock_index.search.call_args
    assert kwargs["namespace"] == "avatar-poc"
    assert kwargs["query"] == {"inputs": {"text": "what is OLAP"}, "top_k": 3}
    assert "rerank" not in kwargs


@pytest.mark.asyncio
async def test_search_explicit_top_k_overrides_setting():
    mock_index = create_autospec(Index, instance=True)
    mock_index.search.return_value = {"result": {"hits": []}}

    with (
        patch("app.services.retrieval.ensure_index", return_value=mock_index),
        patch("app.services.retrieval.settings") as mock_settings,
    ):
        mock_settings.pinecone_namespace = "avatar-poc"
        mock_settings.retrieval_top_k = 3

        await search("what is OLAP", top_k=7)

    _, kwargs = mock_index.search.call_args
    assert kwargs["query"]["top_k"] == 7


@pytest.mark.asyncio
async def test_search_maps_hits_to_retrieved_chunks():
    mock_index = create_autospec(Index, instance=True)
    mock_index.search.return_value = {
        "result": {
            "hits": [
                _make_hit("chunk one text", "doc.pdf", 0, 0.91),
                _make_hit("chunk two text", "doc.pdf", 1, 0.78),
            ]
        }
    }

    with (
        patch("app.services.retrieval.ensure_index", return_value=mock_index),
        patch("app.services.retrieval.settings") as mock_settings,
    ):
        mock_settings.pinecone_namespace = "avatar-poc"
        mock_settings.retrieval_top_k = 3

        results = await search("anything")

    assert len(results) == 2
    assert results[0].text == "chunk one text"
    assert results[0].source_file == "doc.pdf"
    assert results[0].chunk_index == 0
    assert results[0].score == 0.91


@pytest.mark.asyncio
async def test_search_returns_empty_list_when_no_hits():
    mock_index = create_autospec(Index, instance=True)
    mock_index.search.return_value = {"result": {"hits": []}}

    with (
        patch("app.services.retrieval.ensure_index", return_value=mock_index),
        patch("app.services.retrieval.settings") as mock_settings,
    ):
        mock_settings.pinecone_namespace = "avatar-poc"
        mock_settings.retrieval_top_k = 3

        assert await search("anything") == []


@pytest.mark.asyncio
async def test_search_runs_the_blocking_sdk_call_off_the_event_loop():
    """Regression: the Pinecone SDK is synchronous, and calling it directly
    from async code froze the whole event loop for its duration -- 1.3s warm,
    3.9s cold, measured. That didn't just delay retrieval; it stopped every
    other task, including the HeyGen session setup avatar_routes.py starts
    early SPECIFICALLY so it can overlap retrieval. A real trace showed
    HeyGen setup beginning 13ms after retrieval finished instead of
    alongside it. See P18 in architecture-and-query-flow.md.

    Asserts on the THREAD the SDK call lands on rather than timing it: an
    earlier version of this test counted event-loop ticks during a slow
    search and passed even with the blocking call restored, because
    langsmith's @traceable awaits around the call and yields anyway. Thread
    identity has no such loophole.
    """
    import threading

    loop_thread = threading.get_ident()
    sdk_call_thread = {}

    mock_index = create_autospec(Index, instance=True)

    def record_thread(*_args, **_kwargs):
        sdk_call_thread["id"] = threading.get_ident()
        return {"result": {"hits": []}}

    mock_index.search.side_effect = record_thread

    with (
        patch("app.services.retrieval.ensure_index", return_value=mock_index),
        patch("app.services.retrieval.settings") as mock_settings,
    ):
        mock_settings.pinecone_namespace = "avatar-poc"
        mock_settings.retrieval_top_k = 3

        await search("anything")

    assert sdk_call_thread["id"] != loop_thread, (
        "Pinecone's blocking SDK call ran on the event loop thread -- it will "
        "freeze every other task for the duration of the search"
    )
