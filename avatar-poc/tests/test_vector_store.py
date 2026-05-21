"""Unit tests for app/ingestion/vector_store.py -- Pinecone is mocked
(paid, cloud), matching the project's convention of never hitting real
paid APIs in the automated suite.

Mocks use create_autospec() against the REAL Pinecone/Index classes, not
plain MagicMock -- a real bug slipped through here once already: the first
version of upsert_chunks() called upsert_records() positionally
(index.upsert_records(namespace, records)), but the actual installed SDK
requires both arguments to be keyword-only. A plain MagicMock happily
accepts any call shape and the test passed anyway; autospec enforces the
real signature, so a regression like that fails the test instead of only
surfacing in a real (paid, slow) run.
"""

from unittest.mock import create_autospec, patch

import pytest
from pinecone import Pinecone
from pinecone.index import Index

import app.ingestion.vector_store as vector_store
from app.ingestion.vector_store import TEXT_FIELD, ensure_index, upsert_chunks


@pytest.fixture(autouse=True)
def _reset_singletons():
    # Module-level lazy singletons must not leak state across tests.
    vector_store._pc = None
    vector_store._index = None
    yield
    vector_store._pc = None
    vector_store._index = None


def test_get_client_raises_clear_error_without_api_key():
    with patch("app.ingestion.vector_store.settings") as mock_settings:
        mock_settings.pinecone_api_key = ""
        with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
            vector_store._get_client()


def test_ensure_index_creates_index_when_missing():
    mock_pc = create_autospec(Pinecone, instance=True)
    mock_pc.has_index.return_value = False
    mock_pc.describe_index.return_value.host = "fake-host.pinecone.io"

    with (
        patch("app.ingestion.vector_store.settings") as mock_settings,
        patch("app.ingestion.vector_store.Pinecone", return_value=mock_pc),
    ):
        mock_settings.pinecone_api_key = "fake-key"
        mock_settings.pinecone_index_name = "avatar-poc-content"
        mock_settings.embedding_model = "llama-text-embed-v2"

        ensure_index()

    mock_pc.has_index.assert_called_once_with("avatar-poc-content")
    mock_pc.create_index_for_model.assert_called_once()
    _, kwargs = mock_pc.create_index_for_model.call_args
    assert kwargs["name"] == "avatar-poc-content"
    assert kwargs["embed"]["model"] == "llama-text-embed-v2"
    assert kwargs["embed"]["field_map"] == {"text": TEXT_FIELD}
    mock_pc.Index.assert_called_once_with(host="fake-host.pinecone.io")


def test_ensure_index_skips_creation_when_index_exists():
    mock_pc = create_autospec(Pinecone, instance=True)
    mock_pc.has_index.return_value = True
    mock_pc.describe_index.return_value.host = "fake-host.pinecone.io"

    with (
        patch("app.ingestion.vector_store.settings") as mock_settings,
        patch("app.ingestion.vector_store.Pinecone", return_value=mock_pc),
    ):
        mock_settings.pinecone_api_key = "fake-key"
        mock_settings.pinecone_index_name = "avatar-poc-content"

        ensure_index()

    mock_pc.create_index_for_model.assert_not_called()


def test_ensure_index_caches_connection_across_calls():
    mock_pc = create_autospec(Pinecone, instance=True)
    mock_pc.has_index.return_value = True
    mock_pc.describe_index.return_value.host = "fake-host.pinecone.io"

    with (
        patch("app.ingestion.vector_store.settings") as mock_settings,
        patch("app.ingestion.vector_store.Pinecone", return_value=mock_pc),
    ):
        mock_settings.pinecone_api_key = "fake-key"
        mock_settings.pinecone_index_name = "avatar-poc-content"

        ensure_index()
        ensure_index()

    mock_pc.describe_index.assert_called_once()


def test_upsert_chunks_calls_upsert_records_with_keyword_args():
    mock_index = create_autospec(Index, instance=True)
    mock_index.upsert_records.return_value.record_count = 1

    mock_pc = create_autospec(Pinecone, instance=True)
    mock_pc.has_index.return_value = True
    mock_pc.describe_index.return_value.host = "fake-host.pinecone.io"
    mock_pc.Index.return_value = mock_index

    records = [{"_id": "doc:0", TEXT_FIELD: "hello world"}]

    with (
        patch("app.ingestion.vector_store.settings") as mock_settings,
        patch("app.ingestion.vector_store.Pinecone", return_value=mock_pc),
    ):
        mock_settings.pinecone_api_key = "fake-key"
        mock_settings.pinecone_index_name = "avatar-poc-content"
        mock_settings.pinecone_namespace = "avatar-poc"

        result = upsert_chunks(records)

    mock_index.upsert_records.assert_called_once_with(records=records, namespace="avatar-poc")
    assert result == 1


def test_upsert_chunks_noop_on_empty_list():
    with patch("app.ingestion.vector_store.settings") as mock_settings:
        mock_settings.pinecone_api_key = "fake-key"
        result = upsert_chunks([])

    assert result == 0
    assert vector_store._pc is None
    assert vector_store._index is None
