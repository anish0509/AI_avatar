"""Unit tests for app/ingestion/ingest_cache.py. Pure filesystem + hashing
-- no Pinecone call, matching the project's convention of never hitting
real paid APIs in the automated suite."""

import json
from pathlib import Path

from app.ingestion.ingest_cache import mark_ingested, needs_ingesting


def test_needs_ingesting_true_when_never_ingested(tmp_path: Path) -> None:
    chunks_path = tmp_path / "doc.chunks.jsonl"
    chunks_path.write_text('{"chunk_index": 0}', encoding="utf-8")
    assert needs_ingesting(chunks_path, "somehash") is True


def test_needs_ingesting_false_when_hash_unchanged(tmp_path: Path) -> None:
    chunks_path = tmp_path / "doc.chunks.jsonl"
    chunks_path.write_text('{"chunk_index": 0}', encoding="utf-8")

    mark_ingested(chunks_path, source_hash="abc123", chunk_count=1, namespace="avatar-poc")

    assert needs_ingesting(chunks_path, "abc123") is False


def test_needs_ingesting_true_when_hash_changed(tmp_path: Path) -> None:
    chunks_path = tmp_path / "doc.chunks.jsonl"
    chunks_path.write_text('{"chunk_index": 0}', encoding="utf-8")

    mark_ingested(chunks_path, source_hash="abc123", chunk_count=1, namespace="avatar-poc")

    assert needs_ingesting(chunks_path, "xyz789") is True


def test_needs_ingesting_true_when_marker_corrupt(tmp_path: Path) -> None:
    chunks_path = tmp_path / "doc.chunks.jsonl"
    chunks_path.write_text('{"chunk_index": 0}', encoding="utf-8")
    marker_path = chunks_path.with_suffix(".ingested.json")
    marker_path.write_text("{not json", encoding="utf-8")

    assert needs_ingesting(chunks_path, "abc123") is True


def test_mark_ingested_writes_expected_fields(tmp_path: Path) -> None:
    chunks_path = tmp_path / "doc.chunks.jsonl"
    chunks_path.write_text('{"chunk_index": 0}', encoding="utf-8")

    metadata = mark_ingested(chunks_path, source_hash="abc123", chunk_count=42, namespace="avatar-poc")

    marker_path = chunks_path.with_suffix(".ingested.json")
    saved = json.loads(marker_path.read_text(encoding="utf-8"))
    assert saved["source_hash"] == "abc123"
    assert saved["chunk_count"] == 42
    assert saved["namespace"] == "avatar-poc"
    assert metadata.source_file == chunks_path.name
