"""Unit tests for the content-hash skip logic (app/ingestion/parse_cache.py).
Pure filesystem + hashing -- no LlamaParse call, matching the project's
convention of never hitting real paid APIs in the automated suite."""

import json
from pathlib import Path

from app.ingestion.parse_cache import (
    compute_file_hash,
    markdown_path_for,
    metadata_path_for,
    needs_parsing,
    write_output,
)


def test_compute_file_hash_stable_for_same_content(tmp_path: Path) -> None:
    f1 = tmp_path / "a.pdf"
    f2 = tmp_path / "b.pdf"
    f1.write_bytes(b"same bytes")
    f2.write_bytes(b"same bytes")
    assert compute_file_hash(f1) == compute_file_hash(f2)


def test_compute_file_hash_changes_with_content(tmp_path: Path) -> None:
    f1 = tmp_path / "a.pdf"
    f1.write_bytes(b"version one")
    hash_before = compute_file_hash(f1)
    f1.write_bytes(b"version two")
    hash_after = compute_file_hash(f1)
    assert hash_before != hash_after


def test_needs_parsing_true_when_no_prior_output(tmp_path: Path) -> None:
    source = tmp_path / "input" / "deck.pdf"
    output_dir = tmp_path / "output"
    assert needs_parsing(output_dir, source, "somehash") is True


def test_needs_parsing_false_when_hash_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "input" / "deck.pdf"
    output_dir = tmp_path / "output"
    write_output(
        output_dir,
        source,
        "parsed markdown",
        source_hash="abc123",
        job_id="job-1",
        page_count=3,
        tier="cost_effective",
    )
    assert needs_parsing(output_dir, source, "abc123") is False


def test_needs_parsing_true_when_hash_changed(tmp_path: Path) -> None:
    source = tmp_path / "input" / "deck.pdf"
    output_dir = tmp_path / "output"
    write_output(
        output_dir,
        source,
        "parsed markdown",
        source_hash="abc123",
        job_id="job-1",
        page_count=3,
        tier="cost_effective",
    )
    assert needs_parsing(output_dir, source, "xyz789") is True


def test_needs_parsing_true_when_sidecar_corrupt(tmp_path: Path) -> None:
    source = tmp_path / "input" / "deck.pdf"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    metadata_path_for(output_dir, source).write_text("{not json", encoding="utf-8")
    assert needs_parsing(output_dir, source, "abc123") is True


def test_write_output_creates_markdown_and_metadata_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "input" / "deck.pdf"
    output_dir = tmp_path / "output"

    metadata = write_output(
        output_dir,
        source,
        "# Hello\n\nWorld",
        source_hash="abc123",
        job_id="job-1",
        page_count=2,
        tier="cost_effective",
    )

    md_path = markdown_path_for(output_dir, source)
    meta_path = metadata_path_for(output_dir, source)

    assert md_path.read_text(encoding="utf-8") == "# Hello\n\nWorld"
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved["source_hash"] == "abc123"
    assert saved["job_id"] == "job-1"
    assert saved["page_count"] == 2
    assert saved["char_count"] == len("# Hello\n\nWorld")
    assert metadata.markdown_file == md_path.name
