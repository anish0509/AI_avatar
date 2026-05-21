"""Content-hash based skip logic for the ingestion stage: a chunks file is
only re-embedded/re-upserted (a paid, per-token Pinecone operation) when
its content has actually changed since the last successful ingest. Mirrors
parse_cache.py's idempotency idiom -- reuses its hash utility -- applied to
the .chunks.jsonl file instead of the source PDF.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.parse_cache import compute_file_hash

__all__ = ["IngestMetadata", "compute_file_hash", "mark_ingested", "needs_ingesting"]


def _ingested_marker_path(chunks_path: Path) -> Path:
    return chunks_path.with_suffix(".ingested.json")


def needs_ingesting(chunks_path: Path, current_hash: str) -> bool:
    """True if chunks_path has never been ingested, or its content hash has
    changed since the last successful ingest. False -> safe to skip."""
    marker_path = _ingested_marker_path(chunks_path)
    if not marker_path.exists():
        return True

    try:
        previous = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True  # corrupt/unreadable marker -- safest is to re-ingest

    return previous.get("source_hash") != current_hash


@dataclass
class IngestMetadata:
    source_file: str
    source_hash: str
    ingested_at: str
    chunk_count: int
    namespace: str


def mark_ingested(
    chunks_path: Path,
    *,
    source_hash: str,
    chunk_count: int,
    namespace: str,
) -> IngestMetadata:
    """Record a successful ingest so the next run can skip unchanged work."""
    metadata = IngestMetadata(
        source_file=chunks_path.name,
        source_hash=source_hash,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=chunk_count,
        namespace=namespace,
    )
    _ingested_marker_path(chunks_path).write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata
