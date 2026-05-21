"""Content-hash based skip logic for the parse stage: a PDF is only
re-parsed by LlamaParse (a paid, slow, cloud call) when its bytes have
actually changed since the last run. Pure logic + filesystem I/O, no
LlamaParse import here -- kept separate so it's testable without touching
the network, and reusable once chunking/ingestion need the same
"skip unchanged work" idiom.

Idempotency model: each parsed PDF gets a Markdown file plus a metadata
sidecar (source hash, job id, page/char counts) in the output dir. The
sidecar's source_hash is the single source of truth for "has this file
changed" -- no separate manifest to fall out of sync.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def compute_file_hash(file_path: Path) -> str:
    """SHA-256 of the file's bytes -- stable across renames/moves, changes
    the instant the PDF's content changes."""
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_path_for(output_dir: Path, source_file: Path) -> Path:
    return output_dir / f"{source_file.stem}.md"


def metadata_path_for(output_dir: Path, source_file: Path) -> Path:
    return output_dir / f"{source_file.stem}.meta.json"


def needs_parsing(output_dir: Path, source_file: Path, current_hash: str) -> bool:
    """True if source_file has no prior output, or its content hash has
    changed since it was last parsed. False -> safe to skip re-parsing."""
    meta_path = metadata_path_for(output_dir, source_file)
    if not meta_path.exists():
        return True

    try:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True  # corrupt/unreadable sidecar -- safest is to re-parse

    return previous.get("source_hash") != current_hash


@dataclass
class ParseMetadata:
    source_file: str
    source_hash: str
    parsed_at: str
    parser: str
    tier: str
    job_id: str | None
    page_count: int | None
    char_count: int
    markdown_file: str


def write_output(
    output_dir: Path,
    source_file: Path,
    markdown: str,
    *,
    source_hash: str,
    job_id: str | None,
    page_count: int | None,
    tier: str,
) -> ParseMetadata:
    """Write the parsed Markdown + its metadata sidecar. Both are written
    together so a crash between the two can never leave a stale sidecar
    pointing at content that was never actually written."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = markdown_path_for(output_dir, source_file)
    md_path.write_text(markdown, encoding="utf-8")

    metadata = ParseMetadata(
        source_file=source_file.name,
        source_hash=source_hash,
        parsed_at=datetime.now(timezone.utc).isoformat(),
        parser="llamaparse",
        tier=tier,
        job_id=job_id,
        page_count=page_count,
        char_count=len(markdown),
        markdown_file=md_path.name,
    )
    metadata_path_for(output_dir, source_file).write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata
