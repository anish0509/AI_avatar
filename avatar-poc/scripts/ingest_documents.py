"""Manual CLI entry point for Stage 3 (ingestion) of the pipeline: reads
every *.chunks.jsonl file in data/output/ (Stage 2's output) and upserts
each chunk into Pinecone via integrated inference (Pinecone embeds the text
server-side -- no local model, no separate embedding API call).

Run from avatar-poc/ (after scripts.chunk_documents has produced .chunks.jsonl files):
    python -m scripts.ingest_documents

Hits the real Pinecone API (paid, per-token embedding) -- not part of the
automated test suite. Skips files whose content hasn't changed since the
last successful ingest (unlike Stage 2's free local chunking, this is a
billed operation, so skip-caching matters here). Vector IDs are
deterministic (f"{source_stem}:{chunk_index}") so re-running on unchanged
content overwrites rather than duplicates.
"""

import json
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.ingest_cache import compute_file_hash, mark_ingested, needs_ingesting
from app.ingestion.vector_store import TEXT_FIELD, upsert_chunks

logger = get_logger(__name__)

# Conservative default for Pinecone's integrated-inference upsert_records
# per-call record cap -- not yet confirmed against the real API for this
# corpus's chunk sizes; adjust if a real run reports a batch-too-large error.
_UPSERT_BATCH_SIZE = 90


def _ingest_one(chunks_path: Path) -> None:
    current_hash = compute_file_hash(chunks_path)

    if not needs_ingesting(chunks_path, current_hash):
        print(f"SKIP  {chunks_path.name} (unchanged since last ingest)")
        return

    source_stem = chunks_path.stem.removesuffix(".chunks")
    records = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            records.append(
                {
                    "_id": f"{source_stem}:{chunk['chunk_index']}",
                    TEXT_FIELD: chunk["text"],
                    "source_file": source_stem,
                    "chunk_index": chunk["chunk_index"],
                    "char_count": chunk["char_count"],
                }
            )

    total_batches = (len(records) + _UPSERT_BATCH_SIZE - 1) // _UPSERT_BATCH_SIZE
    print(f"INGEST {chunks_path.name} ({len(records)} chunks, {total_batches} batch(es)) ...")
    try:
        for batch_num, i in enumerate(range(0, len(records), _UPSERT_BATCH_SIZE), start=1):
            batch = records[i : i + _UPSERT_BATCH_SIZE]
            print(f"  batch {batch_num}/{total_batches}: upserting {len(batch)} chunks ...")
            confirmed = upsert_chunks(batch)
            logger.info(
                "batch upserted",
                extra={
                    "node_name": "ingest_documents",
                    "source_file": chunks_path.name,
                    "batch": f"{batch_num}/{total_batches}",
                    "records_sent": len(batch),
                    "records_confirmed": confirmed,
                },
            )
            print(f"  batch {batch_num}/{total_batches}: confirmed {confirmed} chunks upserted")
    except Exception as exc:
        logger.error(
            "ingest failed, skipping file",
            extra={"node_name": "ingest_documents", "source_file": chunks_path.name},
            exc_info=True,
        )
        print(f"FAIL  {chunks_path.name}: {exc}")
        return

    mark_ingested(
        chunks_path,
        source_hash=current_hash,
        chunk_count=len(records),
        namespace=settings.pinecone_namespace,
    )
    print(f"DONE  {chunks_path.name} -> {len(records)} chunks upserted to namespace '{settings.pinecone_namespace}'")


def main() -> None:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")

    output_dir = Path(settings.ingestion_output_dir)
    chunks_files = sorted(output_dir.glob("*.chunks.jsonl"))

    if not chunks_files:
        print(f"No chunk files found in {output_dir}/ -- run scripts.chunk_documents first.")
        return

    print(f"Found {len(chunks_files)} chunk file(s) in {output_dir}/")
    for chunks_path in chunks_files:
        _ingest_one(chunks_path)


if __name__ == "__main__":
    main()
