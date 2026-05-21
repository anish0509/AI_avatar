"""Pinecone wrapper for Stage 3 (embed + upsert) of the ingestion pipeline,
following Stage 2's chunking (app/ingestion/chunking.py).

Uses Pinecone's INTEGRATED INFERENCE: chunk text goes in, Pinecone embeds it
server-side with the configured hosted model (llama-text-embed-v2) -- no
local embedding model, no torch, no separate embedding API call. Keeps the
dependency footprint to just the `pinecone` SDK.

Deliberately a SEPARATE index from ai-avatar-backend's ai-avatar-content --
that index is reserved for the production sales-coaching knowledge base;
this POC's general-document demo corpus stays clearly apart from it.
"""

from __future__ import annotations

from pinecone import Pinecone

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# The record field Pinecone's integrated inference reads chunk text from.
# Must match the index's embed.field_map at creation time.
TEXT_FIELD = "chunk_text"

_pc: Pinecone | None = None
_index = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set in .env")
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def ensure_index():
    """Create the index (integrated inference) if it doesn't already exist,
    and return a connected Index handle. Index creation is asynchronous on
    Pinecone's side; describe_index() is called fresh each time this needs
    to (re)connect so a newly-created index's host is always resolved
    correctly rather than assumed."""
    global _index
    if _index is not None:
        return _index

    pc = _get_client()

    if not pc.has_index(settings.pinecone_index_name):
        logger.info(
            "creating pinecone index",
            extra={"node_name": "vector_store", "index_name": settings.pinecone_index_name},
        )
        pc.create_index_for_model(
            name=settings.pinecone_index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": settings.embedding_model,
                "field_map": {"text": TEXT_FIELD},
            },
        )

    description = pc.describe_index(settings.pinecone_index_name)
    _index = pc.Index(host=description.host)
    return _index


def upsert_chunks(records: list[dict]) -> int:
    """Upsert chunk records into the configured namespace. Returns the
    number of records Pinecone confirmed it received.

    Each record must include "_id" (a deterministic ID so re-running
    overwrites rather than duplicates) and TEXT_FIELD ("chunk_text"); any
    other keys are stored as metadata alongside the embedded vector.

    upsert_records()'s real signature (confirmed by introspecting the
    installed SDK, since the public docs described an older positional-arg
    shape that no longer matches) takes `records` and `namespace` as
    KEYWORD-ONLY arguments.
    """
    if not records:
        return 0

    index = ensure_index()
    response = index.upsert_records(records=records, namespace=settings.pinecone_namespace)
    return response.record_count
