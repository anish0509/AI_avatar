"""Search Pinecone for relevant document chunks. Run its synchronous client outside the event loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import logfire
from langsmith import traceable

from app.core.config import settings
from app.ingestion.vector_store import TEXT_FIELD, ensure_index


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    chunk_index: int
    score: float


def _search_blocking(query: str, resolved_top_k: int) -> list[RetrievedChunk]:
    """The actual synchronous Pinecone call. Runs on a worker thread (see
    search()), never on the event loop.

    Note `index.search()` with integrated inference does TWO things
    server-side: embeds `query` with the index's hosted model
    (llama-text-embed-v2), then searches. Pinecone's dashboard "Query"
    latency covers only the search half, which is why ~50-150ms there
    corresponds to ~1.3s here (the rest being embedding + round trip)."""
    index = ensure_index()
    results = index.search(
        namespace=settings.pinecone_namespace,
        query={"inputs": {"text": query}, "top_k": resolved_top_k},
        fields=["source_file", "chunk_index", TEXT_FIELD],
    )
    return [
        RetrievedChunk(
            text=hit.fields[TEXT_FIELD],
            source_file=hit.fields["source_file"],
            chunk_index=int(hit.fields["chunk_index"]),
            score=hit.score,
        )
        for hit in results["result"]["hits"]
    ]


@traceable(name="search", run_type="retriever")
async def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Dense semantic search against the ingested corpus. Returns the top_k
    most similar chunks, ranked by score descending (Pinecone's default).

    Async purely so the blocking SDK call can be pushed to a thread -- the
    search itself is no faster, it just stops holding the event loop hostage
    while it runs (see this module's docstring)."""
    resolved_top_k = top_k if top_k is not None else settings.retrieval_top_k
    with logfire.span("Pinecone dense search", query=query[:200], top_k=resolved_top_k):
        chunks = await asyncio.to_thread(_search_blocking, query, resolved_top_k)
        logfire.info("retrieved {count} chunks", count=len(chunks))
        return chunks
