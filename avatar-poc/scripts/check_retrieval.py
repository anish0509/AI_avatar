"""Manual sanity check for Stage 3's output: runs a real semantic search
against the ingested Pinecone index and prints the top matches, proving the
whole parse -> chunk -> embed -> upsert pipeline actually produced
retrievable content (not just "no errors during ingest").

Usage:
    python -m scripts.check_retrieval "your question here"
    python -m scripts.check_retrieval               # uses a default query

Hits the real Pinecone API (paid) -- not part of the automated test suite,
same convention as the project's other real-API check scripts
(scripts/check_realtime_stt.py etc.).
"""

import sys

from app.core.config import settings
from app.ingestion.vector_store import ensure_index

DEFAULT_QUERY = "what is the difference between OLAP and data mining"


def main() -> None:
    query_text = " ".join(sys.argv[1:]) or DEFAULT_QUERY

    index = ensure_index()
    results = index.search(
        namespace=settings.pinecone_namespace,
        query={"inputs": {"text": query_text}, "top_k": 5},
        fields=["source_file", "chunk_index", "chunk_text"],
    )

    hits = results["result"]["hits"]
    print(f'Query: "{query_text}"')
    print(f"Namespace: {settings.pinecone_namespace}  |  {len(hits)} result(s)\n")

    if not hits:
        print("No results -- has scripts.ingest_documents been run yet?")
        return

    for rank, hit in enumerate(hits, start=1):
        source = hit.fields.get("source_file", "?")
        chunk_index = int(hit.fields.get("chunk_index", -1))
        text_preview = hit.fields.get("chunk_text", "")[:220].replace("\n", " ")
        print(f"#{rank}  score={hit.score:.4f}  {source} [chunk {chunk_index}]")
        print(f"     {text_preview}\n")


if __name__ == "__main__":
    main()
