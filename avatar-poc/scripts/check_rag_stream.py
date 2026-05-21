"""Manual check: confirm stream_rag_response() actually retrieves relevant
chunks and streams a grounded answer (piece by piece), not a hallucinated
one. Prints each delta as it arrives, plus a timestamp, so streaming
behavior is visible -- mirrors scripts/check_llm_stream.py, but for the
retrieval-grounded path.

Usage: python -m scripts.check_rag_stream
       python -m scripts.check_rag_stream "your question here"
"""

import asyncio
import sys
import time

from app.services.rag_stream import stream_rag_response

DEFAULT_QUESTION = "What is the difference between OLAP and data mining?"


async def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f'Question: "{question}"\n')

    start = time.monotonic()
    piece_count = 0
    full_text = ""

    async for delta in stream_rag_response(question):
        elapsed = round(time.monotonic() - start, 2)
        piece_count += 1
        full_text += delta
        print(f"[{elapsed:>5}s] piece #{piece_count}: {delta!r}")

    print("\n--- full grounded answer ---")
    print(full_text)
    print(f"\nReceived {piece_count} pieces over {round(time.monotonic() - start, 2)}s")


if __name__ == "__main__":
    asyncio.run(main())
