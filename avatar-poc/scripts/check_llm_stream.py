"""Manual check: confirm stream_llm_response() actually delivers text
incrementally (piece by piece) rather than all at once. Prints each delta
as it arrives, plus a timestamp, so streaming behavior is visible.

Usage: python -m scripts.check_llm_stream
"""

import asyncio
import time

from app.services.llm_stream import stream_llm_response

TEST_PROMPT = "Give a 3-sentence sales tip about handling price objections."


async def main() -> None:
    start = time.monotonic()
    piece_count = 0
    full_text = ""

    async for delta in stream_llm_response(TEST_PROMPT):
        elapsed = round(time.monotonic() - start, 2)
        piece_count += 1
        full_text += delta
        print(f"[{elapsed:>5}s] piece #{piece_count}: {delta!r}")

    print("\n--- full response ---")
    print(full_text)
    print(f"\nReceived {piece_count} pieces over {round(time.monotonic() - start, 2)}s")


if __name__ == "__main__":
    asyncio.run(main())
