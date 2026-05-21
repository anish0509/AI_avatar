"""Buffers streamed LLM text into sentence-sized pieces, so speech can start
after the first sentence instead of waiting for the entire reply. Pure
logic, no API calls -- fully unit tested (see tests/test_text_chunker.py).
"""

SENTENCE_ENDERS = ".!?।"  # includes the Hindi danda, for Hindi/Hinglish content


class TextChunker:
    def __init__(self, max_chars: int = 300) -> None:
        self._buffer = ""
        self._max_chars = max_chars

    def feed(self, delta: str) -> list[str]:
        """Add a piece of streamed text; return any complete sentence(s) now ready to speak."""
        self._buffer += delta
        chunks: list[str] = []

        while True:
            split_at = self._find_sentence_end()
            if split_at is None:
                break
            chunk = self._buffer[: split_at + 1].strip()
            self._buffer = self._buffer[split_at + 1 :]
            if chunk:
                chunks.append(chunk)

        if len(self._buffer) >= self._max_chars:
            # No sentence end in sight and the buffer is growing unbounded
            # (e.g. a long run-on clause) -- force a flush so we don't
            # buffer forever and delay speech indefinitely.
            chunk = self._buffer.strip()
            self._buffer = ""
            if chunk:
                chunks.append(chunk)

        return chunks

    def flush(self) -> str | None:
        """Call once the LLM stream has ended; returns any leftover buffered text."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining or None

    def _find_sentence_end(self) -> int | None:
        """Index of the last character of the first confirmed sentence
        boundary in the buffer, or None if none is confirmed yet.

        A sentence-ender (or run of them, e.g. "...", "?!") only counts as a
        real boundary if it's followed by whitespace we've actually
        received -- if it's the last thing in the buffer so far, we can't
        yet tell "Mr." from "Mr. Smith", so we wait for more text (or the
        final flush()) instead of splitting prematurely.
        """
        i = 0
        n = len(self._buffer)
        while i < n:
            if self._buffer[i] in SENTENCE_ENDERS:
                j = i
                while j < n and self._buffer[j] in SENTENCE_ENDERS:
                    j += 1
                if j == n:
                    return None  # run reaches the end of buffered text -- wait for more
                if self._buffer[j].isspace():
                    return j - 1
                i = j  # e.g. "3.14" -- not whitespace after, not a real sentence end
                continue
            i += 1
        return None
