"""Unit tests for app/ingestion/chunking.py. Pure logic, no API calls --
fixtures are modeled directly on real problems found inspecting Stage 1's
actual LlamaParse output (see chunking.py's module docstring)."""

from app.ingestion.chunking import chunk_markdown, strip_page_breaks


def test_strip_page_breaks_rejoins_mid_sentence_split() -> None:
    # The exact real bug found in a Stage 1 output file: a page boundary
    # landed mid-sentence.
    text = "...becomes critical for\n---\neffective application development."
    result = strip_page_breaks(text)
    assert "---" not in result
    assert "critical for effective application development" in result


def test_strip_page_breaks_removes_marker_between_blocks() -> None:
    text = "```\nSELECT * FROM T;\n```\n---\nThis query returns rows."
    result = strip_page_breaks(text)
    assert "---" not in result
    assert "```\nSELECT * FROM T;\n```" in result
    assert "This query returns rows." in result


def test_strip_page_breaks_handles_consecutive_markers() -> None:
    text = "before\n---\n---\nafter"
    result = strip_page_breaks(text)
    assert "---" not in result
    assert "before" in result
    assert "after" in result


def test_strip_page_breaks_noop_when_no_markers() -> None:
    text = "# Heading\n\nSome normal paragraph text with no page markers."
    assert strip_page_breaks(text) == text


def test_chunk_markdown_keeps_short_document_as_one_chunk() -> None:
    text = "# Title\n\nA short document well under the chunk size."
    chunks = chunk_markdown(text, chunk_size=1500, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_chunk_markdown_splits_long_document_on_headers() -> None:
    section_a = "Paragraph text. " * 50
    section_b = "More paragraph text. " * 50
    text = f"## Section A\n\n{section_a}\n\n## Section B\n\n{section_b}"

    chunks = chunk_markdown(text, chunk_size=300, chunk_overlap=20)

    assert len(chunks) > 1
    # The header should survive as a chunk boundary marker, not be
    # swallowed mid-word by a raw character cut.
    assert any(c.startswith("## Section B") for c in chunks)


def test_chunk_markdown_respects_small_chunk_size() -> None:
    text = "word " * 500
    chunks = chunk_markdown(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_markdown_drops_empty_chunks() -> None:
    text = "\n\n\n# Heading\n\n\n\nBody text.\n\n\n"
    chunks = chunk_markdown(text, chunk_size=1500, chunk_overlap=200)
    assert all(c.strip() for c in chunks)


def test_chunk_markdown_strips_page_breaks_before_splitting() -> None:
    text = "Intro paragraph.\n\n---\n\n## Next Section\n\nBody text here."
    chunks = chunk_markdown(text, chunk_size=1500, chunk_overlap=200)
    combined = "\n".join(chunks)
    assert "---" not in combined
    assert "Next Section" in combined
