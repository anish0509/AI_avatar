"""Splits parsed Markdown into retrieval-sized chunks. This is Stage 2 of
the ingestion pipeline (parse -> chunk -> embed+upsert), following Stage 1's
LlamaParse output (app/ingestion/loaders/pdf_loader.py).

Strategy, grounded in inspecting Stage 1's real output on 4 real PDFs (an
academic paper, a 244-page textbook, a blog post, a book chapter) rather
than assumed upfront:

- LlamaParse inserts a literal "---" line between every page. Confirmed in
  practice that these land mid-sentence and mid-table (e.g. "...critical
  for\n---\neffective application development." in one real document) --
  left in place, a naive splitter can produce a chunk boundary, or even a
  chunk's actual content, broken across a page seam. strip_page_breaks()
  removes these before chunking ever sees the text.
- Heading LEVEL is not a reliable structural signal in this corpus --
  confirmed the same chapter title renders as H1 in one place and H2 in
  another (LlamaParse infers levels per-page, not from one stable
  document-wide outline). A flat separator-priority list (try any heading
  level, then code fences, then paragraphs) is more robust to that than a
  strict level-aware hierarchy would be.
- Table-of-contents pages can produce heading-shaped lines (e.g. "### 1
  Introduction ... 3" with a trailing page number) that aren't real
  structure. Not filtered here -- accepted as minor, occasional noise in
  exchange for not building a fragile heuristic to detect them.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

# Matches one or more consecutive LlamaParse inter-page markers -- a lone
# "---" on its own line, optionally repeated back-to-back for a blank page.
_PAGE_BREAK_RE = re.compile(r"(?:\n-{3,})+\n")

# Header/code-fence-priority separators: prefer splitting at structural
# boundaries, falling back to paragraph/line/word/character splits only
# when no structural boundary is available nearby. All heading levels are
# tried (not just the outermost) since level itself isn't reliable here --
# see module docstring.
_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n```", "\n\n", "\n", " ", ""]


def strip_page_breaks(markdown: str) -> str:
    """Remove LlamaParse's inter-page '---' markers, rejoining the text
    that was split across a page boundary with a single space -- the
    marker carries no semantic meaning, it's a pagination artifact of the
    source PDF, not document structure. Must run before chunking."""
    return _PAGE_BREAK_RE.sub(" ", markdown)


def chunk_markdown(
    markdown: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split parsed Markdown into retrieval-sized chunks.

    Strips page-break artifacts first, then splits preferring structural
    boundaries (headings, code fences, paragraphs) over raw character
    cuts. Empty/whitespace-only chunks are dropped.
    """
    cleaned = strip_page_breaks(markdown)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=_SEPARATORS,
    )

    return [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]
