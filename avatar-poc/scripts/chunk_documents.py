"""Manual CLI entry point for Stage 2 (chunking) of the ingestion pipeline:
reads every parsed Markdown file in data/output/ (Stage 1's output) and
splits it into retrieval-sized chunks, writing a <name>.chunks.jsonl sidecar
next to it (one JSON object per line) so the real chunk boundaries can be
inspected. JSONL rather than a single JSON array so Stage 3 (embed+upsert)
can stream chunks one at a time instead of loading a whole document's
chunks into memory at once.

Run from avatar-poc/ (after scripts.parse_documents has produced .md files):
    python -m scripts.chunk_documents

Unlike Stage 1, this involves no paid API -- pure local text processing --
so it always re-chunks and overwrites rather than skip-caching by content
hash; re-running is cheap and deterministic.
"""

import json
from pathlib import Path

from app.core.config import settings
from app.ingestion.chunking import chunk_markdown


def _chunk_one(md_path: Path) -> None:
    markdown = md_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(markdown)

    out_path = md_path.with_suffix(".chunks.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {"chunk_index": i, "char_count": len(chunk), "text": chunk}
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    sizes = [len(c) for c in chunks]
    avg_size = sum(sizes) // len(sizes) if sizes else 0
    print(
        f"DONE  {md_path.name} -> {out_path.name} "
        f"({len(chunks)} chunks, avg {avg_size} chars, "
        f"min {min(sizes, default=0)}, max {max(sizes, default=0)})"
    )


def main() -> None:
    output_dir = Path(settings.ingestion_output_dir)
    # Only files Stage 1 actually produced (i.e. have a metadata sidecar) --
    # `*.md` alone would also pick up unrelated files like the static
    # data/output/README.md placeholder.
    md_files = sorted(p for p in output_dir.glob("*.md") if p.with_suffix(".meta.json").exists())

    if not md_files:
        print(f"No parsed Markdown files found in {output_dir}/ -- run scripts.parse_documents first.")
        return

    print(f"Found {len(md_files)} parsed file(s) in {output_dir}/")
    for md_path in md_files:
        _chunk_one(md_path)


if __name__ == "__main__":
    main()
