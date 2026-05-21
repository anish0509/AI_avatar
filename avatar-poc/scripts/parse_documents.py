"""Manual CLI entry point for Stage 1 (parsing) of the ingestion pipeline:
scans data/input/ for PDFs, skips any whose content hash hasn't changed
since the last run, and parses the rest via LlamaParse's cost_effective
tier -- writing Markdown + a metadata sidecar to data/output/ for each.

Drop PDFs into data/input/ and run:
    python -m scripts.parse_documents

Hits the real LlamaParse API (paid, cloud) -- not part of the automated
test suite, same convention as the project's other real-API check scripts
(scripts/check_realtime_stt.py etc.). One failing PDF is logged and skipped
rather than aborting the batch.
"""

import asyncio
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.loaders.pdf_loader import PARSE_TIER, parse_pdf
from app.ingestion.parse_cache import compute_file_hash, needs_parsing, write_output

logger = get_logger(__name__)


async def _parse_one(pdf_path: Path, output_dir: Path) -> None:
    current_hash = compute_file_hash(pdf_path)

    if not needs_parsing(output_dir, pdf_path, current_hash):
        print(f"SKIP  {pdf_path.name} (unchanged since last parse)")
        return

    print(f"PARSE {pdf_path.name} ...")
    try:
        parsed = await parse_pdf(pdf_path)
    except Exception as exc:
        logger.error(
            "parse failed, skipping file",
            extra={"node_name": "parse_documents", "source_file": pdf_path.name},
            exc_info=True,
        )
        print(f"FAIL  {pdf_path.name}: {exc}")
        return

    metadata = write_output(
        output_dir,
        pdf_path,
        parsed.markdown,
        source_hash=current_hash,
        job_id=parsed.job_id,
        page_count=parsed.page_count,
        tier=PARSE_TIER,
    )
    print(
        f"DONE  {pdf_path.name} -> {metadata.markdown_file} "
        f"({metadata.char_count} chars, {metadata.page_count or '?'} pages)"
    )


async def main() -> None:
    if not settings.llama_cloud_api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is not set in .env")

    input_dir = Path(settings.ingestion_input_dir)
    output_dir = Path(settings.ingestion_output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}/ -- drop some in and re-run.")
        return

    print(f"Found {len(pdfs)} PDF(s) in {input_dir}/")
    for pdf_path in pdfs:
        await _parse_one(pdf_path, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
