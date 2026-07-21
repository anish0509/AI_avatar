"""Parse PDFs into Markdown with LlamaParse. Tier-based parsing requires a version and explicit result-status checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llama_cloud_services import LlamaParse

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Middle tier: AI-backed (Markdown, table/heading reconstruction), cheaper
# than `agentic`/`agentic_plus`. LlamaCloud picks the underlying model.
PARSE_TIER = "cost_effective"

GENERAL_DOCUMENT_PROMPT = """You are parsing a document for a downstream RAG \
(retrieval-augmented generation) pipeline. Follow these rules strictly:
- Preserve all text VERBATIM -- exact wording matters for downstream retrieval \
and grounding; do not summarize, paraphrase, invent, or omit content.
- Preserve every table as a clean GitHub-flavored Markdown table; never flatten a \
table into prose.
- Preserve code blocks, structured data, or verbatim quotes exactly, inside fenced \
code blocks where appropriate.
- Preserve the heading hierarchy (#, ##, ###), including any section/chapter \
numbering, exactly as it appears in the source.
- Preserve any non-English text (e.g. Hindi/Hinglish) exactly as it appears -- do \
NOT translate or transliterate it.
- For diagrams, charts, or screenshots, emit a concise text description of what the \
figure conveys.
- Output clean Markdown only -- no commentary, no preamble."""


@dataclass
class ParsedDocument:
    """Result of parsing one PDF: the Markdown text plus enough metadata for
    the caller to write a sidecar JSON and skip re-parsing an unchanged file
    next time (see app/ingestion/parse_cache.py)."""

    markdown: str
    source_file: str
    job_id: str | None
    page_count: int | None
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.markdown)


def _build_parser() -> LlamaParse:
    if not settings.llama_cloud_api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is not set in .env")

    return LlamaParse(
        api_key=settings.llama_cloud_api_key,
        tier=PARSE_TIER,
        version="latest",
        user_prompt=GENERAL_DOCUMENT_PROMPT,
        high_res_ocr=True,
    )


async def parse_pdf(file_path: Path) -> ParsedDocument:
    """Parse one PDF asynchronously via LlamaParse's cost_effective tier.

    Raises whatever LlamaParse raises on failure (auth, unreadable file,
    quota), AND explicitly raises if the job itself completed with
    status="ERROR" -- confirmed in practice that a server-side job failure
    (e.g. a bad request parameter) does NOT raise from `.aparse()`; it comes
    back as a normal-looking result with empty pages/text and the real
    reason buried in result.error/result.error_code. Trusting "no exception"
    to mean "parsed successfully" would silently treat that as an empty
    document instead of a failure. The caller (scripts/parse_documents.py)
    is responsible for per-file error isolation so one bad PDF doesn't abort
    a batch -- the same discipline the ingestion project's per-video
    isolation uses.
    """
    parser = _build_parser()

    logger.info(
        "parsing pdf",
        extra={"node_name": "parse_pdf", "source_file": file_path.name},
    )

    result = await parser.aparse(str(file_path))

    if getattr(result, "status", None) == "ERROR":
        raise RuntimeError(
            f"LlamaParse job failed for {file_path.name}: "
            f"{getattr(result, 'error_code', None)} - {getattr(result, 'error', None)}"
        )

    documents = result.get_markdown_documents()
    markdown = "\n\n".join(doc.text for doc in documents if doc.text)

    page_count = len(result.pages) if getattr(result, "pages", None) else None
    job_id = getattr(result, "job_id", None)

    if not markdown.strip():
        logger.warning(
            "no text extracted from pdf",
            extra={"node_name": "parse_pdf", "source_file": file_path.name},
        )

    logger.info(
        "pdf parsed",
        extra={
            "node_name": "parse_pdf",
            "source_file": file_path.name,
            "char_count": len(markdown),
            "page_count": page_count,
        },
    )

    return ParsedDocument(
        markdown=markdown,
        source_file=file_path.name,
        job_id=job_id,
        page_count=page_count,
    )
