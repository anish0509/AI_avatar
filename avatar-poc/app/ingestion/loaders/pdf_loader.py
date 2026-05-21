"""Parses PDFs into Markdown via LlamaParse's tier-based parse API, steered
by a general-purpose prompt tuned for high-fidelity RAG ingestion -- so
verbatim text, tables, structure, and Hindi/Hinglish content survive parsing
instead of being summarized, translated, or flattened. This is Stage 1 of
the ingestion pipeline (parse -> chunk -> embed+upsert), built one stage at
a time; only PDF is wired up now, but the module is split under loaders/ so
office/html/text loaders are a one-file addition later (mirrors
Multimodal_RAG's loaders/ shape -- see plan for the full reuse map).

Tier choice: `cost_effective` -- the middle tier (below `agentic` /
`agentic_plus`), still AI-backed so Markdown/table/heading structure is
preserved, unlike the zero-model `parse_page_without_llm` mode. Uses
`tier=` rather than the older `parse_mode=`/`model=` combo: LlamaParse's v2
API moved to tier-based selection and picks the underlying model itself, so
pinning a specific model name (e.g. `openai-gpt-4-1-mini`) breaks the moment
that model is deprecated on their end -- confirmed in practice: the first
real run against this exact model returned "410 Gone" / "Model
openai-gpt-4-1-mini is no longer supported ... We recommend migrating to
tiers." Billed entirely through LlamaCloud's own credit system
(LLAMA_CLOUD_API_KEY) -- no separate OpenAI/Anthropic key needed.

`version="latest"` is REQUIRED alongside `tier=` -- confirmed in practice:
omitting it doesn't error at request time, it silently returns a "done" job
with zero pages and empty text (error="Must specify a version with a tier",
error_code="MISSING_VERSION_FOR_TIER" -- visible only via result.status/
result.error, which the SDK does NOT raise on its own). That's also why
parse_pdf() below explicitly checks result.status/result.error itself
rather than trusting "no exception raised" to mean "parsing worked."

Not sales-specific: the corpus being ingested right now is general documents,
not exclusively Sanjay's sales-coaching material, so the parsing instruction
below stays domain-agnostic. If/when sales-specific parsing behavior is
needed (e.g. preserving scripts/case-study names as a first-class rule),
tighten this prompt then rather than presuming the domain now.

LlamaParse runs in the cloud (LLAMA_CLOUD_API_KEY required) and `.aparse()`
is natively async, so no thread-pool wrapping is needed.
"""

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

# Steers the model. Deliberately domain-agnostic (see module docstring) --
# these are general high-fidelity-RAG-ingestion rules, not tuned to any one
# document's subject matter. Summarizing, translating, or paraphrasing
# source content would quietly degrade whatever knowledge base this feeds
# into (the same "mixing quality levels silently degrades retrieval" concern
# implementation.md already raised for ASR).
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
