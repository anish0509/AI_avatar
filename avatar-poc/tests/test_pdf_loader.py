"""Unit test for app/ingestion/loaders/pdf_loader.py -- LlamaParse is
mocked (paid, cloud, non-deterministic), matching the project's convention
of never hitting real paid APIs in the automated suite. Verifies the
cost_effective tier + custom prompt are actually wired into the call, and
that the parsed result is shaped correctly for the cache/writer stage."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.loaders.pdf_loader import GENERAL_DOCUMENT_PROMPT, PARSE_TIER, parse_pdf


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResult:
    def __init__(
        self,
        texts: list[str],
        job_id: str,
        page_count: int,
        status: str | None = None,
        error: str | None = None,
        error_code: str | None = None,
    ) -> None:
        self._texts = texts
        self.job_id = job_id
        self.pages = list(range(page_count))
        self.status = status
        self.error = error
        self.error_code = error_code

    def get_markdown_documents(self):
        return [_FakeDoc(t) for t in self._texts]


@pytest.mark.asyncio
async def test_parse_pdf_uses_cost_effective_tier_and_domain_prompt(tmp_path: Path) -> None:
    pdf_path = tmp_path / "course.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    fake_result = _FakeResult(["# Module 1\n\nBody text"], job_id="job-42", page_count=5)
    mock_parser = MagicMock()
    mock_parser.aparse = AsyncMock(return_value=fake_result)

    with (
        patch("app.ingestion.loaders.pdf_loader.settings") as mock_settings,
        patch("app.ingestion.loaders.pdf_loader.LlamaParse", return_value=mock_parser) as mock_cls,
    ):
        mock_settings.llama_cloud_api_key = "fake-key"

        result = await parse_pdf(pdf_path)

    _, kwargs = mock_cls.call_args
    assert kwargs["tier"] == PARSE_TIER == "cost_effective"
    assert kwargs["version"] == "latest"
    assert kwargs["user_prompt"] == GENERAL_DOCUMENT_PROMPT
    assert kwargs["api_key"] == "fake-key"
    assert "model" not in kwargs
    assert "parse_mode" not in kwargs

    mock_parser.aparse.assert_awaited_once_with(str(pdf_path))
    assert result.markdown == "# Module 1\n\nBody text"
    assert result.job_id == "job-42"
    assert result.page_count == 5
    assert result.char_count == len("# Module 1\n\nBody text")


@pytest.mark.asyncio
async def test_parse_pdf_raises_clear_error_without_api_key(tmp_path: Path) -> None:
    pdf_path = tmp_path / "course.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("app.ingestion.loaders.pdf_loader.settings") as mock_settings:
        mock_settings.llama_cloud_api_key = ""
        with pytest.raises(RuntimeError, match="LLAMA_CLOUD_API_KEY"):
            await parse_pdf(pdf_path)


@pytest.mark.asyncio
async def test_parse_pdf_raises_on_job_level_error_status(tmp_path: Path) -> None:
    """Regression test: confirmed in practice that a server-side job failure
    (e.g. a bad request parameter) does NOT raise from `.aparse()` -- it
    comes back as a normal-looking result with empty pages/text and
    status="ERROR". Silently treating that as "no text extracted" would
    mask a real failure as an empty document."""
    pdf_path = tmp_path / "course.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    fake_result = _FakeResult(
        [],
        job_id="job-99",
        page_count=0,
        status="ERROR",
        error="Must specify a version with a tier. Tier: cost_effective",
        error_code="MISSING_VERSION_FOR_TIER",
    )
    mock_parser = MagicMock()
    mock_parser.aparse = AsyncMock(return_value=fake_result)

    with (
        patch("app.ingestion.loaders.pdf_loader.settings") as mock_settings,
        patch("app.ingestion.loaders.pdf_loader.LlamaParse", return_value=mock_parser),
    ):
        mock_settings.llama_cloud_api_key = "fake-key"

        with pytest.raises(RuntimeError, match="MISSING_VERSION_FOR_TIER"):
            await parse_pdf(pdf_path)


@pytest.mark.asyncio
async def test_parse_pdf_warns_but_does_not_raise_on_empty_extraction(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    fake_result = _FakeResult([""], job_id="job-1", page_count=1)
    mock_parser = MagicMock()
    mock_parser.aparse = AsyncMock(return_value=fake_result)

    with (
        patch("app.ingestion.loaders.pdf_loader.settings") as mock_settings,
        patch("app.ingestion.loaders.pdf_loader.LlamaParse", return_value=mock_parser),
    ):
        mock_settings.llama_cloud_api_key = "fake-key"

        result = await parse_pdf(pdf_path)

    assert result.markdown == ""
    assert result.char_count == 0
