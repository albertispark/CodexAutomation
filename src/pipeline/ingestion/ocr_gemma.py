"""Gemma OCR with content-hash page caching and a bounded quality fallback."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import fitz

from pipeline.config import Settings
from pipeline.ingestion.router import Document
from pipeline.local_llm.ollama_client import OllamaClient

OCR_PROMPT = """You are a precise OCR engine. Transcribe ALL visible text and numbers \
from this image into Markdown.
Rules:
1. Transcribe EVERYTHING: every word, number, header, footer, footnote, and label. \
Never summarize, never paraphrase, never omit content.
2. Preserve table structure as Markdown tables (| cell | cell |). Keep every row and \
every column, including empty cells.
3. Follow the page's reading order: top to bottom, left to right.
4. Reproduce numbers exactly as printed, including currency symbols, parentheses for \
negatives, thousands separators, decimals, and percent signs. Never normalize, round, \
or recompute a value.
5. If a cell, word, or number cannot be read with confidence, write [ILLEGIBLE] in its \
place. Never guess.
6. Output ONLY the Markdown transcription. No commentary, no preamble, no code fences."""

OCR_PROMPT_VERSION: str = "1"
DEFAULT_DPI = 200
DEGRADED_MIN_ALNUM_SPACE_RATIO = 0.5
DEGRADED_MIN_CHARS = 20
DEGRADED_MAX_ILLEGIBLE = 10
FALLBACK_DPI = 300
DEGRADED_FILE_QUARANTINE_RATIO = 0.5
LOW_CONFIDENCE_MARKER = "<!-- OCR_LOW_CONFIDENCE page={page_number} -->"


def render_page_png(page: fitz.Page, dpi: int = DEFAULT_DPI) -> bytes:
    return page.get_pixmap(dpi=dpi).tobytes("png")


def _cache_path(cfg: Settings, file_sha256: str, page_number: int) -> Path:
    return cfg.paths.ocr_cache / file_sha256 / f"page_{page_number}.json"


def _read_cache(path: Path, cfg: Settings) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("model") == cfg.ollama.ocr_model
            and data.get("prompt_version") == OCR_PROMPT_VERSION
            and isinstance(data.get("text"), str)
        ):
            return data["text"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink(missing_ok=True)


def _write_cache(path: Path, text: str, dpi: int, cfg: Settings) -> None:
    _atomic_json(
        path,
        {
            "text": text,
            "dpi": dpi,
            "model": cfg.ollama.ocr_model,
            "prompt_version": OCR_PROMPT_VERSION,
        },
    )


def _ends_mid_table(page_text: str) -> bool:
    lines = [line.strip() for line in page_text.rstrip().splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1]
    return last.startswith("|") and not last.endswith("|")


def _is_degraded(page_text: str) -> bool:
    total = len(page_text)
    ratio = (
        sum(char.isalnum() or char.isspace() for char in page_text) / total
        if total
        else 0.0
    )
    non_whitespace = sum(not char.isspace() for char in page_text)
    return (
        ratio < DEGRADED_MIN_ALNUM_SPACE_RATIO
        or non_whitespace < DEGRADED_MIN_CHARS
        or page_text.count("[ILLEGIBLE]") > DEGRADED_MAX_ILLEGIBLE
        or _ends_mid_table(page_text)
    )


def _ocr_page(
    page: fitz.Page, page_number: int, client: OllamaClient
) -> tuple[str, int]:
    first = client.ocr_image(render_page_png(page, DEFAULT_DPI), OCR_PROMPT)
    first_truncated = bool(getattr(client, "last_ocr_truncated", False))
    accepted = first
    accepted_dpi = DEFAULT_DPI
    accepted_truncated = first_truncated
    if _is_degraded(first) or first_truncated:
        second = client.ocr_image(render_page_png(page, FALLBACK_DPI), OCR_PROMPT)
        second_truncated = bool(getattr(client, "last_ocr_truncated", False))
        if len(second) > len(first):
            accepted = second
            accepted_dpi = FALLBACK_DPI
            accepted_truncated = second_truncated
    if _is_degraded(accepted) or accepted_truncated:
        accepted = f"{LOW_CONFIDENCE_MARKER.format(page_number=page_number)}\n{accepted}"
    return accepted, accepted_dpi


def ocr_pdf_pages(
    path: Path,
    page_numbers: list[int],
    file_sha256: str,
    cfg: Settings,
    client: OllamaClient,
) -> list[Document]:
    """OCR selected 1-based pages, serving valid page cache entries first."""
    output: list[Document] = []
    with fitz.open(path) as pdf:
        for number in sorted(page_numbers):
            cache_path = _cache_path(cfg, file_sha256, number)
            text = _read_cache(cache_path, cfg)
            if text is None:
                text, dpi = _ocr_page(pdf[number - 1], number, client)
                _write_cache(cache_path, text, dpi, cfg)
            output.append(Document(path, number, f"page {number}", text, "ocr"))
    return output


def ocr_image_file(
    path: Path, file_sha256: str, cfg: Settings, client: OllamaClient
) -> list[Document]:
    """OCR a standalone source image as one page."""
    cache_path = _cache_path(cfg, file_sha256, 1)
    text = _read_cache(cache_path, cfg)
    if text is None:
        text = client.ocr_image(path.read_bytes(), OCR_PROMPT)
        if _is_degraded(text) or bool(
            getattr(client, "last_ocr_truncated", False)
        ):
            text = f"{LOW_CONFIDENCE_MARKER.format(page_number=1)}\n{text}"
        _write_cache(cache_path, text, 0, cfg)
    return [Document(path, 1, "page 1", text, "ocr")]
