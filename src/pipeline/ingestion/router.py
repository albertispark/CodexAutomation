"""Stage 1 router and canonical ingestion data contracts."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import fitz

from pipeline.config import Settings
from pipeline.local_llm.ollama_client import OllamaClient


class FileKind(str, Enum):
    NATIVE_PDF = "native_pdf"
    SCANNED_PDF = "scanned_pdf"
    MIXED_PDF = "mixed_pdf"
    IMAGE = "image"
    EXCEL = "excel"
    UNSUPPORTED = "unsupported"


PDF_EXTS: frozenset[str] = frozenset({".pdf"})
IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)
EXCEL_EXTS: frozenset[str] = frozenset({".xlsx", ".xlsm"})
CSV_EXTS: frozenset[str] = frozenset({".csv"})
SUPPORTED_EXTS: frozenset[str] = PDF_EXTS | IMAGE_EXTS | EXCEL_EXTS | CSV_EXTS

PageAction = Literal["native_text", "ocr", "excel_sheet"]
Origin = Literal["native", "ocr", "excel"]


@dataclass(frozen=True)
class Document:
    """One 1-based page or physical spreadsheet sheet."""

    source_path: Path
    page_number: int
    label: str
    text: str
    origin: Origin


@dataclass(frozen=True)
class IngestedFile:
    """Canonical Stage-1 output."""

    doc_sha: str
    source_path: Path
    kind: FileKind
    documents: list[Document]
    reason: str = ""


@dataclass(frozen=True)
class PagePlan:
    page_number: int
    action: PageAction
    label: str


@dataclass(frozen=True)
class IngestPlan:
    source_path: Path
    kind: FileKind
    file_sha256: str
    size_mb: float
    pages: list[PagePlan] = field(default_factory=list)
    reason: str = ""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream-hash file bytes in constant memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unsupported(
    path: Path, file_sha: str, size_mb: float, reason: str
) -> IngestPlan:
    return IngestPlan(
        source_path=path,
        kind=FileKind.UNSUPPORTED,
        file_sha256=file_sha,
        size_mb=size_mb,
        reason=reason,
    )


def detect_kind(path: Path, cfg: Settings) -> IngestPlan:
    """Classify a file and produce its deterministic per-page plan; never raise."""
    path = Path(path)
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        file_sha = sha256_file(path)
    except (OSError, ValueError) as exc:
        fallback = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        return _unsupported(path, fallback, 0.0, f"unreadable file: {exc.__class__.__name__}")

    if size_mb > cfg.pipeline.max_file_mb:
        return _unsupported(
            path,
            file_sha,
            size_mb,
            "file exceeds pipeline.max_file_mb "
            f"({size_mb:.1f} MB > {cfg.pipeline.max_file_mb} MB)",
        )

    suffix = path.suffix.lower()
    if suffix == ".xls":
        return _unsupported(
            path,
            file_sha,
            size_mb,
            "legacy .xls is unsupported — convert the file to .xlsx",
        )
    if suffix in IMAGE_EXTS:
        return IngestPlan(
            path,
            FileKind.IMAGE,
            file_sha,
            size_mb,
            [PagePlan(1, "ocr", "page 1")],
        )
    if suffix in EXCEL_EXTS or suffix in CSV_EXTS:
        return IngestPlan(path, FileKind.EXCEL, file_sha, size_mb)
    if suffix not in PDF_EXTS:
        display = suffix or "<none>"
        return _unsupported(
            path, file_sha, size_mb, f"unsupported file extension: {display}"
        )

    try:
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                return _unsupported(
                    path, file_sha, size_mb, "encrypted or unreadable PDF"
                )
            if pdf.page_count == 0:
                return _unsupported(path, file_sha, size_mb, "PDF contains zero pages")
            pages: list[PagePlan] = []
            for index, page in enumerate(pdf):
                number = index + 1
                text = page.get_text("text")
                action: PageAction = (
                    "native_text"
                    if len(text.strip()) >= cfg.pipeline.scanned_page_min_chars
                    else "ocr"
                )
                pages.append(PagePlan(number, action, f"page {number}"))
    except Exception:
        return _unsupported(path, file_sha, size_mb, "encrypted or unreadable PDF")

    actions = {page.action for page in pages}
    if actions == {"native_text"}:
        kind = FileKind.NATIVE_PDF
    elif actions == {"ocr"}:
        kind = FileKind.SCANNED_PDF
    else:
        kind = FileKind.MIXED_PDF
    return IngestPlan(path, kind, file_sha, size_mb, pages)


def execute_plan(
    plan: IngestPlan, cfg: Settings, client: OllamaClient
) -> list[Document]:
    """Execute a previously classified plan and return sorted documents."""
    if plan.kind is FileKind.UNSUPPORTED:
        return []
    if plan.kind is FileKind.IMAGE:
        from pipeline.ingestion.ocr_gemma import ocr_image_file

        return ocr_image_file(plan.source_path, plan.file_sha256, cfg, client)
    if plan.kind is FileKind.EXCEL:
        from pipeline.ingestion.excel_reader import read_workbook

        return read_workbook(plan.source_path, cfg)

    documents: list[Document] = []
    native_numbers = [p.page_number for p in plan.pages if p.action == "native_text"]
    ocr_numbers = [p.page_number for p in plan.pages if p.action == "ocr"]
    if native_numbers:
        from pipeline.ingestion.pdf_native import extract_text, to_documents

        documents.extend(
            to_documents(plan.source_path, extract_text(plan.source_path, native_numbers))
        )
    if ocr_numbers:
        from pipeline.ingestion.ocr_gemma import ocr_pdf_pages

        documents.extend(
            ocr_pdf_pages(
                plan.source_path, ocr_numbers, plan.file_sha256, cfg, client
            )
        )
    return sorted(documents, key=lambda document: document.page_number)


def ingest(path: Path, settings: Settings, ollama: OllamaClient) -> IngestedFile:
    """The only Stage-1 entry point used by orchestration."""
    plan = detect_kind(path, settings)
    return IngestedFile(
        doc_sha=plan.file_sha256,
        source_path=path,
        kind=plan.kind,
        documents=execute_plan(plan, settings, ollama),
        reason=plan.reason,
    )
