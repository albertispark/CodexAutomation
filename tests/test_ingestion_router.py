from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import fitz

from pipeline.ingestion.router import (
    CSV_EXTS,
    EXCEL_EXTS,
    IMAGE_EXTS,
    PDF_EXTS,
    SUPPORTED_EXTS,
    FileKind,
    detect_kind,
)

DATA = Path(__file__).parent / "data"


def test_supported_extensions_have_one_authority() -> None:
    assert SUPPORTED_EXTS == PDF_EXTS | IMAGE_EXTS | EXCEL_EXTS | CSV_EXTS
    assert ".xls" not in SUPPORTED_EXTS


def test_fixture_pdf_routing(settings) -> None:
    native = detect_kind(DATA / "native_report.pdf", settings)
    scanned = detect_kind(DATA / "scanned_report.pdf", settings)
    assert native.kind is FileKind.NATIVE_PDF
    assert [page.action for page in native.pages] == ["native_text", "native_text"]
    assert scanned.kind is FileKind.SCANNED_PDF
    assert [page.action for page in scanned.pages] == ["ocr"]


def test_mixed_pdf_is_planned_per_page(tmp_path: Path, settings) -> None:
    path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "native content long enough for routing")
    doc.new_page()
    doc.save(path)
    doc.close()
    plan = detect_kind(path, settings)
    assert plan.kind is FileKind.MIXED_PDF
    assert [page.action for page in plan.pages] == ["native_text", "ocr"]


def test_corrupt_pdf_and_xls_never_raise(tmp_path: Path, settings) -> None:
    corrupt = tmp_path / "bad.pdf"
    corrupt.write_bytes(b"not a PDF")
    assert detect_kind(corrupt, settings).kind is FileKind.UNSUPPORTED
    legacy = tmp_path / "legacy.xls"
    legacy.write_bytes(b"fake")
    plan = detect_kind(legacy, settings)
    assert plan.kind is FileKind.UNSUPPORTED
    assert "convert the file to .xlsx" in plan.reason


def test_size_guard_reason(tmp_path: Path, settings) -> None:
    settings.pipeline.max_file_mb = 1
    path = tmp_path / "large.csv"
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    plan = detect_kind(path, settings)
    assert plan.kind is FileKind.UNSUPPORTED
    assert "file exceeds pipeline.max_file_mb" in plan.reason


def test_fixture_generator_is_byte_deterministic(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(DATA / "make_fixtures.py"))
    namespace["main"].__globals__["ROOT"] = tmp_path
    namespace["main"]()
    names = [
        "native_report.pdf",
        "scanned_report.pdf",
        "financials.xlsx",
        "financials.csv",
        "encrypted.pdf",
    ]
    first = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in names
    }
    namespace["main"]()
    second = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in names
    }
    assert first == second
