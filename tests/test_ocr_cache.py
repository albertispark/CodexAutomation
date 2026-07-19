from __future__ import annotations

import json
from pathlib import Path

from pipeline.ingestion.ocr_gemma import (
    FALLBACK_DPI,
    OCR_PROMPT_VERSION,
    _cache_path,
    _is_degraded,
    ocr_pdf_pages,
)

DATA = Path(__file__).parent / "data"


class OCRClient:
    def __init__(
        self, replies: list[str], truncated: list[bool] | None = None
    ) -> None:
        self.replies = replies
        self.truncated = list(truncated or [False] * len(replies))
        self.calls: list[bytes] = []
        self.last_ocr_truncated = False

    def ocr_image(self, image: bytes, prompt: str) -> str:
        self.calls.append(image)
        self.last_ocr_truncated = self.truncated.pop(0) if self.truncated else False
        return self.replies.pop(0)


def test_quality_gate_boundaries() -> None:
    assert _is_degraded("")
    assert _is_degraded("abc")
    assert _is_degraded("[ILLEGIBLE] " * 11 + "readable words and numbers 123")
    assert not _is_degraded("Revenue for fiscal year 2025 was 1,250 dollars")


def test_cache_roundtrip_skips_second_ocr(settings) -> None:
    reply = "Revenue for fiscal year 2025 was 1,250 dollars"
    client = OCRClient([reply])
    first = ocr_pdf_pages(DATA / "scanned_report.pdf", [1], "fixture", settings, client)
    second = ocr_pdf_pages(DATA / "scanned_report.pdf", [1], "fixture", settings, client)
    assert first == second
    assert len(client.calls) == 1


def test_model_or_prompt_mismatch_invalidates_but_dpi_does_not(settings) -> None:
    cache = _cache_path(settings, "mismatch", 1)
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps({"text": "cached valid transcription with enough characters", "dpi": 999,
                    "model": settings.ollama.ocr_model, "prompt_version": OCR_PROMPT_VERSION}),
        encoding="utf-8",
    )
    client = OCRClient([])
    ocr_pdf_pages(DATA / "scanned_report.pdf", [1], "mismatch", settings, client)
    assert client.calls == []
    data = json.loads(cache.read_text())
    data["model"] = "other"
    cache.write_text(json.dumps(data), encoding="utf-8")
    client.replies.append("fresh valid transcription with enough readable characters")
    ocr_pdf_pages(DATA / "scanned_report.pdf", [1], "mismatch", settings, client)
    assert len(client.calls) == 1


def test_degraded_retries_once_then_flags(settings) -> None:
    client = OCRClient(["bad", "still bad"])
    documents = ocr_pdf_pages(DATA / "scanned_report.pdf", [1], "degraded", settings, client)
    assert len(client.calls) == 2
    assert documents[0].text.startswith("<!-- OCR_LOW_CONFIDENCE page=1 -->")
    payload = json.loads(_cache_path(settings, "degraded", 1).read_text())
    assert payload["dpi"] in {200, FALLBACK_DPI}


def test_generation_ceiling_retries_even_when_text_looks_readable(settings) -> None:
    first = "Revenue for fiscal year 2025 was 1,250 dollars"
    second = first + " and net income was 125 dollars"
    client = OCRClient([first, second], truncated=[True, False])
    documents = ocr_pdf_pages(
        DATA / "scanned_report.pdf", [1], "truncated", settings, client
    )
    assert len(client.calls) == 2
    assert documents[0].text == second
