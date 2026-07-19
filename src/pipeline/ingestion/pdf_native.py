"""Native PDF text extraction with no cleanup or normalization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from pipeline.ingestion.router import Document


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def extract_text(path: Path, page_numbers: list[int] | None = None) -> list[PageText]:
    """Extract selected 1-based pages with one ``fitz.open`` call."""
    with fitz.open(path) as pdf:
        numbers = list(range(1, pdf.page_count + 1)) if page_numbers is None else sorted(page_numbers)
        pages: list[PageText] = []
        for number in numbers:
            if number < 1 or number > pdf.page_count:
                raise IndexError(f"PDF page number out of range: {number}")
            pages.append(PageText(number, pdf[number - 1].get_text("text")))
        return pages


def to_documents(path: Path, pages: list[PageText]) -> list[Document]:
    """Wrap private ``PageText`` values in the canonical data contract."""
    return [
        Document(path, page.page_number, f"page {page.page_number}", page.text, "native")
        for page in pages
    ]
