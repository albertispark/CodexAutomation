from __future__ import annotations

from pathlib import Path

from pipeline.config import IndexConfig
from pipeline.indexing.chunker import TABLE_ROW_RE, chunk_documents, split_blocks
from pipeline.ingestion.router import Document, FileKind, IngestedFile


def _ingested(documents: list[Document]) -> IngestedFile:
    return IngestedFile("a" * 64, Path("source.pdf"), FileKind.NATIVE_PDF, documents)


def test_table_blocks_stay_atomic() -> None:
    blocks = split_blocks("Intro\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nEnd")
    assert blocks == [
        ("Intro", False),
        ("| A | B |\n| --- | --- |\n| 1 | 2 |", True),
        ("End", False),
    ]


def test_large_table_splits_between_rows_and_repeats_header() -> None:
    table = "\n".join(
        ["| Metric | Value |", "| --- | --- |"]
        + [f"| Revenue item {index} | {index * 1000} |" for index in range(60)]
    )
    doc = Document(Path("table.pdf"), 1, "page 1", table, "native")
    chunks = chunk_documents(_ingested([doc]), IndexConfig(chunk_tokens=64, chunk_overlap=8))
    assert len(chunks) > 2
    for chunk in chunks:
        lines = chunk.text.splitlines()
        assert lines[:2] == ["| Metric | Value |", "| --- | --- |"]
        assert all(TABLE_ROW_RE.fullmatch(line) for line in lines)
        assert chunk.token_estimate <= int(64 * 1.25)
        assert chunk.contains_table


def test_document_boundaries_are_hard_and_ids_are_wide() -> None:
    docs = [
        Document(Path("x"), 1, "page 1", "alpha " * 100, "native"),
        Document(Path("x"), 2, "page 2", "beta " * 100, "ocr"),
    ]
    chunks = chunk_documents(_ingested(docs), IndexConfig(chunk_tokens=64, chunk_overlap=8))
    assert all(chunk.page_start == chunk.page_end for chunk in chunks)
    assert {chunk.origin for chunk in chunks} == {"native", "ocr"}
    assert chunks[0].chunk_id.endswith(":000001")
    first_page2 = next(chunk for chunk in chunks if chunk.page_start == 2)
    assert "alpha" not in first_page2.text


def test_oversized_prose_is_bounded() -> None:
    prose = "A" * 2000
    doc = Document(Path("x"), 1, "page 1", prose, "native")
    chunks = chunk_documents(_ingested([doc]), IndexConfig(chunk_tokens=64, chunk_overlap=8))
    assert len(chunks) > 1
    assert max(chunk.token_estimate for chunk in chunks) <= 64
