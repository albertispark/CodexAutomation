from pathlib import Path

from pipeline.ingestion.excel_reader import _sheet_to_text, read_workbook

DATA = Path(__file__).parent / "data"


def test_workbook_physical_sheet_ordinals_and_empty_skip(settings) -> None:
    documents = read_workbook(DATA / "financials.xlsx", settings)
    assert [document.page_number for document in documents] == [1, 3]
    assert [document.label for document in documents] == ["sheet:Income", "sheet:Balance"]
    assert all(document.origin == "excel" for document in documents)
    assert "| Metric | FY2024 | FY2025 |" in documents[0].text
    assert "| --- | --- | --- |" in documents[0].text


def test_csv_is_one_document(settings) -> None:
    documents = read_workbook(DATA / "financials.csv", settings)
    assert len(documents) == 1
    assert documents[0].page_number == 1
    assert documents[0].label == "sheet:financials"
    assert documents[0].origin == "excel"


def test_pipe_is_escaped(settings) -> None:
    import pandas as pd

    text = _sheet_to_text("Pipe", pd.DataFrame([["A|B", None]]))
    assert r"A\|B" in text
    assert "| A\\|B |  |" in text
