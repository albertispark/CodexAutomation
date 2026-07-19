from pathlib import Path

from pipeline.ingestion.pdf_native import extract_text, to_documents

DATA = Path(__file__).parent / "data"


def test_extract_text_is_one_based_and_subsetted() -> None:
    pages = extract_text(DATA / "native_report.pdf", [2])
    assert [page.page_number for page in pages] == [2]
    assert "Balance Sheet" in pages[0].text
    documents = to_documents(DATA / "native_report.pdf", pages)
    assert documents[0].page_number == 2
    assert documents[0].label == "page 2"
    assert documents[0].origin == "native"
    assert documents[0].text == pages[0].text
