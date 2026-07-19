"""Excel/CSV ingestion rendered as standard Markdown pipe tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import Settings
from pipeline.ingestion.router import CSV_EXTS, Document


class UnreadableWorkbookError(RuntimeError):
    """A spreadsheet or CSV cannot be parsed."""


def _cell_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return (
        str(value)
        .replace("|", r"\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "<br>")
    )


def _row(values: list[object]) -> str:
    return "| " + " | ".join(_cell_text(value) for value in values) + " |"


def _sheet_to_text(sheet_name: str, df: pd.DataFrame) -> str:
    """Render a raw sheet with row 1 plus a Markdown separator and all rows."""
    rows = [_row(list(row)) for row in df.itertuples(index=False, name=None)]
    if not rows:
        return f"### Sheet: {sheet_name}\n"
    separator = "| " + " | ".join("---" for _ in range(len(df.columns))) + " |"
    return "\n".join([f"### Sheet: {sheet_name}", rows[0], separator, *rows[1:]])


def _is_fully_empty(df: pd.DataFrame) -> bool:
    return df.empty or df.dropna(axis=0, how="all").empty


def read_workbook(path: Path, cfg: Settings) -> list[Document]:
    """Read CSV or all nonempty workbook sheets while retaining physical ordinals."""
    del cfg
    try:
        if path.suffix.lower() in CSV_EXTS:
            frame = pd.read_csv(path, header=None, dtype=object)
            return [
                Document(
                    path,
                    1,
                    f"sheet:{path.stem}",
                    _sheet_to_text(path.stem, frame),
                    "excel",
                )
            ]
        sheets = pd.read_excel(
            path, sheet_name=None, header=None, dtype=object, engine="openpyxl"
        )
    except Exception as exc:
        raise UnreadableWorkbookError(f"unreadable_workbook: {path.name}") from exc

    documents: list[Document] = []
    for physical_ordinal, (name, frame) in enumerate(sheets.items(), start=1):
        if _is_fully_empty(frame):
            continue
        documents.append(
            Document(
                path,
                physical_ordinal,
                f"sheet:{name}",
                _sheet_to_text(name, frame),
                "excel",
            )
        )
    return documents
