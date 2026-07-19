"""Canonical local/cloud payload models and JSON-schema exports."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StatementType(str, Enum):
    income_statement = "income_statement"
    balance_sheet = "balance_sheet"
    cash_flow = "cash_flow"
    other = "other"


class FinancialFigure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(
        default="",
        description=(
            "Stable id 'F0001', 'F0002', ... assigned by the Bouncer after "
            "validation; never emitted by the local model."
        ),
    )
    label: str = Field(description="Line-item name exactly as printed.")
    value: float | None = Field(
        description=(
            "Number copied verbatim with symbols/separators stripped and "
            "parentheses interpreted as negative; null if unreadable."
        )
    )
    unit: str = Field(
        description=(
            "One of: absolute, thousands, millions, billions, percent, ratio, shares."
        )
    )
    currency: str | None = Field(description="ISO-4217 code if stated, else null.")
    period: str = Field(description="Reporting period exactly as labeled.")
    statement: StatementType = Field(description="Financial statement classification.")
    source_page: int | None = Field(
        description="1-based source page or sheet ordinal, null if unknown."
    )
    verbatim_context: str = Field(
        max_length=200,
        description=(
            "Exact local-only source snippet containing the figure; excluded "
            "from all outbound serialization."
        ),
    )


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str | None = Field(description="Company name as printed, or null.")
    doc_type: str = Field(description="Document type, or unknown.")
    currency_default: str | None = Field(
        description="Document-wide ISO-4217 currency, or null."
    )
    periods: list[str] = Field(description="All distinct reporting periods found.")
    figures: list[FinancialFigure] = Field(description="Every raw figure found.")
    warnings: list[str] = Field(description="Extraction ambiguities or OCR warnings.")


@dataclass(frozen=True)
class RedactionEvent:
    pattern_name: str
    field_path: str
    match_preview: str


SCHEMA_PATH = Path("config/schemas/financial_payload.schema.json")


def payload_json_schema() -> dict:
    return ExtractionPayload.model_json_schema()


def _remove_figure_id(node: Any) -> None:
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict) and "figure_id" in properties:
            properties.pop("figure_id", None)
            required = node.get("required")
            if isinstance(required, list):
                node["required"] = [item for item in required if item != "figure_id"]
        for value in node.values():
            _remove_figure_id(value)
    elif isinstance(node, list):
        for value in node:
            _remove_figure_id(value)


def bouncer_schema() -> dict:
    schema = copy.deepcopy(payload_json_schema())
    _remove_figure_id(schema)
    return schema


def export_schema(path: Path = SCHEMA_PATH) -> None:
    """Atomically export the full documentation schema, including figure_id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload_json_schema(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    export_schema()
