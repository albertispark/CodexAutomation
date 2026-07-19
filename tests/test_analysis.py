from __future__ import annotations

import json

import pytest

from pipeline.cloud.analysis import build_user_message
from pipeline.extraction.redactor import Redactor
from pipeline.extraction.schemas import ExtractionPayload, FinancialFigure, StatementType


def _redacted(settings):
    payload = ExtractionPayload(
        company="Acme analyst@example.com",
        doc_type="annual report",
        currency_default="USD",
        periods=["FY2025"],
        figures=[
            FinancialFigure(
                figure_id="F0001",
                label="Revenue",
                value=1250,
                unit="thousands",
                currency="USD",
                period="FY2025",
                statement=StatementType.income_statement,
                source_page=1,
                verbatim_context="PRIVATE SOURCE SNIPPET Revenue 1,250",
            )
        ],
        warnings=[],
    )
    return Redactor(settings).redact_payload(payload)


def test_message_is_stable_redacted_and_omits_verbatim(settings) -> None:
    redacted = _redacted(settings)
    first = build_user_message(redacted, ["metrics", "variance"])
    second = build_user_message(redacted, ["metrics", "variance"])
    assert first == second
    assert "PRIVATE SOURCE SNIPPET" not in first
    assert "verbatim_context" not in first
    assert "analyst@example.com" not in first
    assert "[REDACTED:email]" in first
    assert "F0001" in first
    payload = json.loads(first.split("PAYLOAD:\n", 1)[1])
    assert payload["figures"][0]["figure_id"] == "F0001"


def test_message_rejects_unknown_task_and_raw_payload(settings) -> None:
    redacted = _redacted(settings)
    with pytest.raises(ValueError, match="valid tasks"):
        build_user_message(redacted, ["invent"])
    with pytest.raises(TypeError):
        build_user_message(redacted.payload, ["metrics"])  # type: ignore[arg-type]
