from __future__ import annotations

from pipeline.extraction.redactor import Redactor
from pipeline.extraction.schemas import ExtractionPayload, FinancialFigure, StatementType


def _payload() -> ExtractionPayload:
    return ExtractionPayload(
        company="Acme contact analyst@example.com or 212-555-0199",
        doc_type="account 987654321",
        currency_default="USD",
        periods=["FY2025 SSN 123-45-6789", "EIN 12-3456789"],
        figures=[
            FinancialFigure(
                figure_id="F0001",
                label="Card 4111 1111 1111 1111",
                value=123456789.25,
                unit="absolute",
                currency="USD",
                period="FY2025",
                statement=StatementType.balance_sheet,
                source_page=1,
                verbatim_context="IBAN GB82 WEST 1234 5698 7654 32",
            )
        ],
        warnings=["compact IBAN GB82WEST12345698765432"],
    )


def test_all_builtins_and_config_pattern_are_redacted(settings) -> None:
    original = _payload()
    redacted = Redactor(settings).redact_payload(original)
    encoded = redacted.payload.model_dump_json()
    for secret in (
        "analyst@example.com", "212-555-0199", "987654321", "123-45-6789",
        "12-3456789", "4111 1111 1111 1111", "GB82 WEST 1234 5698 7654 32",
        "GB82WEST12345698765432",
    ):
        assert secret not in encoded
    assert redacted.payload.figures[0].value == original.figures[0].value
    assert redacted.payload.figures[0].figure_id == "F0001"
    names = {event.pattern_name for event in redacted.events}
    assert {"email", "us_phone", "account_number", "ssn", "ein", "card_number", "iban"} <= names
    assert all(event.match_preview.endswith("...") for event in redacted.events)


def test_override_and_disable(settings) -> None:
    settings.redaction.patterns = {"ssn": "", "email": r"Acme"}
    redacted = Redactor(settings).redact_payload(_payload())
    assert "123-45-6789" in redacted.payload.periods[0]
    assert redacted.payload.company.startswith("[REDACTED:email]")


def test_disabled_is_deep_passthrough(settings) -> None:
    settings.redaction.enabled = False
    payload = _payload()
    result = Redactor(settings).redact_payload(payload)
    assert result.events == []
    assert result.payload == payload
    assert result.payload is not payload
