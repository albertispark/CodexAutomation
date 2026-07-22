from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from pipeline.cloud.claude_client import AnalysisResult, ComputedMetric
from pipeline.cloud.openai_reviewer import (
    OpenAIReviewer,
    PeerReviewResult,
    build_review_user_message,
    validate_peer_review,
)
from pipeline.extraction.redactor import Redactor
from pipeline.extraction.schemas import ExtractionPayload, FinancialFigure, StatementType


def _redacted(settings):
    payload = ExtractionPayload(
        company="Acme",
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
                verbatim_context="SOURCE-SECRET-SNIPPET Revenue 1,250",
            )
        ],
        warnings=[],
    )
    return Redactor(settings).redact_payload(payload)


def _analysis(figure_id: str = "F0001") -> AnalysisResult:
    return AnalysisResult(
        computed_metrics=[
            ComputedMetric(
                name="Reported revenue",
                value=1250,
                formula_used="printed revenue",
                inputs=[figure_id],
                period="FY2025",
            )
        ],
        variance_analysis=[],
        adjustments=[],
        data_quality_flags=[],
    )


def test_reviewer_uses_structured_responses_and_disables_storage(
    settings, monkeypatch
) -> None:
    settings.review.enabled = True
    settings.review.api_key = SecretStr("test-openai")
    analysis = _analysis()
    approved = PeerReviewResult(
        verdict="approved", issues=[], reviewed_analysis=analysis
    )

    class Responses:
        calls: list[dict] = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                status="completed",
                output_parsed=approved,
                output=[],
                usage=SimpleNamespace(
                    input_tokens=80,
                    output_tokens=20,
                    input_tokens_details=SimpleNamespace(cached_tokens=12),
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.responses = Responses()

        def with_options(self, **kwargs):
            self.options = kwargs
            return self

    monkeypatch.setattr(
        "pipeline.cloud.openai_reviewer.openai.OpenAI", FakeOpenAI
    )
    redacted = _redacted(settings)
    result, usage = OpenAIReviewer(settings).review(
        redacted, ["metrics"], analysis
    )

    assert result.verdict == "approved"
    assert usage.input_tokens == 80
    assert usage.cache_read_input_tokens == 12
    call = Responses.calls[0]
    assert call["model"] == settings.review.model
    assert call["text_format"] is PeerReviewResult
    assert call["reasoning"] == {"effort": "medium"}
    assert call["store"] is False
    outbound = call["input"][1]["content"]
    assert "SOURCE-SECRET-SNIPPET" not in outbound
    assert "verbatim_context" not in outbound


def test_review_message_requires_redacted_payload(settings) -> None:
    with pytest.raises(TypeError, match="RedactedPayload"):
        build_review_user_message(object(), ["metrics"], _analysis())  # type: ignore[arg-type]


def test_approved_or_corrected_analysis_cannot_cite_unknown_ids(settings) -> None:
    redacted = _redacted(settings)
    original = _analysis()
    invalid = PeerReviewResult(
        verdict="corrected",
        issues=[],
        reviewed_analysis=_analysis("F9999"),
    )
    with pytest.raises(ValueError, match="F9999"):
        validate_peer_review(invalid, original, redacted)
