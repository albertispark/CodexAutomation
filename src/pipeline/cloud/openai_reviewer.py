"""OpenAI peer reviewer: the only module allowed to import the OpenAI SDK."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import openai
from pydantic import BaseModel, ConfigDict, Field

from pipeline.cloud.analysis import outbound_payload_dict
from pipeline.cloud.claude_client import AnalysisResult, SYSTEM_PROMPT
from pipeline.config import Settings
from pipeline.extraction.redactor import RedactedPayload

REVIEW_PROMPT_VERSION = "1"


class ReviewIssue(BaseModel):
    """One concrete concern found during independent review."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["warning", "error"]
    category: Literal[
        "arithmetic",
        "citation",
        "formula",
        "period",
        "unit",
        "scope",
        "adjustment",
        "data_quality",
        "other",
    ]
    description: str
    related_items: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)


class PeerReviewResult(BaseModel):
    """Structured decision returned by the independent reviewer."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["approved", "corrected", "rejected"]
    issues: list[ReviewIssue]
    reviewed_analysis: AnalysisResult


class ReviewUsage(BaseModel):
    """Provider-neutral counters for one OpenAI Responses call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class PeerReviewRefusalError(RuntimeError):
    """OpenAI declined to perform the requested peer review."""


REVIEW_SYSTEM_PROMPT = f"""\
You are the independent second reviewer for a financial-analysis pipeline.
Claude produced the candidate analysis, but you must not defer to it. Recompute
and verify the work directly from the supplied redacted payload.

Review requirements (hard constraints):
1. Check every formula, arithmetic result, cited figure_id, reporting period,
   duration, currency, unit scale, entity scope, sign convention, variance, and
   normalization adjustment against the payload and the canonical contract.
2. Use only the supplied payload. Never invent a number, fact, cause, figure_id,
   or missing relationship, and never use outside knowledge.
3. Return "approved" only when the candidate can be used unchanged and contains
   no material error. In that case reviewed_analysis must exactly equal the
   candidate analysis.
4. Return "corrected" when every material issue can be repaired from payload
   evidence. Put the complete corrected answer in reviewed_analysis and record
   each correction as an issue.
5. Return "rejected" when a reliable final answer cannot be produced from the
   payload. Include at least one error issue. reviewed_analysis may retain the
   candidate only for audit purposes; rejected work is never published.
6. For issue.figure_ids, list only relevant payload IDs. If the issue itself is
   an invented or nonexistent citation, leave figure_ids empty and name that
   bad citation in the description.
7. Conform to the response schema exactly. Keep issue descriptions concise.

The canonical production-analysis contract follows. Apply it as the review
standard, not as evidence that Claude's answer is correct:

{SYSTEM_PROMPT}
"""


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def build_review_user_message(
    redacted: RedactedPayload,
    tasks: list[str],
    claude_analysis: AnalysisResult,
) -> str:
    """Build the exact reviewer input from redacted, structured data only."""
    if not isinstance(redacted, RedactedPayload):
        raise TypeError("build_review_user_message requires RedactedPayload")
    if not isinstance(claude_analysis, AnalysisResult):
        raise TypeError("build_review_user_message requires AnalysisResult")
    return json.dumps(
        {
            "requested_tasks": tasks,
            "redacted_payload": outbound_payload_dict(redacted),
            "claude_analysis": claude_analysis.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _refusal_text(response: Any) -> str | None:
    for output in _get(response, "output", []) or []:
        for content in _get(output, "content", []) or []:
            if _get(content, "type") == "refusal":
                return str(_get(content, "refusal", "OpenAI refused the review"))
    return None


def _normalize_usage(response: Any) -> ReviewUsage:
    usage = _get(response, "usage", {}) or {}
    input_details = _get(usage, "input_tokens_details", {}) or {}
    return ReviewUsage(
        input_tokens=int(_get(usage, "input_tokens", 0) or 0),
        output_tokens=int(_get(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(_get(input_details, "cached_tokens", 0) or 0),
    )


def validate_peer_review(
    review: PeerReviewResult,
    original: AnalysisResult,
    redacted: RedactedPayload,
) -> None:
    """Enforce invariants before a reviewed result can reach a workbook."""
    original_data = original.model_dump(mode="json")
    reviewed_data = review.reviewed_analysis.model_dump(mode="json")
    if review.verdict == "approved" and reviewed_data != original_data:
        raise ValueError("approved review changed the Claude analysis")
    if review.verdict == "corrected" and reviewed_data == original_data:
        raise ValueError("corrected review did not change the Claude analysis")
    if review.verdict == "rejected":
        if not any(issue.severity == "error" for issue in review.issues):
            raise ValueError("rejected review must include at least one error issue")
        return

    valid_ids = {figure.figure_id for figure in redacted.payload.figures}
    cited_ids = {
        figure_id
        for metric in review.reviewed_analysis.computed_metrics
        for figure_id in metric.inputs
    }
    cited_ids.update(
        figure_id
        for adjustment in review.reviewed_analysis.adjustments
        for figure_id in adjustment.inputs
    )
    unknown = sorted(cited_ids - valid_ids)
    if unknown:
        raise ValueError(
            "reviewed analysis cites figure IDs absent from the redacted payload: "
            + ", ".join(unknown)
        )


class OpenAIReviewer:
    """Structured OpenAI Responses client for independent financial review."""

    def __init__(self, cfg: Settings) -> None:
        self._client = openai.OpenAI(
            api_key=(
                cfg.review.api_key.get_secret_value() if cfg.review.api_key else None
            ),
            base_url=cfg.review.base_url,
        ).with_options(timeout=180.0, max_retries=3)
        self._model = cfg.review.model
        self._reasoning_effort = cfg.review.reasoning_effort
        self._max_output_tokens = cfg.review.max_output_tokens

    def review(
        self,
        redacted: RedactedPayload,
        tasks: list[str],
        claude_analysis: AnalysisResult,
    ) -> tuple[PeerReviewResult, ReviewUsage]:
        if not isinstance(redacted, RedactedPayload):
            raise TypeError("OpenAIReviewer.review requires RedactedPayload")
        if not isinstance(claude_analysis, AnalysisResult):
            raise TypeError("OpenAIReviewer.review requires AnalysisResult")

        response = self._client.responses.parse(
            model=self._model,
            input=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_review_user_message(
                        redacted, tasks, claude_analysis
                    ),
                },
            ],
            text_format=PeerReviewResult,
            reasoning={"effort": self._reasoning_effort},
            max_output_tokens=self._max_output_tokens,
            store=False,
        )
        usage = _normalize_usage(response)
        refusal = _refusal_text(response)
        if refusal:
            raise PeerReviewRefusalError(refusal)
        status = _get(response, "status")
        if status != "completed":
            detail = _get(response, "incomplete_details", status or "unknown")
            raise RuntimeError(f"OpenAI peer review did not complete: {detail}")
        parsed = _get(response, "output_parsed")
        if not isinstance(parsed, PeerReviewResult):
            parsed = PeerReviewResult.model_validate(parsed)
        validate_peer_review(parsed, claude_analysis, redacted)
        return parsed, usage
