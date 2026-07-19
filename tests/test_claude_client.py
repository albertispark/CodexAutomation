from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.cloud.claude_client import (
    PRICE_PER_MTOK,
    SYSTEM_PROMPT,
    AnalysisResult,
    BudgetExceededError,
    BudgetLedger,
    ClaudeClient,
    CloudRefusalError,
    ComputedMetric,
)
from pipeline.config import load_settings
from pipeline.extraction.redactor import Redactor
from pipeline.extraction.schemas import ExtractionPayload, FinancialFigure, StatementType


def _usage(input_tokens=100, output_tokens=50, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        computed_metrics=[
            ComputedMetric(
                name="Operating Margin",
                value=0.2,
                formula_used="operating_income / revenue",
                inputs=["F0001", "F0002"],
                period="FY2025",
            )
        ],
        variance_analysis=[],
        adjustments=[],
        data_quality_flags=[],
    )


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
                value=1000,
                unit="thousands",
                currency="USD",
                period="FY2025",
                statement=StatementType.income_statement,
                source_page=1,
                verbatim_context="Revenue 1,000",
            )
        ],
        warnings=[],
    )
    return Redactor(settings).redact_payload(payload)


class FakeBatches:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="msgbatch_test")


class FakeMessages:
    def __init__(self) -> None:
        self.count_calls: list[dict] = []
        self.parse_calls: list[dict] = []
        self.responses: list[object] = []
        self.batches = FakeBatches()

    def count_tokens(self, **kwargs):
        self.count_calls.append(kwargs)
        return SimpleNamespace(input_tokens=5000)

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return self.responses.pop(0)


class FakeAnthropic:
    def __init__(self, *, api_key, base_url) -> None:
        self.init_args = {"api_key": api_key, "base_url": base_url}
        self.options: dict | None = None
        self.messages = FakeMessages()

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


def _client(settings, monkeypatch):
    instances: list[FakeAnthropic] = []

    def factory(**kwargs):
        instance = FakeAnthropic(**kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr("pipeline.cloud.claude_client.anthropic.Anthropic", factory)
    client = ClaudeClient(settings)
    return client, instances[0]


def _response(reason="end_turn", usage=None, parsed=None):
    return SimpleNamespace(
        stop_reason=reason,
        usage=usage or _usage(),
        parsed_output=parsed if parsed is not None else _analysis(),
    )


def test_prompt_is_cacheable_offline_and_schema_batch_safe() -> None:
    assert len(SYSTEM_PROMPT) // 4 >= 4300
    assert "{GLOSSARY}" not in SYSTEM_PROMPT
    schema = json.dumps(AnalysisResult.model_json_schema())
    assert "maxLength" not in schema
    assert '"additionalProperties": false' in schema


def test_analyze_uses_exact_structured_surface(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.messages.responses = [_response()]
    result, usage = client.analyze(_redacted(settings), ["metrics"], file_sha12="a" * 12)
    assert result == _analysis()
    assert usage.output_tokens == 50
    assert sdk.options == {"timeout": 180.0, "max_retries": 3}
    assert sdk.messages.count_calls[0]["system"][0]["text"] == SYSTEM_PROMPT
    assert sdk.messages.count_calls[0]["system"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    call = sdk.messages.parse_calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_format"] is AnalysisResult
    assert not {"temperature", "top_p", "top_k", "budget_tokens"} & call.keys()
    ledger_line = json.loads((settings.paths.logs / "spend.jsonl").read_text())
    assert ledger_line["kind"] == "call"


def test_budget_guard_precedes_parse(settings, monkeypatch) -> None:
    settings.cloud.monthly_budget_usd = 0.01
    client, sdk = _client(settings, monkeypatch)
    with pytest.raises(BudgetExceededError):
        client.analyze(_redacted(settings), ["metrics"], file_sha12="b" * 12)
    assert sdk.messages.parse_calls == []


def test_max_tokens_retries_once_and_records_both(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.messages.responses = [
        _response("max_tokens", _usage(100, 8000), parsed=None),
        _response("end_turn", _usage(100, 100), _analysis()),
    ]
    result, _ = client.analyze(_redacted(settings), ["metrics"], file_sha12="c" * 12)
    assert result == _analysis()
    assert [call["max_tokens"] for call in sdk.messages.parse_calls] == [8000, 16000]
    lines = (settings.paths.logs / "spend.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert client.last_cost_usd > 0


def test_refusal_is_billed_then_raised(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.messages.responses = [_response("refusal", _usage(100, 10), None)]
    with pytest.raises(CloudRefusalError) as caught:
        client.analyze(_redacted(settings), ["metrics"], file_sha12="d" * 12)
    assert caught.value.cost_usd > 0
    assert (settings.paths.logs / "spend.jsonl").read_text().count("\n") == 1


def test_max_tokens_at_hard_ceiling_does_not_retry(settings, monkeypatch) -> None:
    settings.cloud.max_tokens = 16000
    client, sdk = _client(settings, monkeypatch)
    sdk.messages.responses = [
        _response("max_tokens", _usage(100, 16000), parsed=None)
    ]
    with pytest.raises(RuntimeError) as caught:
        client.analyze(_redacted(settings), ["metrics"], file_sha12="1" * 12)
    assert getattr(caught.value, "cost_usd") > 0
    assert len(sdk.messages.parse_calls) == 1
    assert (settings.paths.logs / "spend.jsonl").read_text().count("\n") == 1


def test_unknown_stop_reason_is_billed_and_carries_cost(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.messages.responses = [_response("pause_turn", _usage(100, 10), None)]
    with pytest.raises(RuntimeError, match="unexpected stop_reason") as caught:
        client.analyze(_redacted(settings), ["metrics"], file_sha12="2" * 12)
    assert getattr(caught.value, "cost_usd") > 0
    assert (settings.paths.logs / "spend.jsonl").read_text().count("\n") == 1


def test_batch_uses_json_schema_and_reserves_liability(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    batch_id = client.analyze_batch(
        [("a" * 12 + "-" + "b" * 12, _redacted(settings), ["metrics"])]
    )
    assert batch_id == "msgbatch_test"
    request = sdk.messages.batches.calls[0]["requests"][0]
    assert request["custom_id"] == "a" * 12 + "-" + "b" * 12
    fmt = request["params"]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == AnalysisResult.model_json_schema()
    line = json.loads((settings.paths.logs / "spend.jsonl").read_text())
    assert line["kind"] == "batch_reservation"
    assert line["cost_usd"] == client.last_batch_reservation_usd


def test_ledger_prices_four_counters_and_skips_torn_line(tmp_path: Path, caplog) -> None:
    ledger = BudgetLedger(tmp_path / "spend.jsonl", 100)
    cost = ledger.record(
        _usage(1_000_000, 1_000_000, 1_000_000, 1_000_000),
        file_sha12="e" * 12,
        payload_sha12="f" * 12,
    )
    assert cost == sum(PRICE_PER_MTOK.values())
    with (tmp_path / "spend.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("torn{")
    assert ledger.current_month_spend_usd() == cost
    assert "line 2" in caplog.text


@pytest.mark.anthropic_integration
@pytest.mark.skipif(
    os.environ.get("ANTHROPIC_ITESTS") != "1",
    reason="requires an explicit Anthropic count_tokens integration run",
)
def test_live_system_prompt_meets_cache_floor() -> None:
    settings = load_settings()
    if settings.cloud.api_key is None:
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    client = ClaudeClient(settings)
    response = client._client.messages.count_tokens(
        model=settings.cloud.model,
        system=client._system_blocks(),
        messages=[{"role": "user", "content": "x"}],
    )
    assert response.input_tokens >= 4096
