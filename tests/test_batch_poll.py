from __future__ import annotations

import json
from pathlib import Path

from pydantic import SecretStr
from rich.console import Console

from pipeline.cloud.claude_client import AnalysisResult, BudgetLedger
from pipeline.extraction.bouncer import Bouncer
from pipeline.extraction.redactor import Redactor
from pipeline.indexing.embedder import Embedder
from pipeline.indexing.vector_store import NumpyVectorStore
from pipeline.ingestion.router import detect_kind
from pipeline.orchestrator import _payload_sha256, poll_batch
from tests.test_orchestrator import NoCallOllama, _analysis, _payload


def test_batch_poll_succeeds_records_half_rate_and_settles(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "batch.csv"
    source.write_text("Metric,FY2025\nRevenue,1250\n", encoding="utf-8")
    plan = detect_kind(source, settings)
    helper = Bouncer(
        NoCallOllama(settings), Embedder(NoCallOllama(settings), settings),
        NumpyVectorStore(), settings,
    )
    cache = {
        **helper._determinants("csv"),
        "retrieved_chunk_ids": [f"{plan.file_sha256}:000001"],
        "context_token_estimate": 50,
        "payload": _payload().model_dump(mode="json"),
    }
    (settings.paths.bouncer_cache / f"{plan.file_sha256}.payload.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )
    payload_sha, request_sha = _payload_sha256(
        Redactor(settings).redact_payload(_payload()), settings, ["metrics"]
    )
    custom_id = f"{plan.file_sha256[:12]}-{payload_sha[:12]}"
    manifest_path = settings.paths.batches_dir / "msgbatch_poll.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": "msgbatch_poll",
                "model": settings.cloud.model,
                "prompt_version": "1",
                "reservation_usd": 0.5,
                "settled": False,
                "jobs": {
                    custom_id: {
                        "input_path": str(source),
                        "payload_sha": payload_sha,
                        "request_sha": request_sha,
                        "doc_sha": plan.file_sha256,
                        "tasks": ["metrics"],
                        "payload_tokens": 100,
                        "raw_tokens_est": 1000,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    settings.cloud.api_key = SecretStr("test")

    class BatchClient:
        def __init__(self, cfg):
            self.ledger = BudgetLedger(cfg.paths.logs / "spend.jsonl", 25)
        def retrieve_batch(self, batch_id):
            return {"processing_status": "ended"}
        def batch_results(self, batch_id):
            usage = {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
            message = {
                "stop_reason": "end_turn",
                "usage": usage,
                "content": [{"type": "text", "text": _analysis().model_dump_json()}],
            }
            return [{"custom_id": custom_id, "result": {"type": "succeeded", "message": message}}]

    monkeypatch.setattr("pipeline.orchestrator.ClaudeClient", BatchClient)
    results = poll_batch(
        "msgbatch_poll", settings, console=Console(file=None, quiet=True), poll_interval_s=0
    )
    assert results[0].status == "success"
    assert results[0].artifacts["xlsx"].is_file()
    lines = [json.loads(line) for line in (settings.paths.logs / "spend.jsonl").read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["call", "batch_settlement"]
    assert lines[0]["batch"] is True
    assert lines[1]["cost_usd"] == -0.5
    assert json.loads(manifest_path.read_text())["settled"] is True


def test_malformed_batch_result_does_not_abort_neighbor(
    settings, tmp_path: Path, monkeypatch
) -> None:
    settings.cloud.api_key = SecretStr("test")
    helper = Bouncer(
        NoCallOllama(settings),
        Embedder(NoCallOllama(settings), settings),
        NumpyVectorStore(),
        settings,
    )
    jobs: dict[str, dict] = {}
    custom_ids: list[str] = []
    for index, name in enumerate(("bad.csv", "good.csv"), start=1):
        source = tmp_path / name
        source.write_text(
            f"Metric,FY2025\nRevenue,{1250 + index}\n", encoding="utf-8"
        )
        plan = detect_kind(source, settings)
        cache = {
            **helper._determinants("csv"),
            "retrieved_chunk_ids": [f"{plan.file_sha256}:000001"],
            "context_token_estimate": 50,
            "payload": _payload().model_dump(mode="json"),
        }
        (settings.paths.bouncer_cache / f"{plan.file_sha256}.payload.json").write_text(
            json.dumps(cache), encoding="utf-8"
        )
        payload_sha, request_sha = _payload_sha256(
            Redactor(settings).redact_payload(_payload()), settings, ["metrics"]
        )
        custom_id = f"{plan.file_sha256[:12]}-{payload_sha[:12]}"
        custom_ids.append(custom_id)
        jobs[custom_id] = {
            "input_path": str(source),
            "payload_sha": payload_sha,
            "request_sha": request_sha,
            "doc_sha": plan.file_sha256,
            "tasks": ["metrics"],
            "payload_tokens": 100,
            "raw_tokens_est": 1000,
        }
    manifest_path = settings.paths.batches_dir / "msgbatch_mixed.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": "msgbatch_mixed",
                "reservation_usd": 0.2,
                "settled": False,
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )

    class BatchClient:
        def __init__(self, cfg):
            self.ledger = BudgetLedger(cfg.paths.logs / "spend.jsonl", 25)

        def retrieve_batch(self, batch_id):
            return {"processing_status": "ended"}

        def batch_results(self, batch_id):
            usage = {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
            good_message = {
                "stop_reason": "end_turn",
                "usage": usage,
                "content": [
                    {"type": "text", "text": _analysis().model_dump_json()}
                ],
            }
            return [
                {
                    "custom_id": custom_ids[0],
                    "result": {"type": "succeeded"},
                },
                {
                    "custom_id": custom_ids[1],
                    "result": {"type": "succeeded", "message": good_message},
                },
            ]

    monkeypatch.setattr("pipeline.orchestrator.ClaudeClient", BatchClient)
    results = poll_batch(
        "msgbatch_mixed",
        settings,
        console=Console(file=None, quiet=True),
        poll_interval_s=0,
    )
    assert [result.status for result in results] == ["failed", "success"]
    assert results[1].artifacts["xlsx"].is_file()
    ledger = [
        json.loads(line)
        for line in (settings.paths.logs / "spend.jsonl").read_text().splitlines()
    ]
    assert [line["kind"] for line in ledger] == ["call", "batch_settlement"]
