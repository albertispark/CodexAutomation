from __future__ import annotations

import errno
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import SecretStr
from rich.console import Console

from pipeline.cloud.claude_client import (
    AnalysisResult,
    BudgetExceededError,
    CloudRefusalError,
    ComputedMetric,
)
from pipeline.extraction.bouncer import Bouncer
from pipeline.extraction.schemas import ExtractionPayload, FinancialFigure, StatementType
from pipeline.indexing.embedder import EMBED_DIM, Embedder
from pipeline.indexing.vector_store import NumpyVectorStore
from pipeline.ingestion.router import detect_kind
from pipeline.orchestrator import (
    PipelineEnvironmentError,
    _embed_query_with_memory_retry,
    pipeline_run_lock,
    run_pipeline,
)

DATA = Path(__file__).parent / "data"


def _payload() -> ExtractionPayload:
    return ExtractionPayload(
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
                verbatim_context="Revenue 1,250",
            )
        ],
        warnings=[],
    )


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        computed_metrics=[
            ComputedMetric(
                name="Revenue Growth",
                value=0.1,
                formula_used="(revenue_current - revenue_prior) / revenue_prior",
                inputs=["F0001"],
                period="FY2025",
            )
        ],
        variance_analysis=[],
        adjustments=[],
        data_quality_flags=[],
    )


class NoCallOllama:
    calls: list[str] = []

    def __init__(self, settings) -> None:
        self.settings = settings

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            type(self).calls.append(name)
            raise AssertionError(f"unexpected Ollama call: {name}")
        return fail


def _write_bouncer_cache(path: Path, settings) -> str:
    plan = detect_kind(path, settings)
    helper = Bouncer(
        NoCallOllama(settings),
        Embedder(NoCallOllama(settings), settings),
        NumpyVectorStore(),
        settings,
    )
    hint = path.suffix.lower().lstrip(".")
    data = {
        **helper._determinants(hint),
        "retrieved_chunk_ids": [f"{plan.file_sha256}:000001"],
        "context_token_estimate": 100,
        "payload": _payload().model_dump(mode="json"),
    }
    target = settings.paths.bouncer_cache / f"{plan.file_sha256}.payload.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return plan.file_sha256


def _copy_csv(tmp_path: Path, name: str, suffix: str = "") -> Path:
    path = tmp_path / name
    path.write_text(
        "Metric,FY2024,FY2025\nRevenue,1000,1250\nNet income,100,125\n" + suffix,
        encoding="utf-8",
    )
    return path


def test_fully_cached_no_cloud_run_makes_zero_ollama_calls(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = _copy_csv(tmp_path, "cached.csv")
    _write_bouncer_cache(source, settings)
    NoCallOllama.calls = []
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)
    results = run_pipeline([source], settings, no_cloud=True, console=Console(file=None, quiet=True))
    assert results[0].status == "success"
    assert results[0].artifacts["payload_json"].is_file()
    assert NoCallOllama.calls == []
    outbound = json.loads(results[0].artifacts["payload_json"].read_text())
    assert "verbatim_context" not in outbound["figures"][0]


def test_dry_run_makes_no_cloud_client_or_count_tokens(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = _copy_csv(tmp_path, "dry.csv")
    _write_bouncer_cache(source, settings)
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)
    monkeypatch.setattr(
        "pipeline.orchestrator.ClaudeClient",
        lambda *_: (_ for _ in ()).throw(AssertionError("cloud client constructed")),
    )
    results = run_pipeline([source], settings, dry_run=True, console=Console(file=None, quiet=True))
    assert results[0].status == "success"
    assert results[0].payload_tokens == 0


def test_stage_major_ocr_precedes_every_bouncer_call(
    settings, tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    first = tmp_path / "scan1.pdf"
    second = tmp_path / "scan2.pdf"
    raw = (DATA / "scanned_report.pdf").read_bytes()
    first.write_bytes(raw + b"\n% first")
    second.write_bytes(raw + b"\n% second")

    class FakeOllama:
        def __init__(self, cfg): pass
        def health_check(self): events.append("health")
        def ocr_image(self, image, prompt):
            events.append("ocr")
            return "Revenue for fiscal year 2025 was 1,250 dollars"
        def warm_embed(self, keep_alive=None): events.append("warm_embed")
        def embed(self, texts):
            events.append("embed_query" if texts and texts[0].startswith("search_query:") else "embed_chunks")
            vectors = []
            for _ in texts:
                vector = [0.0] * EMBED_DIM
                vector[0] = 1.0
                vectors.append(vector)
            return vectors
        def chat_json(self, **kwargs):
            events.append("bouncer")
            raw_payload = _payload().model_dump(mode="json")
            for figure in raw_payload["figures"]:
                figure.pop("figure_id", None)
            return raw_payload

    class FakeManager:
        def __init__(self, client, cfg): pass
        def swap_to(self, model): events.append(f"swap:{model}")
        def evict_large_models(self): events.append("evict_large")
        def release_all(self): events.append("release_all")

    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", FakeOllama)
    monkeypatch.setattr("pipeline.orchestrator.ModelManager", FakeManager)
    results = run_pipeline([first, second], settings, no_cloud=True, console=Console(file=None, quiet=True))
    assert all(result.status == "success" for result in results)
    assert events.count("ocr") == 2
    assert events.count("bouncer") == 2
    assert max(i for i, event in enumerate(events) if event == "ocr") < min(
        i for i, event in enumerate(events) if event == "bouncer"
    )
    assert events.index("evict_large") < events.index("embed_chunks")
    assert events.index(f"swap:{settings.ollama.extract_model}") < events.index("bouncer")


class FakeUsage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


def test_cloud_cache_makes_immediate_rerun_zero_cloud_calls(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = _copy_csv(tmp_path, "cloud.csv")
    _write_bouncer_cache(source, settings)
    settings.cloud.api_key = SecretStr("test")
    NoCallOllama.calls = []
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)

    class FakeCloud:
        constructions = 0
        calls = 0
        def __init__(self, cfg):
            type(self).constructions += 1
            self.last_payload_tokens = 100
            self.last_cost_usd = 0.001
        def analyze(self, *args, **kwargs):
            type(self).calls += 1
            return _analysis(), FakeUsage()

    monkeypatch.setattr("pipeline.orchestrator.ClaudeClient", FakeCloud)
    first = run_pipeline([source], settings, console=Console(file=None, quiet=True))
    second = run_pipeline([source], settings, console=Console(file=None, quiet=True))
    assert first[0].status == second[0].status == "success"
    assert FakeCloud.calls == 1
    assert FakeCloud.constructions == 1
    assert NoCallOllama.calls == []


def test_budget_exceeded_downgrades_all_remaining_files(
    settings, tmp_path: Path, monkeypatch
) -> None:
    sources = [_copy_csv(tmp_path, "one.csv"), _copy_csv(tmp_path, "two.csv", "x,y,z\n")]
    for source in sources:
        _write_bouncer_cache(source, settings)
    settings.cloud.api_key = SecretStr("test")
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)

    class BudgetCloud:
        def __init__(self, cfg): pass
        def analyze(self, *args, **kwargs):
            raise BudgetExceededError("stop")

    monkeypatch.setattr("pipeline.orchestrator.ClaudeClient", BudgetCloud)
    results = run_pipeline(sources, settings, console=Console(file=None, quiet=True))
    assert [result.status for result in results] == ["budget_stopped", "budget_stopped"]
    assert all(result.artifacts["payload_json"].is_file() for result in results)


def test_cloud_refusal_quarantines_and_records_refused(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = _copy_csv(tmp_path, "refusal.csv")
    _write_bouncer_cache(source, settings)
    settings.cloud.api_key = SecretStr("test")
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)

    class RefusalCloud:
        def __init__(self, cfg): pass
        def analyze(self, *args, **kwargs):
            raise CloudRefusalError("no", cost_usd=0.002)

    monkeypatch.setattr("pipeline.orchestrator.ClaudeClient", RefusalCloud)
    result = run_pipeline([source], settings, console=Console(file=None, quiet=True))[0]
    assert result.status == "refused"
    assert result.artifacts["quarantine_json"].is_file()
    assert result.error == "CloudRefusalError: cloud_refusal"


def test_batch_manifest_is_immediate_and_custom_id_has_no_filename(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = _copy_csv(tmp_path, "secret_filename.csv")
    _write_bouncer_cache(source, settings)
    settings.cloud.api_key = SecretStr("test")
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)

    class BatchCloud:
        jobs = []
        def __init__(self, cfg):
            self.last_batch_payload_tokens = {}
            self.last_batch_reservation_usd = 0.25
        def analyze_batch(self, jobs):
            type(self).jobs = jobs
            self.last_batch_payload_tokens = {custom_id: 123 for custom_id, *_ in jobs}
            return "msgbatch_fixture"

    monkeypatch.setattr("pipeline.orchestrator.ClaudeClient", BatchCloud)
    result = run_pipeline([source], settings, batch=True, console=Console(file=None, quiet=True))[0]
    assert result.status == "batch_pending"
    manifest = json.loads((settings.paths.batches_dir / "msgbatch_fixture.manifest.json").read_text())
    custom_id = next(iter(manifest["jobs"]))
    assert custom_id.count("-") == 1
    assert "secret_filename" not in custom_id
    assert manifest["settled"] is False


def test_encrypted_fixture_quarantines_without_ollama(settings, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)
    result = run_pipeline(
        [DATA / "encrypted.pdf"], settings, no_cloud=True, console=Console(file=None, quiet=True)
    )[0]
    assert result.status == "quarantined"
    artifact = json.loads(result.artifacts["quarantine_json"].read_text())
    assert artifact["stage"] == "ingest"
    assert artifact["reason"] == "encrypted or unreadable PDF"


def test_empty_cached_extraction_quarantines_in_no_cloud_mode(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = _copy_csv(tmp_path, "empty.csv")
    plan = detect_kind(source, settings)
    payload = _payload().model_copy(deep=True)
    payload.figures = []
    helper = Bouncer(
        NoCallOllama(settings),
        Embedder(NoCallOllama(settings), settings),
        NumpyVectorStore(),
        settings,
    )
    cache = {
        **helper._determinants("csv"),
        "retrieved_chunk_ids": [f"{plan.file_sha256}:000001"],
        "context_token_estimate": 10,
        "payload": payload.model_dump(mode="json"),
    }
    (settings.paths.bouncer_cache / f"{plan.file_sha256}.payload.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)
    result = run_pipeline(
        [source],
        settings,
        no_cloud=True,
        console=Console(file=None, quiet=True),
    )[0]
    assert result.status == "quarantined"
    assert result.quarantine_reason == "no_figures_extracted"
    assert "payload_json" not in result.artifacts


def test_disk_full_aborts_as_environment_failure_and_audits(
    settings, tmp_path: Path, monkeypatch
) -> None:
    source = _copy_csv(tmp_path, "disk-full.csv")
    _write_bouncer_cache(source, settings)
    settings.cloud.api_key = SecretStr("test")
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)

    class FakeCloud:
        def __init__(self, cfg):
            self.last_payload_tokens = 100
            self.last_cost_usd = 0.001
            self.last_aggregate_usage = None

        def analyze(self, *args, **kwargs):
            return _analysis(), FakeUsage()

    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr("pipeline.orchestrator.ClaudeClient", FakeCloud)
    monkeypatch.setattr("pipeline.orchestrator.write_workbook", disk_full)
    with pytest.raises(PipelineEnvironmentError, match="disk_full"):
        run_pipeline([source], settings, console=Console(file=None, quiet=True))
    records = [
        json.loads(line)
        for line in (settings.paths.logs / "audit.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["error"] == "EnvironmentFailure: disk_full_payload_preserved"


def test_query_embedding_memory_error_releases_and_retries_once() -> None:
    class ResponseError(RuntimeError):
        pass

    class FlakyEmbedder:
        calls = 0

        def embed_query(self, query):
            self.calls += 1
            if self.calls == 1:
                raise ResponseError("out of memory")
            return np.zeros(EMBED_DIM, dtype=np.float32)

    class FakeOllama:
        def __init__(self):
            self.warms = []

        def warm_embed(self, keep_alive=None):
            self.warms.append(keep_alive)

    class FakeManager:
        releases = 0

        def release_all(self):
            self.releases += 1

    embedder = FlakyEmbedder()
    ollama = FakeOllama()
    manager = FakeManager()
    vector = _embed_query_with_memory_retry(
        embedder, "balance sheet", ollama, manager
    )
    assert vector.shape == (EMBED_DIM,)
    assert embedder.calls == 2
    assert manager.releases == 1
    assert ollama.warms == [0, None]


def test_run_lock_rejects_a_second_writer(settings) -> None:
    with pipeline_run_lock(settings):
        with pytest.raises(
            PipelineEnvironmentError,
            match="another pipeline invocation is running",
        ):
            with pipeline_run_lock(settings):
                raise AssertionError("second lock unexpectedly acquired")
