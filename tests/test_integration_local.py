from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pipeline.config import load_settings
from pipeline.local_llm.model_manager import ModelManager
from pipeline.local_llm.ollama_client import OllamaClient
from pipeline.orchestrator import run_pipeline

DATA = Path(__file__).parent / "data"


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("OLLAMA_ITESTS") != "1", reason="requires local Ollama"
)
def test_real_model_swap_never_leaves_both_large_models_loaded() -> None:
    settings = load_settings()
    client = OllamaClient(settings)
    client.health_check()
    manager = ModelManager(client, settings)
    try:
        manager.swap_to(settings.ollama.ocr_model)
        loaded = client.loaded_models()
        assert not any(
            name in {
                settings.ollama.extract_model,
                f"{settings.ollama.extract_model}:latest",
            }
            for name in loaded
        )
        manager.swap_to(settings.ollama.extract_model)
        loaded = client.loaded_models()
        assert not any(
            name in {
                settings.ollama.ocr_model,
                f"{settings.ollama.ocr_model}:latest",
            }
            for name in loaded
        )
        assert any(
            name in {
                settings.ollama.extract_model,
                f"{settings.ollama.extract_model}:latest",
            }
            for name in loaded
        )
    finally:
        manager.release_all()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("OLLAMA_ITESTS") != "1", reason="requires local Ollama"
)
def test_full_local_run_all_fixture_types() -> None:
    settings = load_settings()
    inputs = [
        DATA / "native_report.pdf",
        DATA / "scanned_report.pdf",
        DATA / "financials.xlsx",
        DATA / "financials.csv",
    ]
    results = run_pipeline(inputs, settings, no_cloud=True)
    assert all(result.status == "success" for result in results)
    assert all(result.artifacts["payload_json"].is_file() for result in results)
    payloads = [
        json.loads(result.artifacts["payload_json"].read_text(encoding="utf-8"))
        for result in results
    ]
    assert all(payload["figures"] for payload in payloads)
    assert all(
        figure["figure_id"] == f"F{index:04d}"
        for payload in payloads
        for index, figure in enumerate(payload["figures"], start=1)
    )
    assert any(
        figure["label"].lower() == "total assets"
        for figure in payloads[0]["figures"]
    )
