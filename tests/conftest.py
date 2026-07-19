from __future__ import annotations

from pathlib import Path
import json
import runpy
from types import SimpleNamespace

import pytest
import yaml

from pipeline.config import load_settings


FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parent / "data"

# Binary/table fixtures are deterministic build artifacts, not repository
# blobs. Generate them before collection so a clean checkout is self-contained.
runpy.run_path(str(DATA / "make_fixtures.py"))["main"]()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PIPELINE_CONFIG", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {
        "paths": {"inputs": "inputs", "outputs": "outputs", "cache": "cache", "logs": "logs"},
        "review": {"enabled": False},
        "redaction": {
            "enabled": True,
            "patterns": {
                "us_phone": r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"
            },
        },
    }
    config_path = config_dir / "settings.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_settings(config_path)


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch):
    """Canonical SDK fake backed by tests/fixtures canned local responses."""
    valid = json.loads((FIXTURES / "ollama_bouncer_valid.json").read_text())

    class FakeOllamaSDK:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []
            self.resident: list[str] = []

        def list(self):
            return {
                "models": [
                    {"model": "gemma4:e4b"},
                    {"model": "qwen3:8b"},
                    {"model": "nomic-embed-text"},
                ]
            }

        def ps(self):
            return {"models": [{"model": name} for name in self.resident]}

        def chat(self, **kwargs):
            self.calls.append(("chat", kwargs))
            model = kwargs["model"]
            if kwargs.get("messages") == []:
                if kwargs.get("keep_alive") == 0:
                    self.resident = [name for name in self.resident if name != model]
                elif model not in self.resident:
                    self.resident.append(model)
                return {"message": {"content": ""}}
            return {"message": {"content": json.dumps(valid)}}

        def embed(self, **kwargs):
            self.calls.append(("embed", kwargs))
            return {"embeddings": [[1.0] + [0.0] * 767 for _ in kwargs["input"]]}

    sdk = FakeOllamaSDK()
    monkeypatch.setattr("pipeline.local_llm.ollama_client.Client", lambda **_: sdk)
    return sdk


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch):
    """Canonical Anthropic fake; it never opens a network connection."""
    valid = json.loads((FIXTURES / "anthropic_analysis_valid.json").read_text())

    class Messages:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []
            self.batches = SimpleNamespace(create=self.create_batch)

        def count_tokens(self, **kwargs):
            self.calls.append(("count_tokens", kwargs))
            return SimpleNamespace(input_tokens=5000)

        def parse(self, **kwargs):
            self.calls.append(("parse", kwargs))
            return SimpleNamespace(
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
                parsed_output=valid,
            )

        def create_batch(self, **kwargs):
            self.calls.append(("batch", kwargs))
            return SimpleNamespace(id="msgbatch_fixture")

    class FakeAnthropicSDK:
        def __init__(self, **kwargs) -> None:
            self.init_kwargs = kwargs
            self.messages = Messages()
            self.options: dict | None = None

        def with_options(self, **kwargs):
            self.options = kwargs
            return self

    instances: list[FakeAnthropicSDK] = []

    def factory(**kwargs):
        instance = FakeAnthropicSDK(**kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr("pipeline.cloud.claude_client.anthropic.Anthropic", factory)
    return instances
