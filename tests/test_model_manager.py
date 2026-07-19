from __future__ import annotations

import pytest

from pipeline.local_llm.model_manager import ModelManager


class ResidencyClient:
    def __init__(self, loaded: list[str]) -> None:
        self.loaded = list(loaded)
        self.calls: list[tuple[str, str | int]] = []

    def loaded_models(self) -> list[str]:
        return list(self.loaded)

    def load_ping(self, model: str, keep_alive: str | int) -> None:
        self.calls.append((model, keep_alive))
        aliases = {model, f"{model}:latest"}
        if keep_alive == 0:
            self.loaded = [item for item in self.loaded if item not in aliases]
        elif model not in self.loaded:
            self.loaded.append(model)


def test_swap_evicts_other_before_loading_target(settings) -> None:
    client = ResidencyClient(["gemma4:e4b:latest"])
    manager = ModelManager(client, settings)
    manager.swap_to("qwen3:8b")
    assert client.calls == [("gemma4:e4b", 0), ("qwen3:8b", "5m")]
    assert client.loaded == ["qwen3:8b"]


def test_swap_noop_when_already_sole_resident(settings) -> None:
    client = ResidencyClient(["qwen3:8b:latest", "nomic-embed-text:latest"])
    ModelManager(client, settings).swap_to("qwen3:8b")
    assert client.calls == []


def test_swap_rejects_embedder(settings) -> None:
    client = ResidencyClient([])
    with pytest.raises(ValueError, match="large models"):
        ModelManager(client, settings).swap_to("nomic-embed-text")


def test_evict_and_release_normalize_latest(settings) -> None:
    client = ResidencyClient(["gemma4:e4b:latest", "qwen3:8b:latest"])
    manager = ModelManager(client, settings)
    manager.evict_large_models()
    assert client.loaded == []
    client.loaded = ["gemma4:e4b"]
    manager.release_all()
    assert client.loaded == []
