from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.local_llm.ollama_client import (
    JSON_RETRY_PROMPT,
    LocalInferenceError,
    LocalJSONError,
    ModelMissingError,
    OllamaClient,
)


class FakeSDK:
    def __init__(self) -> None:
        self.chat_responses: list[dict] = []
        self.chat_calls: list[dict] = []
        self.embed_calls: list[dict] = []

    def list(self):
        return {"models": [
            {"model": "gemma4:e4b"}, {"model": "qwen3:8b"},
            {"model": "nomic-embed-text:latest"},
        ]}

    def ps(self):
        return {"models": []}

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return self.chat_responses.pop(0)

    def embed(self, **kwargs):
        self.embed_calls.append(kwargs)
        return {"embeddings": [[float(i)] * 3 for i, _ in enumerate(kwargs["input"])]}


def _client(settings, monkeypatch):
    sdk = FakeSDK()
    monkeypatch.setattr("pipeline.local_llm.ollama_client.Client", lambda **_: sdk)
    return OllamaClient(settings), sdk


def test_health_check_accepts_latest_alias(settings, monkeypatch) -> None:
    client, _ = _client(settings, monkeypatch)
    client.health_check()


def test_health_check_reports_first_missing(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.list = lambda: {"models": []}
    with pytest.raises(ModelMissingError, match="ollama pull gemma4:e4b"):
        client.health_check()


def test_chat_json_retries_decode_only_without_mutation(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.chat_responses = [
        {"message": {"content": "not json"}},
        {"message": {"content": '{"ok":true}'}},
    ]
    messages = [{"role": "user", "content": "extract"}]
    assert client.chat_json("qwen3:8b", messages, {"type": "object"}, 8192) == {"ok": True}
    assert messages == [{"role": "user", "content": "extract"}]
    assert sdk.chat_calls[1]["messages"][-1]["content"] == JSON_RETRY_PROMPT
    assert sdk.chat_calls[0]["think"] is False


def test_chat_json_second_failure_carries_raw(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.chat_responses = [
        {"message": {"content": "bad one"}},
        {"message": {"content": "bad two"}},
    ]
    with pytest.raises(LocalJSONError) as caught:
        client.chat_json("qwen3:8b", [{"role": "user", "content": "x"}], {}, 8192)
    assert caught.value.raw_text == "bad two"


def test_context_guard_makes_no_call(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    with pytest.raises(LocalInferenceError):
        client.chat_json(
            "qwen3:8b", [{"role": "user", "content": "x" * 20_000}], {}, 4096
        )
    assert sdk.chat_calls == []


def test_json_retry_history_is_context_guarded(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.chat_responses = [{"message": {"content": "x" * 5000}}]
    with pytest.raises(LocalInferenceError):
        client.chat_json(
            "qwen3:8b", [{"role": "user", "content": "extract"}], {}, 4096
        )
    assert len(sdk.chat_calls) == 1


def test_ocr_records_generation_ceiling(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    sdk.chat_responses = [
        {
            "message": {"content": "complete-looking text"},
            "done_reason": "length",
            "prompt_eval_count": 20,
            "eval_count": 20,
        }
    ]
    client.ocr_image(b"png", "transcribe")
    assert client.last_ocr_truncated is True


def test_embed_is_single_batch_layer(settings, monkeypatch) -> None:
    client, sdk = _client(settings, monkeypatch)
    result = client.embed([str(i) for i in range(130)])
    assert len(result) == 130
    assert [len(call["input"]) for call in sdk.embed_calls] == [64, 64, 2]
