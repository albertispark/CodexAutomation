from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.extraction.bouncer import Bouncer, BouncerQuarantineError
from pipeline.indexing.chunker import Chunk
from pipeline.indexing.vector_store import SearchResult
from pipeline.ingestion.router import Document, FileKind, IngestedFile
from pipeline.local_llm.ollama_client import LocalInferenceError


def _raw(label: str = "Revenue") -> dict:
    return {
        "company": "Acme",
        "doc_type": "annual report",
        "currency_default": "USD",
        "periods": ["FY2025"],
        "figures": [
            {
                "label": label,
                "value": 1250.0,
                "unit": "thousands",
                "currency": "USD",
                "period": "FY2025",
                "statement": "income_statement",
                "source_page": 1,
                "verbatim_context": "Revenue 1,250",
            }
        ],
        "warnings": [],
    }


class FakeClient:
    def __init__(self, replies: list[object]) -> None:
        self.replies = replies
        self.calls: list[dict] = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.zeros(768, dtype=np.float32)


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, vector, top_k):
        return self.results[:top_k]


def _fixture():
    chunk = Chunk("sha:000001", "sha", "source.pdf", "native", 1, 1,
                  "Revenue | 1,250", 4, True)
    result = SearchResult(chunk, 0.9)
    ingested = IngestedFile(
        "sha", Path("source.pdf"), FileKind.NATIVE_PDF,
        [Document(Path("source.pdf"), 1, "page 1", chunk.text, "native")],
    )
    return result, ingested


def test_valid_extract_assigns_ids_and_caches(settings) -> None:
    search, ingested = _fixture()
    client = FakeClient([_raw()])
    embedder = FakeEmbedder()
    bouncer = Bouncer(client, embedder, FakeStore([search]), settings)
    first = bouncer.extract(ingested, "pdf")
    assert first.payload.figures[0].figure_id == "F0001"
    assert first.repair_attempted is False and first.from_cache is False
    assert client.calls[0]["think"] is False
    assert client.calls[0]["num_ctx"] == 8192
    assert client.calls[0]["messages"][1]["content"].startswith("Document type hint: pdf")
    second = bouncer.extract(ingested, "pdf")
    assert second.from_cache is True
    assert len(client.calls) == 1


def test_validation_repair_is_full_multiturn(settings) -> None:
    search, ingested = _fixture()
    client = FakeClient([{"company": "missing everything"}, _raw()])
    result = Bouncer(client, FakeEmbedder(), FakeStore([search]), settings).extract(
        ingested, "pdf"
    )
    assert result.repair_attempted
    roles = [message["role"] for message in client.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "validation" in client.calls[1]["messages"][-1]["content"].lower()


def test_two_invalid_responses_raise_without_cache(settings) -> None:
    search, ingested = _fixture()
    client = FakeClient([{}, {}])
    with pytest.raises(BouncerQuarantineError) as caught:
        Bouncer(client, FakeEmbedder(), FakeStore([search]), settings).extract(
            ingested, "pdf"
        )
    assert caught.value.chunk_ids == ["sha:000001"]
    assert not (settings.paths.bouncer_cache / "sha.payload.json").exists()


def test_empty_retrieval_never_calls_model(settings) -> None:
    _, ingested = _fixture()
    client = FakeClient([])
    with pytest.raises(BouncerQuarantineError, match="no_indexable_content"):
        Bouncer(client, FakeEmbedder(), FakeStore([]), settings).extract(ingested, "pdf")
    assert client.calls == []


def test_context_overflow_halves_once(settings) -> None:
    search, ingested = _fixture()
    overflow = LocalInferenceError("qwen3:8b", 9000, 8192)
    client = FakeClient([overflow, _raw()])
    result = Bouncer(client, FakeEmbedder(), FakeStore([search]), settings).extract(
        ingested, "other"
    )
    assert result.payload.figures
    assert len(client.calls) == 2


def test_retrieve_dedupes_max_score_then_document_order(settings) -> None:
    one = Chunk("sha:000002", "sha", "x", "native", 2, 2, "two", 1, False)
    two = Chunk("sha:000001", "sha", "x", "native", 1, 1, "one", 1, False)
    store = FakeStore([SearchResult(one, 0.5), SearchResult(two, 0.7), SearchResult(one, 0.9)])
    bouncer = Bouncer(FakeClient([]), FakeEmbedder(), store, settings, queries=("q",))
    results = bouncer.retrieve()
    assert [item.chunk.chunk_id for item in results] == ["sha:000001", "sha:000002"]
    assert results[1].score == 0.9


def test_malformed_cache_is_a_miss_and_is_overwritten(settings) -> None:
    search, ingested = _fixture()
    first_client = FakeClient([_raw()])
    Bouncer(
        first_client, FakeEmbedder(), FakeStore([search]), settings
    ).extract(ingested, "pdf")
    path = settings.paths.bouncer_cache / "sha.payload.json"
    cached = json.loads(path.read_text(encoding="utf-8"))
    cached["retrieved_chunk_ids"] = "not-a-list"
    cached["payload"]["figures"][0]["figure_id"] = ""
    path.write_text(json.dumps(cached), encoding="utf-8")

    second_client = FakeClient([_raw()])
    result = Bouncer(
        second_client, FakeEmbedder(), FakeStore([search]), settings
    ).extract(ingested, "pdf")
    assert result.from_cache is False
    assert len(second_client.calls) == 1
    assert result.payload.figures[0].figure_id == "F0001"
