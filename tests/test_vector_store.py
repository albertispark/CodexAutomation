from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.indexing.chunker import Chunk
from pipeline.indexing.vector_store import (
    NumpyVectorStore,
    compute_index_key,
    create_vector_store,
    load_vector_store,
)


def _chunks() -> list[Chunk]:
    return [
        Chunk("doc:000001", "doc", "x", "native", 1, 1, "assets", 2, False),
        Chunk("doc:000002", "doc", "x", "excel", 3, 3, "revenue", 2, True),
    ]


def test_numpy_search_roundtrip_bit_exact(tmp_path: Path) -> None:
    vectors = np.zeros((2, 768), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    store = NumpyVectorStore()
    store.add(_chunks(), vectors)
    assert store.search(vectors[1], 1)[0].chunk.chunk_id == "doc:000002"
    assert abs(store.search(vectors[1], 1)[0].score - 1.0) < 1e-5
    store.save(tmp_path, "key")
    loaded = NumpyVectorStore.load(tmp_path, "key")
    np.testing.assert_array_equal(loaded.vectors, vectors)
    assert loaded.chunks == _chunks()


def test_empty_store_searches_empty() -> None:
    assert NumpyVectorStore().search(np.zeros(768, dtype=np.float32), 8) == []


def test_composite_key_changes_with_each_determinant(settings) -> None:
    original = compute_index_key(settings, "doc")
    settings.index.chunk_tokens += 1
    assert compute_index_key(settings, "doc") != original
    settings.index.chunk_tokens -= 1
    settings.ollama.embed_model = "other"
    assert compute_index_key(settings, "doc") != original


def test_factory_cache_meta_validates(settings) -> None:
    vectors = np.zeros((2, 768), dtype=np.float32)
    vectors[:, :2] = np.eye(2, dtype=np.float32)
    store = create_vector_store(settings)
    store.add(_chunks(), vectors)
    key = compute_index_key(settings, "doc")
    store.save(settings.paths.index_cache, key)
    loaded = load_vector_store(settings, "doc")
    assert loaded.search(vectors[0], 1)[0].chunk.chunk_id == "doc:000001"
