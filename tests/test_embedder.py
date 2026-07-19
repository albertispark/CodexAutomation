from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.indexing.chunker import Chunk
from pipeline.indexing.embedder import DOC_PREFIX, EMBED_DIM, QUERY_PREFIX, Embedder


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        output = []
        for index, _ in enumerate(texts):
            vector = np.zeros(EMBED_DIM, dtype=np.float32)
            vector[index % EMBED_DIM] = 2.0
            output.append(vector.tolist())
        return output


def _chunk(text: str, seq: int) -> Chunk:
    return Chunk(f"sha:{seq:06d}", "sha", "x", "native", 1, 1, text, 1, False)


def test_prefixes_full_list_and_normalizes(settings) -> None:
    client = FakeClient()
    embedder = Embedder(client, settings)
    matrix = embedder.embed_chunks([_chunk("one", 1), _chunk("two", 2)])
    assert client.calls == [[DOC_PREFIX + "one", DOC_PREFIX + "two"]]
    assert matrix.shape == (2, EMBED_DIM)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])
    query = embedder.embed_query("balance sheet")
    assert client.calls[-1] == [QUERY_PREFIX + "balance sheet"]
    assert query.shape == (EMBED_DIM,)
