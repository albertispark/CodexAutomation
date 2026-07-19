"""Nomic task prefixes and L2 normalization; batching stays in OllamaClient."""
from __future__ import annotations

import numpy as np

from pipeline.config import Settings
from pipeline.indexing.chunker import Chunk
from pipeline.local_llm.ollama_client import OllamaClient

EMBED_DIM: int = 768
DOC_PREFIX: str = "search_document: "
QUERY_PREFIX: str = "search_query: "


class Embedder:
    """Embed chunks and queries with the required nomic prefixes."""

    def __init__(self, client: OllamaClient, cfg: Settings) -> None:
        self._client = client
        self._cfg = cfg

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        values = self._client.embed([DOC_PREFIX + chunk.text for chunk in chunks])
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (len(chunks), EMBED_DIM):
            raise ValueError(
                f"Expected embeddings shape {(len(chunks), EMBED_DIM)}, got {matrix.shape}"
            )
        return self._l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        values = self._client.embed([QUERY_PREFIX + text])
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (1, EMBED_DIM):
            raise ValueError(f"Expected query embedding shape (1, {EMBED_DIM}), got {matrix.shape}")
        return self._l2_normalize(matrix)[0]

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.maximum(norms, np.float32(1e-12))).astype(np.float32)
