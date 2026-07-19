"""Numpy-default and optional FAISS vector stores with atomic persistence."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pipeline.config import Settings
from pipeline.indexing.chunker import Chunk
from pipeline.indexing.embedder import EMBED_DIM


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]: ...
    def save(self, index_dir: Path, index_key: str) -> None: ...


def compute_index_key(cfg: Settings, doc_sha: str) -> str:
    source = (
        f"{doc_sha}:{cfg.ollama.embed_model}:"
        f"{cfg.index.chunk_tokens}:{cfg.index.chunk_overlap}"
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def index_paths(cfg: Settings, doc_sha: str) -> tuple[Path, Path]:
    base = cfg.paths.index_cache
    key = compute_index_key(cfg, doc_sha)
    return base / f"{key}.npz", base / f"{key}.chunks.json"


def _atomic_npz(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            np.savez(handle, vectors=vectors)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink(missing_ok=True)


class NumpyVectorStore:
    """Float32 matrix + parallel immutable chunk metadata."""

    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self.meta: dict[str, Any] = dict(meta or {})

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            return np.empty((0, int(self.meta.get("embed_dim", EMBED_DIM))), dtype=np.float32)
        return self._vectors.copy()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors)
        if vectors.dtype != np.float32:
            raise TypeError("vectors must have dtype float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("vector row count must equal chunk count")
        if self._vectors is not None and vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError("embedding dimensions differ")
        self._vectors = vectors.copy() if self._vectors is None else np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)
        if chunks:
            doc_shas = {chunk.doc_sha for chunk in self._chunks}
            if len(doc_shas) != 1:
                raise ValueError("a persisted vector store must contain one document")
            self.meta["doc_sha"] = next(iter(doc_shas))
        self.meta["embed_dim"] = int(vectors.shape[1]) if vectors.ndim == 2 else EMBED_DIM

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._vectors is None or not self._chunks or top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError("query embedding dimension does not match the store")
        scores = self._vectors @ query
        indices = np.argsort(scores, kind="stable")[::-1][: min(top_k, len(self._chunks))]
        return [SearchResult(self._chunks[int(index)], float(scores[int(index)])) for index in indices]

    def save(self, index_dir: Path, index_key: str) -> None:
        vectors = self.vectors
        meta = dict(self.meta)
        meta.setdefault("doc_sha", self._chunks[0].doc_sha if self._chunks else "")
        meta.setdefault("embed_model", "")
        meta.setdefault("chunk_tokens", 0)
        meta.setdefault("chunk_overlap", 0)
        meta["embed_dim"] = int(vectors.shape[1]) if vectors.ndim == 2 else EMBED_DIM
        _atomic_npz(Path(index_dir) / f"{index_key}.npz", vectors)
        _atomic_json(
            Path(index_dir) / f"{index_key}.chunks.json",
            {"meta": meta, "chunks": [asdict(chunk) for chunk in self._chunks]},
        )

    @classmethod
    def load(cls, index_dir: Path, index_key: str) -> "NumpyVectorStore":
        vector_path = Path(index_dir) / f"{index_key}.npz"
        chunks_path = Path(index_dir) / f"{index_key}.chunks.json"
        if not vector_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(f"incomplete index cache entry: {index_key}")
        with np.load(vector_path, allow_pickle=False) as archive:
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
            raise ValueError("index metadata is missing")
        chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        meta = payload["meta"]
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("index vector/chunk count mismatch")
        if int(meta.get("embed_dim", -1)) != vectors.shape[1]:
            raise ValueError("index embed_dim mismatch")
        if chunks and (
            any(chunk.doc_sha != meta.get("doc_sha") for chunk in chunks)
            or len({chunk.doc_sha for chunk in chunks}) != 1
        ):
            raise ValueError("index doc_sha metadata mismatch")
        store = cls(meta=meta)
        store._vectors = vectors
        store._chunks = chunks
        return store


class FaissVectorStore:
    """Optional ``faiss.IndexFlatIP`` backend over the same persisted files."""

    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "index.backend=faiss but faiss-cpu is not installed; "
                "pip install faiss-cpu or set index.backend: numpy"
            ) from exc
        self._faiss = faiss
        self._index = None
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self.meta: dict[str, Any] = dict(meta or {})

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors)
        if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("FAISS vectors must be float32 with one row per chunk")
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(vectors.shape[1])
            self._vectors = np.empty((0, vectors.shape[1]), dtype=np.float32)
        self._index.add(vectors)
        self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)
        self.meta.update(
            {
                "doc_sha": chunks[0].doc_sha if chunks else self.meta.get("doc_sha", ""),
                "embed_dim": vectors.shape[1],
            }
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._index is None or not self._chunks or top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, min(top_k, len(self._chunks)))
        return [
            SearchResult(self._chunks[int(index)], float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    def save(self, index_dir: Path, index_key: str) -> None:
        store = NumpyVectorStore(meta=self.meta)
        store._vectors = (
            self._vectors.copy()
            if self._vectors is not None
            else np.empty((0, int(self.meta.get("embed_dim", EMBED_DIM))), dtype=np.float32)
        )
        store._chunks = list(self._chunks)
        store.save(index_dir, index_key)

    @classmethod
    def load(cls, index_dir: Path, index_key: str) -> "FaissVectorStore":
        persisted = NumpyVectorStore.load(index_dir, index_key)
        store = cls(meta=persisted.meta)
        store.add(persisted.chunks, persisted.vectors)
        return store


def create_vector_store(cfg: Settings) -> VectorStore:
    meta = {
        "embed_model": cfg.ollama.embed_model,
        "chunk_tokens": cfg.index.chunk_tokens,
        "chunk_overlap": cfg.index.chunk_overlap,
        "embed_dim": EMBED_DIM,
    }
    if cfg.index.backend == "numpy":
        return NumpyVectorStore(meta=meta)
    if cfg.index.backend == "faiss":
        return FaissVectorStore(meta=meta)
    raise ValueError(f"Unsupported index backend: {cfg.index.backend}")


def load_vector_store(cfg: Settings, doc_sha: str) -> VectorStore:
    """Load and validate a configured cache entry; callers treat any error as a miss."""
    key = compute_index_key(cfg, doc_sha)
    cls = NumpyVectorStore if cfg.index.backend == "numpy" else FaissVectorStore
    store = cls.load(cfg.paths.index_cache, key)
    expected = {
        "doc_sha": doc_sha,
        "embed_model": cfg.ollama.embed_model,
        "chunk_tokens": cfg.index.chunk_tokens,
        "chunk_overlap": cfg.index.chunk_overlap,
        "embed_dim": EMBED_DIM,
    }
    if any(store.meta.get(name) != value for name, value in expected.items()):
        raise ValueError("index cache metadata mismatch")
    return store
