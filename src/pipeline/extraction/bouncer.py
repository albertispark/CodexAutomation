"""Retrieve, locally distill, validate, assign ids, and cache—or raise."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from pydantic import ValidationError

from pipeline.config import Settings
from pipeline.extraction.schemas import ExtractionPayload, bouncer_schema
from pipeline.indexing.embedder import Embedder
from pipeline.indexing.vector_store import SearchResult, VectorStore
from pipeline.ingestion.router import IngestedFile
from pipeline.local_llm.ollama_client import (
    CHARS_PER_TOKEN_DENSE,
    OUTPUT_RESERVE_TOKENS,
    LocalInferenceError,
    LocalJSONError,
    OllamaClient,
)

BOUNCER_PROMPT_VERSION: str = "1"
DEFAULT_RETRIEVAL_QUERIES: tuple[str, ...] = (
    "balance sheet",
    "income statement",
    "operating margins",
    "adjustments",
    "revenue",
    "cash flow",
)

BOUNCER_SYSTEM_PROMPT: str = """\
You are a financial data extraction engine. You receive excerpts from ONE \
financial document, separated by markers like "--- [page 3-4] ---". Return a \
single JSON object conforming to the provided schema. Rules:

1. Extract ONLY raw financial figures that appear verbatim in the excerpts: \
line items from the balance sheet, income statement, and cash flow statement, \
stated margins and ratios, and any stated adjustments or restatements.
2. NEVER compute, derive, aggregate, or infer a number. If a subtotal or total \
is not printed in the text, do not create it. Copy printed values exactly, \
stripping currency symbols and thousands separators; parentheses mean negative.
3. If a figure is present but unreadable, garbled, or ambiguous, set its \
"value" to null and add a short note to "warnings".
4. Discard everything that is not a financial figure: legal boilerplate, \
safe-harbor and forward-looking-statement disclaimers, auditor opinion text, \
narrative commentary, marketing language, page headers and footers, tables of \
contents.
5. For every figure, set "verbatim_context" to the exact source snippet \
(maximum 200 characters) that contains it, so a human can audit the extraction.
6. Record "unit" and "period" exactly as the document labels them. If a table \
states a scale once (e.g. "in thousands"), apply it to that table's figures.
7. Set "source_page" from the nearest page marker; use null if unclear.
8. Output the JSON object only. No explanations, no markdown fences.
"""

REPAIR_PROMPT_TEMPLATE: str = """\
Your previous JSON response failed schema validation with these errors:

{errors}

Return a corrected JSON object that satisfies the schema. Fix ONLY the \
validation errors; do not add, remove, or alter any figures. Output JSON only.
"""


class BouncerQuarantineError(RuntimeError):
    """Local extraction cannot safely progress; the orchestrator quarantines."""

    def __init__(
        self, raw_response: dict | str, errors: str, chunk_ids: list[str]
    ) -> None:
        self.raw_response = raw_response
        self.errors = errors
        self.chunk_ids = chunk_ids
        super().__init__(f"Bouncer output failed validation twice: {errors[:200]}")


@dataclass(frozen=True)
class BouncerResult:
    payload: ExtractionPayload
    retrieved_chunk_ids: list[str]
    context_token_estimate: int
    repair_attempted: bool
    from_cache: bool


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink(missing_ok=True)


class Bouncer:
    def __init__(
        self,
        client: OllamaClient,
        embedder: Embedder,
        store: VectorStore,
        cfg: Settings,
        queries: tuple[str, ...] = DEFAULT_RETRIEVAL_QUERIES,
        query_vectors: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._store = store
        self._cfg = cfg
        self._queries = queries
        self._query_vectors = dict(query_vectors or {})

    def retrieve(self) -> list[SearchResult]:
        deduped: dict[str, SearchResult] = {}
        for query in self._queries:
            vector = self._query_vectors.get(query)
            if vector is None:
                vector = self._embedder.embed_query(query)
            for result in self._store.search(vector, self._cfg.index.top_k):
                prior = deduped.get(result.chunk.chunk_id)
                if prior is None or result.score > prior.score:
                    deduped[result.chunk.chunk_id] = result
        return sorted(
            deduped.values(), key=lambda item: (item.chunk.page_start, item.chunk.chunk_id)
        )

    @staticmethod
    def _render_context(results: list[SearchResult]) -> str:
        return "\n\n".join(
            f"--- [page {result.chunk.page_start}-{result.chunk.page_end}] ---\n\n"
            f"{result.chunk.text}"
            for result in results
        )

    def build_context(
        self, results: list[SearchResult], doc_type_hint: str = "unknown"
    ) -> str:
        survivors = list(results)
        system_tokens = len(BOUNCER_SYSTEM_PROMPT) // CHARS_PER_TOKEN_DENSE
        hint_tokens = len(f"Document type hint: {doc_type_hint}\n\n") // CHARS_PER_TOKEN_DENSE
        while survivors:
            context = self._render_context(survivors)
            total = system_tokens + hint_tokens + len(context) // CHARS_PER_TOKEN_DENSE
            if total + OUTPUT_RESERVE_TOKENS <= self._cfg.ollama.num_ctx:
                return context
            lowest = min(
                survivors,
                key=lambda item: (item.score, item.chunk.chunk_id),
            )
            survivors.remove(lowest)
        return ""

    @staticmethod
    def _schema_sha() -> str:
        encoded = json.dumps(
            bouncer_schema(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _queries_sha(self) -> str:
        encoded = json.dumps(list(self._queries), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, file_sha: str) -> Path:
        return self._cfg.paths.bouncer_cache / f"{file_sha}.payload.json"

    def _determinants(self, doc_type_hint: str) -> dict[str, object]:
        return {
            "prompt_version": BOUNCER_PROMPT_VERSION,
            "schema_sha": self._schema_sha(),
            "extract_model": self._cfg.ollama.extract_model,
            "top_k": self._cfg.index.top_k,
            "num_ctx": self._cfg.ollama.num_ctx,
            "queries_sha": self._queries_sha(),
            "doc_type_hint": doc_type_hint,
        }

    def _read_cache(self, file_sha: str, doc_type_hint: str) -> BouncerResult | None:
        try:
            data = json.loads(self._cache_path(file_sha).read_text(encoding="utf-8"))
            expected = self._determinants(doc_type_hint)
            if any(data.get(key) != value for key, value in expected.items()):
                return None
            payload = ExtractionPayload.model_validate(data["payload"])
            if any(
                figure.figure_id != f"F{index:04d}"
                for index, figure in enumerate(payload.figures, start=1)
            ):
                return None
            raw_chunk_ids = data["retrieved_chunk_ids"]
            if not isinstance(raw_chunk_ids, list) or not all(
                isinstance(value, str) for value in raw_chunk_ids
            ):
                return None
            chunk_ids = list(raw_chunk_ids)
            context_token_estimate = int(data["context_token_estimate"])
            if context_token_estimate < 0:
                return None
            return BouncerResult(
                payload=payload,
                retrieved_chunk_ids=chunk_ids,
                context_token_estimate=context_token_estimate,
                repair_attempted=False,
                from_cache=True,
            )
        except (OSError, ValueError, TypeError, KeyError, ValidationError):
            return None

    def has_valid_cache(self, file_sha: str, doc_type_hint: str) -> bool:
        return self._read_cache(file_sha, doc_type_hint) is not None

    def _call(
        self, messages: list[dict], chunk_ids: list[str]
    ) -> dict:
        try:
            return self._client.chat_json(
                model=self._cfg.ollama.extract_model,
                messages=messages,
                schema=bouncer_schema(),
                num_ctx=self._cfg.ollama.num_ctx,
                think=False,
            )
        except LocalJSONError as exc:
            raise BouncerQuarantineError(
                exc.raw_text, "invalid_json", chunk_ids
            ) from exc

    @staticmethod
    def _halve_context(context: str) -> str:
        if not context:
            return context
        target = max(1, len(context) // 2)
        cut = context.rfind("\n\n--- [page ", 0, target)
        if cut > 0:
            return context[:cut]
        return context[:target]

    def extract(self, ingested: IngestedFile, doc_type_hint: str) -> BouncerResult:
        cached = self._read_cache(ingested.doc_sha, doc_type_hint)
        if cached is not None:
            return cached
        results = self.retrieve()
        chunk_ids = [result.chunk.chunk_id for result in results]
        if not results:
            raise BouncerQuarantineError("", "no_indexable_content", [])
        context = self.build_context(results, doc_type_hint)
        if not context:
            raise BouncerQuarantineError("", "context_overflow", chunk_ids)
        user_message = f"Document type hint: {doc_type_hint}\n\n{context}"
        messages = [
            {"role": "system", "content": BOUNCER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        try:
            raw = self._call(messages, chunk_ids)
        except LocalInferenceError:
            context = self._halve_context(context)
            user_message = f"Document type hint: {doc_type_hint}\n\n{context}"
            messages = [
                {"role": "system", "content": BOUNCER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            try:
                raw = self._call(messages, chunk_ids)
            except LocalInferenceError as exc:
                raise BouncerQuarantineError(
                    "", "context_overflow", chunk_ids
                ) from exc

        repair_attempted = False
        try:
            payload = ExtractionPayload.model_validate(raw)
        except ValidationError as first_error:
            repair_attempted = True
            repair_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                },
                {
                    "role": "user",
                    "content": REPAIR_PROMPT_TEMPLATE.format(errors=str(first_error)),
                },
            ]
            try:
                repaired = self._call(repair_messages, chunk_ids)
            except LocalInferenceError as exc:
                raise BouncerQuarantineError(
                    raw, "context_overflow", chunk_ids
                ) from exc
            try:
                payload = ExtractionPayload.model_validate(repaired)
            except ValidationError as second_error:
                raise BouncerQuarantineError(
                    repaired, str(second_error), chunk_ids
                ) from second_error

        for index, figure in enumerate(payload.figures, start=1):
            figure.figure_id = f"F{index:04d}"
        context_estimate = len(context) // CHARS_PER_TOKEN_DENSE
        cache_payload = {
            **self._determinants(doc_type_hint),
            "retrieved_chunk_ids": chunk_ids,
            "context_token_estimate": context_estimate,
            "payload": payload.model_dump(mode="json"),
        }
        _atomic_json(self._cache_path(ingested.doc_sha), cache_payload)
        return BouncerResult(
            payload=payload,
            retrieved_chunk_ids=chunk_ids,
            context_token_estimate=context_estimate,
            repair_attempted=repair_attempted,
            from_cache=False,
        )
