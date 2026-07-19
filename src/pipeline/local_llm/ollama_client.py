"""Typed wrapper over the official :mod:`ollama` Python SDK.

This is the single gateway for local inference. No other pipeline module
imports ``ollama`` directly.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx
from ollama import Client, ResponseError

from pipeline.config import Settings

logger = logging.getLogger("pipeline.local_llm.ollama_client")

UNLOAD_NOW: int = 0
EMBED_BATCH_SIZE: int = 64
OUTPUT_RESERVE_TOKENS: int = 3072
CHARS_PER_TOKEN_DENSE: int = 3
JSON_RETRY_PROMPT: str = (
    "Your previous reply was not valid JSON. Respond again with ONLY a single "
    "JSON object that conforms exactly to the schema. No prose, no markdown fences."
)

_CONNECTION_ERRORS = (httpx.ConnectError, httpx.TimeoutException, ConnectionError)


class OllamaNotRunningError(RuntimeError):
    """The Ollama daemon cannot be reached."""


class ModelMissingError(RuntimeError):
    """A configured local model is not installed."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"Required model '{model}' is not installed in Ollama.\n"
            f"Fix:  ollama pull {model}"
        )
        self.model = model


class LocalInferenceError(RuntimeError):
    """The assembled prompt would overflow the configured context window."""

    def __init__(self, model: str, estimated_tokens: int, num_ctx: int) -> None:
        super().__init__(
            f"Prompt for '{model}' estimated at {estimated_tokens} tokens exceeds "
            f"num_ctx={num_ctx} minus the {OUTPUT_RESERVE_TOKENS}-token output reserve."
        )
        self.model = model
        self.estimated_tokens = estimated_tokens
        self.num_ctx = num_ctx


class LocalJSONError(RuntimeError):
    """The extraction model emitted invalid JSON twice."""

    def __init__(self, model: str, raw_text: str) -> None:
        super().__init__(f"Model '{model}' produced unparseable JSON after 1 retry.")
        self.model = model
        self.raw_text = raw_text


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Read SDK response fields from either pydantic objects or mappings."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _model_name(value: Any) -> str:
    return str(_get(value, "model") or _get(value, "name") or "")


class OllamaClient:
    """Thin typed facade over one ``ollama.Client`` instance."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Client(host=settings.ollama.host)
        self.last_ocr_truncated: bool = False

    def _unreachable(self, exc: BaseException) -> OllamaNotRunningError:
        return OllamaNotRunningError(
            f"Cannot reach Ollama at {self._settings.ollama.host}. Start the Ollama "
            "app or run `ollama serve`, then re-run."
        )

    @staticmethod
    def _same_model(configured: str, reported: str) -> bool:
        return reported == configured or reported == f"{configured}:latest"

    def health_check(self) -> None:
        """Ping Ollama and verify all three configured models are installed."""
        installed = self.installed_models()
        required = (
            self._settings.ollama.ocr_model,
            self._settings.ollama.extract_model,
            self._settings.ollama.embed_model,
        )
        missing = [
            model
            for model in required
            if not any(self._same_model(model, present) for present in installed)
        ]
        if missing:
            logger.error("Missing configured Ollama models: %s", ", ".join(missing))
            raise ModelMissingError(missing[0])
        logger.info("Ollama resident models: %s", self.loaded_models())

    def installed_models(self) -> list[str]:
        """Return locally installed model names, raising the standard daemon error."""
        try:
            response = self._client.list()
        except _CONNECTION_ERRORS as exc:
            raise self._unreachable(exc) from exc
        models = _get(response, "models", []) or []
        return [_model_name(model) for model in models]

    def loaded_models(self) -> list[str]:
        """Return the names currently resident according to ``client.ps()``."""
        try:
            response = self._client.ps()
        except _CONNECTION_ERRORS as exc:
            raise self._unreachable(exc) from exc
        return [_model_name(model) for model in (_get(response, "models", []) or [])]

    def load_ping(self, model: str, keep_alive: str | int) -> None:
        """Load or unload a chat-capable model without useful generation."""
        try:
            try:
                self._client.chat(model=model, messages=[], keep_alive=keep_alive)
            except ResponseError:
                self._client.chat(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    options={"num_predict": 1},
                    keep_alive=keep_alive,
                )
        except _CONNECTION_ERRORS as exc:
            raise self._unreachable(exc) from exc

    def warm_embed(self, keep_alive: str | int | None = None) -> None:
        """Warm or evict the embedding model through the embed endpoint."""
        residency = self._settings.ollama.keep_alive if keep_alive is None else keep_alive
        try:
            self._client.embed(
                model=self._settings.ollama.embed_model,
                input=["warmup"],
                keep_alive=residency,
            )
        except _CONNECTION_ERRORS as exc:
            raise self._unreachable(exc) from exc

    def _chat(self, *, model: str, messages: list[dict], schema: dict,
              num_ctx: int, think: bool) -> Any:
        started = time.monotonic()
        try:
            response = self._client.chat(
                model=model,
                messages=messages,
                format=schema,
                options={"temperature": 0, "num_ctx": num_ctx},
                keep_alive=self._settings.ollama.keep_alive,
                think=think,
            )
        except _CONNECTION_ERRORS as exc:
            raise self._unreachable(exc) from exc
        logger.debug(
            "local chat model=%s duration=%.3fs prompt_eval_count=%s eval_count=%s",
            model,
            time.monotonic() - started,
            _get(response, "prompt_eval_count"),
            _get(response, "eval_count"),
        )
        if (_get(response, "prompt_eval_count", 0) or 0) >= num_ctx:
            logger.warning(
                "Ollama prompt_eval_count reached num_ctx=%d for model=%s; "
                "tune CHARS_PER_TOKEN_DENSE",
                num_ctx,
                model,
            )
        return response

    @staticmethod
    def _message_text(response: Any) -> str:
        message = _get(response, "message", {})
        return str(_get(message, "content", ""))

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        num_ctx: int,
        think: bool = False,
    ) -> dict:
        """Make a schema-constrained deterministic call with one JSON retry."""
        self._guard_context(model, messages, num_ctx)

        response = self._chat(
            model=model, messages=messages, schema=schema, num_ctx=num_ctx, think=think
        )
        text = self._message_text(response)
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError("top-level JSON is not an object", text, 0)
            return parsed
        except json.JSONDecodeError:
            retry_messages = [
                *messages,
                {"role": "assistant", "content": text},
                {"role": "user", "content": JSON_RETRY_PROMPT},
            ]
            # The correction history can itself exceed the context even when
            # the original request fit. Never let Ollama front-truncate it.
            self._guard_context(model, retry_messages, num_ctx)
            second = self._chat(
                model=model,
                messages=retry_messages,
                schema=schema,
                num_ctx=num_ctx,
                think=think,
            )
            second_text = self._message_text(second)
            try:
                parsed = json.loads(second_text)
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError(
                        "top-level JSON is not an object", second_text, 0
                    )
                return parsed
            except json.JSONDecodeError as exc:
                raise LocalJSONError(model, second_text) from exc

    @staticmethod
    def _guard_context(model: str, messages: list[dict], num_ctx: int) -> None:
        estimated = sum(
            len(str(message.get("content", ""))) for message in messages
        ) // CHARS_PER_TOKEN_DENSE
        if estimated + OUTPUT_RESERVE_TOKENS > num_ctx:
            raise LocalInferenceError(model, estimated, num_ctx)

    def ocr_image(
        self,
        image_bytes: bytes,
        prompt: str,
        keep_alive: str | int | None = None,
    ) -> str:
        """Transcribe a rendered page with the configured vision model."""
        residency = self._settings.ollama.keep_alive if keep_alive is None else keep_alive
        started = time.monotonic()
        try:
            response = self._client.chat(
                model=self._settings.ollama.ocr_model,
                messages=[
                    {"role": "user", "content": prompt, "images": [image_bytes]}
                ],
                options={"temperature": 0, "num_ctx": self._settings.ollama.num_ctx},
                keep_alive=residency,
            )
        except _CONNECTION_ERRORS as exc:
            raise self._unreachable(exc) from exc
        prompt_tokens = int(_get(response, "prompt_eval_count", 0) or 0)
        generated_tokens = int(_get(response, "eval_count", 0) or 0)
        self.last_ocr_truncated = (
            str(_get(response, "done_reason", "")) == "length"
            or prompt_tokens + generated_tokens >= self._settings.ollama.num_ctx
        )
        logger.debug(
            "local OCR model=%s duration=%.3fs prompt_eval_count=%s "
            "eval_count=%s done_reason=%s",
            self._settings.ollama.ocr_model,
            time.monotonic() - started,
            _get(response, "prompt_eval_count"),
            _get(response, "eval_count"),
            _get(response, "done_reason"),
        )
        return self._message_text(response)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed input in ordered batches of exactly at most 64."""
        if not texts:
            return []
        output: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            began = time.monotonic()
            try:
                response = self._client.embed(
                    model=self._settings.ollama.embed_model,
                    input=batch,
                )
            except _CONNECTION_ERRORS as exc:
                raise self._unreachable(exc) from exc
            output.extend(_get(response, "embeddings", []) or [])
            logger.debug(
                "local embed model=%s batch=%d duration=%.3fs",
                self._settings.ollama.embed_model,
                len(batch),
                time.monotonic() - began,
            )
        if len(output) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(output)} embeddings for {len(texts)} inputs"
            )
        return output
