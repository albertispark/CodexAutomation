"""RAM residency policy for a 16 GB host: at most one large model loaded."""
from __future__ import annotations

import logging
import time

from pipeline.config import Settings
from pipeline.local_llm.ollama_client import OllamaClient, UNLOAD_NOW

logger = logging.getLogger("pipeline.local_llm.model_manager")

SWAP_TIMEOUT_S: float = 120.0
SWAP_POLL_INTERVAL_S: float = 0.5
_ADVICE = (
    "Set OLLAMA_MAX_LOADED_MODELS=1 in the Ollama server environment and "
    "close memory-heavy apps."
)


class ModelSwapError(RuntimeError):
    """The daemon failed to evict or load within the bounded swap window."""


class ModelManager:
    """Enforce mutually exclusive residency for the OCR/extraction models."""

    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._large_models: tuple[str, str] = (
            settings.ollama.ocr_model,
            settings.ollama.extract_model,
        )

    @staticmethod
    def _same_model(configured: str, reported: str) -> bool:
        return reported == configured or reported == f"{configured}:latest"

    def _is_loaded(self, configured: str, loaded: list[str]) -> bool:
        return any(self._same_model(configured, reported) for reported in loaded)

    def _other_large_loaded(self, target: str, loaded: list[str]) -> list[str]:
        return [
            configured
            for configured in self._large_models
            if not self._same_model(configured, target)
            and self._is_loaded(configured, loaded)
        ]

    def swap_to(self, model: str) -> None:
        """Make one configured large model the sole resident large model."""
        target = next(
            (item for item in self._large_models if self._same_model(item, model)),
            None,
        )
        if target is None:
            raise ValueError(
                f"swap_to accepts only configured large models: {self._large_models!r}"
            )
        started = time.monotonic()
        deadline = started + SWAP_TIMEOUT_S
        loaded = self._client.loaded_models()
        others = self._other_large_loaded(target, loaded)
        if self._is_loaded(target, loaded) and not others:
            logger.info("swap_to: %s already sole resident", target)
            return

        for other in others:
            self._client.load_ping(other, keep_alive=UNLOAD_NOW)
        while True:
            loaded = self._client.loaded_models()
            if not self._other_large_loaded(target, loaded):
                break
            if time.monotonic() >= deadline:
                raise ModelSwapError(
                    f"Timed out evicting other large models; ps={loaded!r}. {_ADVICE}"
                )
            time.sleep(SWAP_POLL_INTERVAL_S)

        self._client.load_ping(target, self._settings.ollama.keep_alive)
        while True:
            loaded = self._client.loaded_models()
            if self._is_loaded(target, loaded):
                logger.info(
                    "swap_to: %s loaded in %.2fs", target, time.monotonic() - started
                )
                return
            if time.monotonic() >= deadline:
                raise ModelSwapError(
                    f"Timed out loading {target!r}; ps={loaded!r}. {_ADVICE} "
                    "Check for a concurrent pipeline invocation and reduce ollama.num_ctx."
                )
            time.sleep(SWAP_POLL_INTERVAL_S)

    def evict_large_models(self) -> None:
        """Evict every configured large model and verify the result."""
        deadline = time.monotonic() + SWAP_TIMEOUT_S
        loaded = self._client.loaded_models()
        for configured in self._large_models:
            if self._is_loaded(configured, loaded):
                self._client.load_ping(configured, keep_alive=UNLOAD_NOW)
        while True:
            loaded = self._client.loaded_models()
            if not any(self._is_loaded(item, loaded) for item in self._large_models):
                return
            if time.monotonic() >= deadline:
                raise ModelSwapError(
                    f"Timed out evicting large models; ps={loaded!r}. {_ADVICE}"
                )
            time.sleep(SWAP_POLL_INTERVAL_S)

    def release_all(self) -> None:
        """Best-effort eviction of both large models; never raise."""
        try:
            loaded = self._client.loaded_models()
            for configured in self._large_models:
                if self._is_loaded(configured, loaded):
                    try:
                        self._client.load_ping(configured, keep_alive=UNLOAD_NOW)
                    except Exception:
                        logger.exception("Failed to release model %s", configured)
        except Exception:
            logger.exception("Could not inspect loaded Ollama models during cleanup")
