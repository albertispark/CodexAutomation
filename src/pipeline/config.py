"""Configuration contract for the financial pipeline.

All modules receive a :class:`Settings` instance. YAML and environment
merging happen only in this module, and secrets are held as ``SecretStr``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator


class OllamaConfig(BaseModel):
    """Configuration bound to the ``ollama`` YAML block."""

    host: str = "http://localhost:11434"
    ocr_model: str = "gemma4:e4b"
    extract_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"
    keep_alive: str = "5m"
    num_ctx: int = Field(default=8192, ge=2048)


class CloudConfig(BaseModel):
    """Cloud configuration; key and gateway values are environment-owned."""

    model: str = "claude-opus-4-8"
    max_tokens: int = Field(default=8000, le=16000)
    enable_prompt_caching: bool = True
    monthly_budget_usd: float = Field(default=25.0, gt=0)
    api_key: SecretStr | None = None
    base_url: str | None = None


class PeerReviewConfig(BaseModel):
    """Independent OpenAI review; credentials are environment-owned."""

    enabled: bool = False
    model: str = "gpt-5.6-sol"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "medium"
    max_output_tokens: int = Field(default=12000, ge=1024, le=128000)
    api_key: SecretStr | None = None
    base_url: str | None = None


class IndexConfig(BaseModel):
    """Chunking and vector index configuration."""

    backend: Literal["numpy", "faiss"] = "numpy"
    chunk_tokens: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)
    top_k: int = Field(default=8, ge=1)

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_chunk(cls, value: int, info: Any) -> int:
        chunk_tokens = info.data.get("chunk_tokens", 512)
        if value >= chunk_tokens:
            raise ValueError("chunk_overlap must be less than chunk_tokens")
        return value


class PipelineConfig(BaseModel):
    """Input probing safeguards."""

    scanned_page_min_chars: int = Field(default=32, ge=0)
    max_file_mb: int = Field(default=50, ge=1)


class RedactionConfig(BaseModel):
    """Outbound regex redaction settings."""

    enabled: bool = True
    patterns: dict[str, str] = Field(default_factory=dict)

    @field_validator("patterns")
    @classmethod
    def _compilable(cls, value: dict[str, str]) -> dict[str, str]:
        for name, pattern in value.items():
            if not pattern:
                continue
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"redaction pattern {name!r} is invalid: {exc}") from exc
        return value


class PathsConfig(BaseModel):
    """Filesystem locations, resolved relative to the repository root."""

    inputs: Path = Path("inputs")
    outputs: Path = Path("outputs")
    cache: Path = Path("cache")
    logs: Path = Path("logs")

    @property
    def ocr_cache(self) -> Path:
        return self.cache / "ocr"

    @property
    def index_cache(self) -> Path:
        return self.cache / "index"

    @property
    def bouncer_cache(self) -> Path:
        return self.cache / "bouncer"

    @property
    def cloud_cache(self) -> Path:
        return self.cache / "cloud"

    @property
    def review_cache(self) -> Path:
        return self.cache / "review"

    @property
    def quarantine_dir(self) -> Path:
        return self.outputs / "quarantine"

    @property
    def review_dir(self) -> Path:
        return self.outputs / "reviews"

    @property
    def batches_dir(self) -> Path:
        return self.outputs / "batches"

    def ensure(self) -> None:
        for path in (
            self.inputs,
            self.outputs,
            self.quarantine_dir,
            self.review_dir,
            self.batches_dir,
            self.ocr_cache,
            self.index_cache,
            self.bouncer_cache,
            self.cloud_cache,
            self.review_cache,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Root settings object injected into every component."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    review: PeerReviewConfig = Field(default_factory=PeerReviewConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is not None:
        candidate = Path(path).expanduser()
    elif os.environ.get("PIPELINE_CONFIG"):
        candidate = Path(os.environ["PIPELINE_CONFIG"]).expanduser()
    else:
        candidate = Path("config/settings.yaml")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {candidate}. Create "
            "config/settings.yaml, pass --config PATH, or set $PIPELINE_CONFIG."
        )
    return candidate.resolve()


def load_settings(path: str | Path | None = None) -> Settings:
    """Load YAML, merge approved environment overrides, resolve paths, and mkdir."""

    # PIPELINE_CONFIG itself is allowed in the project dotenv files, so they
    # must run before the resolution chain. .env.local has higher priority
    # because both loads preserve variables that are already set.
    if path is None:
        load_dotenv(Path(".env.local"), override=False)
        load_dotenv(Path(".env"), override=False)
    config_path = _resolve_config_path(path)
    repo_root = config_path.parent.parent
    load_dotenv(repo_root / ".env.local", override=False)
    load_dotenv(repo_root / ".env", override=False)

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    raw: dict[str, Any] = loaded

    cloud = raw.setdefault("cloud", {})
    cloud["api_key"] = os.environ.get("ANTHROPIC_API_KEY") or None
    # These fields are environment-owned even if somebody accidentally puts
    # them in YAML; never retain a YAML secret or gateway value.
    cloud["base_url"] = os.environ.get("ANTHROPIC_BASE_URL") or None
    review = raw.setdefault("review", {})
    review["api_key"] = os.environ.get("OPENAI_API_KEY") or None
    review["base_url"] = os.environ.get("OPENAI_BASE_URL") or None
    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        raw.setdefault("ollama", {})["host"] = ollama_host

    settings = Settings.model_validate(raw)
    rebased: dict[str, Path] = {}
    for name in ("inputs", "outputs", "cache", "logs"):
        configured = getattr(settings.paths, name).expanduser()
        rebased[name] = configured if configured.is_absolute() else repo_root / configured
    settings = settings.model_copy(
        update={"paths": settings.paths.model_copy(update=rebased)}
    )
    settings.paths.ensure()
    return settings
