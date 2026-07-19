"""Append-only per-file audit trail: the system's sole audit artifact."""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PipelineStatus = Literal[
    "success", "failed", "quarantined", "budget_stopped", "refused", "batch_pending"
]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class AuditRecord(BaseModel):
    run_id: str
    timestamp_utc: str
    input_file: str
    input_sha256: str
    stages_run: list[str]
    models_used: dict[str, str]
    token_usage: TokenUsage
    raw_tokens_est: int
    payload_tokens: int
    reduction_pct: float
    spend_usd: float
    output_path: str | None
    status: PipelineStatus
    error: str | None = None
    redaction_enabled: bool = True
    redaction_hits: dict[str, int] = Field(default_factory=dict)
    payload_sha256: str | None = None
    request_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AuditLog:
    def __init__(self, logs_dir: Path) -> None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / "audit.jsonl"

    def append(self, record: AuditRecord) -> Path:
        line = record.model_dump_json() + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return self.path
