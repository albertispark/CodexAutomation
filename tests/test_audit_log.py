from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.output.audit_log import AuditLog, AuditRecord, TokenUsage, sha256_file


def _record(path: Path, status="success") -> AuditRecord:
    return AuditRecord(
        run_id="run",
        timestamp_utc="2026-07-16T00:00:00+00:00",
        input_file=str(path.resolve()),
        input_sha256="a" * 64,
        stages_run=["ingestion", "indexing", "bouncer", "cloud", "excel"],
        models_used={"ocr": "gemma4:e4b", "extract": "qwen3:8b",
                     "embed": "nomic-embed-text", "cloud": "claude-opus-4-8"},
        token_usage=TokenUsage(input_tokens=1, output_tokens=2,
                               cache_read_input_tokens=3, cache_creation_input_tokens=4),
        raw_tokens_est=1000,
        payload_tokens=100,
        reduction_pct=0.9,
        spend_usd=0.01,
        output_path="result.xlsx",
        status=status,
        redaction_enabled=True,
        redaction_hits={"email": 1},
        payload_sha256="b" * 64,
        request_sha256="c" * 64,
    )


def test_append_is_one_line_and_roundtrips(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    path = AuditLog(tmp_path / "logs").append(_record(source))
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    loaded = AuditRecord.model_validate_json(lines[0])
    assert loaded.token_usage.cache_creation_input_tokens == 4
    assert loaded.redaction_hits == {"email": 1}
    assert json.loads(lines[0])["status"] == "success"
    assert sha256_file(source) == "41cf6794ba4200b839c53531555f0f3998df4cbb01a4d5cb0b94e3ca5e23947d"


def test_status_literal_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _record(tmp_path / "x", status="unknown")


def test_audit_contains_no_document_body(tmp_path: Path) -> None:
    secret_body = "CONFIDENTIAL_FINANCIAL_NARRATIVE_123456789"
    source = tmp_path / "source.pdf"
    source.write_text(secret_body)
    audit = AuditLog(tmp_path / "logs").append(_record(source))
    encoded = audit.read_text()
    for start in range(0, len(secret_body) - 20):
        assert secret_body[start : start + 21] not in encoded
