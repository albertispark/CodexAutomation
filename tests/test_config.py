from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pipeline.config import IndexConfig, load_settings


def _write_config(root: Path, **overrides) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    data = {
        "paths": {"inputs": "inputs", "outputs": "outputs", "cache": "cache", "logs": "logs"},
        **overrides,
    }
    path = config_dir / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_paths_rebased_and_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_settings(_write_config(tmp_path))
    assert cfg.paths.inputs == tmp_path / "inputs"
    assert cfg.paths.bouncer_cache.is_dir()
    assert cfg.paths.cloud_cache.is_dir()
    assert cfg.paths.review_cache.is_dir()
    assert cfg.paths.quarantine_dir.is_dir()
    assert cfg.paths.review_dir.is_dir()
    assert cfg.paths.batches_dir.is_dir()


def test_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_config(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "review-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-gateway.invalid/v1")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.invalid")
    cfg = load_settings(path)
    assert cfg.cloud.api_key is not None
    assert cfg.cloud.api_key.get_secret_value() == "secret"
    assert cfg.cloud.base_url == "https://gateway.invalid"
    assert cfg.review.api_key is not None
    assert cfg.review.api_key.get_secret_value() == "review-secret"
    assert cfg.review.base_url == "https://openai-gateway.invalid/v1"
    assert cfg.ollama.host == "http://ollama.invalid"
    assert "secret" not in repr(cfg)
    assert "review-secret" not in repr(cfg)


def test_cloud_endpoint_and_secret_are_environment_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    path = _write_config(
        tmp_path,
        cloud={"api_key": "yaml-secret", "base_url": "https://yaml.invalid"},
        review={
            "api_key": "yaml-review-secret",
            "base_url": "https://yaml-review.invalid",
        },
    )
    cfg = load_settings(path)
    assert cfg.cloud.api_key is None
    assert cfg.cloud.base_url is None
    assert cfg.review.api_key is None
    assert cfg.review.base_url is None


def test_dotenv_local_precedes_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write_config(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=shared\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text(
        "OPENAI_API_KEY=local\n", encoding="utf-8"
    )
    cfg = load_settings(path)
    assert cfg.review.api_key is not None
    assert cfg.review.api_key.get_secret_value() == "local"


def test_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "env"
    explicit_root = tmp_path / "explicit"
    env_path = _write_config(env_root, ollama={"host": "http://env"})
    explicit_path = _write_config(explicit_root, ollama={"host": "http://explicit"})
    monkeypatch.setenv("PIPELINE_CONFIG", str(env_path))
    assert load_settings(explicit_path).ollama.host == "http://explicit"
    assert load_settings().ollama.host == "http://env"


def test_pipeline_config_can_come_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured_root = tmp_path / "selected"
    configured = _write_config(
        configured_root, ollama={"host": "http://dotenv"}
    )
    monkeypatch.delenv("PIPELINE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"PIPELINE_CONFIG={configured}\n", encoding="utf-8"
    )
    assert load_settings().ollama.host == "http://dotenv"


def test_missing_config_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--config PATH.*PIPELINE_CONFIG"):
        load_settings(tmp_path / "missing.yaml")


def test_overlap_must_be_less_than_chunk() -> None:
    with pytest.raises(ValidationError):
        IndexConfig(chunk_tokens=64, chunk_overlap=64)


def test_bad_redaction_regex_fails_load(tmp_path: Path) -> None:
    path = _write_config(tmp_path, redaction={"patterns": {"broken": "["}})
    with pytest.raises(ValidationError, match="broken"):
        load_settings(path)
