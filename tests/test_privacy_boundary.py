from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from pydantic import SecretStr
from rich.console import Console

from pipeline.cloud.analysis import build_user_message
from pipeline.cloud.claude_client import ClaudeClient
from pipeline.extraction.bouncer import Bouncer
from pipeline.extraction.schemas import ExtractionPayload, FinancialFigure, StatementType
from pipeline.indexing.embedder import Embedder
from pipeline.indexing.vector_store import NumpyVectorStore
from pipeline.ingestion.router import detect_kind
from pipeline.orchestrator import run_pipeline

SRC = Path(__file__).parents[1] / "src" / "pipeline"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_only_claude_client_imports_anthropic() -> None:
    importers = []
    for path in SRC.rglob("*.py"):
        if any(name == "anthropic" or name.startswith("anthropic.") for name in _imports(path)):
            importers.append(path.relative_to(SRC).as_posix())
    assert importers == ["cloud/claude_client.py"]


def test_cloud_client_has_no_ingestion_or_indexing_imports() -> None:
    imports = _imports(SRC / "cloud" / "claude_client.py")
    assert not any(name.startswith("pipeline.ingestion") for name in imports)
    assert not any(name.startswith("pipeline.indexing") for name in imports)


def test_cloud_entrypoints_are_typed_redacted_only() -> None:
    analyze = inspect.signature(ClaudeClient.analyze)
    message = inspect.signature(build_user_message)
    assert "RedactedPayload" in str(analyze.parameters["redacted"].annotation)
    assert "RedactedPayload" in str(message.parameters["redacted"].annotation)


def test_redacted_payload_constructor_is_source_local() -> None:
    users = []
    for path in SRC.rglob("*.py"):
        if "RedactedPayload(" in path.read_text(encoding="utf-8"):
            users.append(path.relative_to(SRC).as_posix())
    assert users == ["extraction/redactor.py"]


def test_seeded_identifiers_never_cross_boundary_or_enter_logs(
    settings, tmp_path: Path, monkeypatch, fake_anthropic
) -> None:
    identifiers = [
        "123-45-6789",
        "alice@example.com",
        "415-555-1212",
        "GB82 WEST 1234 5698 7654 32",
        "GB82WEST12345698765432",
        "4111 1111 1111 1111",
    ]
    source = tmp_path / "privacy.csv"
    source.write_text("Metric,FY2025\nRevenue,1250\n", encoding="utf-8")
    plan = detect_kind(source, settings)
    payload = ExtractionPayload(
        company=f"Acme {identifiers[0]}",
        doc_type=f"annual report {identifiers[1]}",
        currency_default="USD",
        periods=[f"FY2025 {identifiers[2]}"],
        figures=[
            FinancialFigure(
                figure_id="F0001",
                label=f"Revenue {identifiers[3]}",
                value=1250,
                unit="thousands",
                currency="USD",
                period=f"FY2025 {identifiers[4]}",
                statement=StatementType.income_statement,
                source_page=1,
                verbatim_context=f"Revenue 1,250 {identifiers[5]}",
            )
        ],
        warnings=[f"Contact {identifiers[5]}"],
    )

    class NoCallOllama:
        def __init__(self, cfg):
            self.cfg = cfg

        def __getattr__(self, name):
            raise AssertionError(f"unexpected Ollama call: {name}")

    helper = Bouncer(
        NoCallOllama(settings),
        Embedder(NoCallOllama(settings), settings),
        NumpyVectorStore(),
        settings,
    )
    cache = {
        **helper._determinants("csv"),
        "retrieved_chunk_ids": [f"{plan.file_sha256}:000001"],
        "context_token_estimate": 50,
        "payload": payload.model_dump(mode="json"),
    }
    (settings.paths.bouncer_cache / f"{plan.file_sha256}.payload.json").write_text(
        json.dumps(cache), encoding="utf-8"
    )
    settings.cloud.api_key = SecretStr("test")
    monkeypatch.setattr("pipeline.orchestrator.OllamaClient", NoCallOllama)
    result = run_pipeline(
        [source], settings, console=Console(file=None, quiet=True)
    )[0]
    assert result.status == "success"

    parse_call = next(
        data for name, data in fake_anthropic[0].messages.calls if name == "parse"
    )
    outbound = str(parse_call["messages"][0]["content"])
    log_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in settings.paths.logs.iterdir()
        if path.is_file()
    )
    for identifier in identifiers:
        assert identifier not in outbound
        assert identifier not in log_text
