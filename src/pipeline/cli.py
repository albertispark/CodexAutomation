"""Typer command-line interface for the financial pipeline."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pipeline.cloud.analysis import DEFAULT_TASKS, TASK_DESCRIPTIONS
from pipeline.config import Settings, load_settings
from pipeline.indexing.chunker import chunk_documents
from pipeline.indexing.embedder import Embedder
from pipeline.indexing.vector_store import compute_index_key, create_vector_store
from pipeline.ingestion.ocr_gemma import _cache_path as ocr_cache_path
from pipeline.ingestion.ocr_gemma import _read_cache as read_ocr_cache
from pipeline.ingestion.router import SUPPORTED_EXTS, FileKind, detect_kind, ingest
from pipeline.local_llm.model_manager import ModelManager
from pipeline.local_llm.ollama_client import OllamaClient
from pipeline.orchestrator import (
    PipelineEnvironmentError,
    PipelineResult,
    pipeline_run_lock,
    poll_batch,
    run_pipeline,
)

app = typer.Typer(name="pipeline", no_args_is_help=True, add_completion=False)
console = Console()
_CONFIG_PATH: Path | None = None


@app.callback()
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        metavar="PATH",
        help=(
            "Path to settings.yaml; overrides $PIPELINE_CONFIG and "
            "./config/settings.yaml."
        ),
    )
) -> None:
    """Local-first financial document pipeline."""
    global _CONFIG_PATH
    _CONFIG_PATH = config


def _settings_or_exit() -> Settings:
    try:
        return load_settings(_CONFIG_PATH)
    except Exception as error:
        console.print(f"[red]Configuration error:[/red] {error}")
        raise typer.Exit(2) from error


def _parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in TASK_DESCRIPTIONS]
    if unknown:
        raise typer.BadParameter(
            f"unknown task(s): {', '.join(unknown)}; valid names: "
            f"{', '.join(TASK_DESCRIPTIONS)}",
            param_hint="--tasks",
        )
    if not tasks:
        raise typer.BadParameter("at least one task is required", param_hint="--tasks")
    return tasks


def _expand_paths(paths: list[Path] | None, settings: Settings) -> list[Path]:
    requested = list(paths or [settings.paths.inputs])
    expanded: set[Path] = set()
    for path in requested:
        path = Path(path).expanduser()
        if path.is_dir():
            expanded.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTS
            )
        else:
            expanded.add(path.resolve())
    return sorted(expanded, key=lambda item: str(item))


def _summary(results: list[PipelineResult]) -> None:
    table = Table("File", "Status", "Output", "Tokens", "Cost")
    for result in results:
        output = result.artifacts.get("xlsx") or result.artifacts.get("payload_json")
        table.add_row(
            result.input_path.name,
            result.status,
            str(output) if output else "—",
            str(result.token_usage.input_tokens + result.token_usage.output_tokens),
            f"${result.token_usage.estimated_cost_usd:.6f}",
        )
    console.print(table)


@app.command()
def run(
    paths: list[Path] | None = typer.Argument(
        None, help="Files/directories; defaults to paths.inputs."
    ),
    tasks: str = typer.Option(
        ",".join(DEFAULT_TASKS),
        "--tasks",
        help="Comma-separated analysis task names.",
    ),
    no_cloud: bool = typer.Option(
        False, "--no-cloud", help="Stop after Bouncer and save redacted payload JSON."
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        help="Submit uncached cloud jobs through the 50%-priced Message Batches API.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print redacted payloads and a local cost preview; make no cloud request.",
    ),
    allow_unredacted: bool = typer.Option(
        False,
        "--allow-unredacted",
        help="Explicit consent required for a run when redaction.enabled=false.",
    ),
) -> None:
    """Run the five-stage pipeline.

    Exit codes: 0 = all success (batch_pending counts as submitted success),
    1 = a file failed/quarantined/budget-stopped/refused, 2 = environment error.
    """
    if dry_run and batch:
        raise typer.BadParameter("--dry-run cannot be combined with --batch")
    if no_cloud and batch:
        raise typer.BadParameter("--no-cloud cannot be combined with --batch")
    settings = _settings_or_exit()
    if not settings.redaction.enabled and not allow_unredacted:
        console.print(
            "[red]Redaction is disabled.[/red] Re-run with --allow-unredacted "
            "to give explicit consent."
        )
        raise typer.Exit(2)
    task_names = _parse_tasks(tasks)
    input_paths = _expand_paths(paths, settings)
    try:
        results = run_pipeline(
            input_paths,
            settings,
            tasks=task_names,
            no_cloud=no_cloud,
            dry_run=dry_run,
            batch=batch,
            console=console,
        )
    except PipelineEnvironmentError as error:
        console.print(f"[red]Environment failure:[/red] {error}")
        raise typer.Exit(2) from error
    except Exception as error:
        console.print(f"[red]Environment failure:[/red] {error}")
        raise typer.Exit(2) from error
    _summary(results)
    good = {"success", "batch_pending"}
    raise typer.Exit(0 if all(result.status in good for result in results) else 1)


@app.command(name="batch-poll")
def batch_poll(
    batch_id: str = typer.Argument(..., help="Batch id printed by pipeline run --batch")
) -> None:
    """Collect an ended batch and complete workbook/audit output per result."""
    settings = _settings_or_exit()
    try:
        results = poll_batch(batch_id, settings, console=console)
    except PipelineEnvironmentError as error:
        console.print(f"[red]Environment failure:[/red] {error}")
        raise typer.Exit(2) from error
    except Exception as error:
        console.print(f"[red]Environment failure:[/red] {error}")
        raise typer.Exit(2) from error
    _summary(results)
    raise typer.Exit(0 if all(result.status == "success" for result in results) else 1)


@app.command()
def doctor() -> None:
    """Run health checks in the contract-defined order; exit 0 or 2."""
    table = Table("Check", "Result", "Fix / detail")
    failed = False
    try:
        settings = load_settings(_CONFIG_PATH)
        table.add_row("1. Config loads", "PASS", str(_CONFIG_PATH or "config/settings.yaml"))
    except Exception as error:
        table.add_row("1. Config loads", "FAIL", str(error))
        console.print(table)
        raise typer.Exit(2) from error

    client = OllamaClient(settings)
    installed: list[str] = []
    ollama_up = False
    try:
        installed = client.installed_models()
        ollama_up = True
        table.add_row("2. Ollama up", "PASS", settings.ollama.host)
    except Exception as error:
        failed = True
        table.add_row(
            "2. Ollama up",
            "FAIL",
            f"{error} Run 'ollama serve' or check ollama.host.",
        )

    required = [
        settings.ollama.ocr_model,
        settings.ollama.extract_model,
        settings.ollama.embed_model,
    ]
    missing = [
        model
        for model in required
        if not any(name == model or name == f"{model}:latest" for name in installed)
    ]
    if ollama_up and not missing:
        table.add_row("3. Models pulled", "PASS", "all configured models installed")
    elif ollama_up:
        failed = True
        table.add_row(
            "3. Models pulled",
            "FAIL",
            "; ".join(f"ollama pull {model}" for model in missing),
        )
    else:
        table.add_row("3. Models pulled", "FAIL", "Ollama must be reachable first")

    if ollama_up:
        try:
            loaded = client.loaded_models()
            unexpected = [
                name
                for name in loaded
                if not any(name == model or name == f"{model}:latest" for model in required)
            ]
            if unexpected:
                table.add_row(
                    "4. RAM headroom",
                    "WARN",
                    "; ".join(f"ollama stop {name}" for name in unexpected),
                )
            else:
                table.add_row("4. RAM headroom", "PASS", "no unrelated resident model")
        except Exception as error:
            failed = True
            table.add_row("4. RAM headroom", "FAIL", str(error))
    else:
        table.add_row("4. RAM headroom", "WARN", "not checked")

    if settings.cloud.api_key:
        gateway = f" via {settings.cloud.base_url}" if settings.cloud.base_url else ""
        table.add_row("5. API key", "PASS", f"configured{gateway}")
    else:
        table.add_row(
            "5. API key", "WARN", "Set ANTHROPIC_API_KEY or use --no-cloud"
        )

    writable = [
        settings.paths.inputs,
        settings.paths.outputs,
        settings.paths.cache,
        settings.paths.logs,
    ]
    for directory in writable:
        probe = directory / ".pipeline-write-probe"
        try:
            probe.touch(exist_ok=False)
            probe.unlink()
        except OSError as error:
            failed = True
            table.add_row("6. Dirs writable", "FAIL", f"{directory}: {error}")
            break
    else:
        table.add_row("6. Dirs writable", "PASS", "inputs/ outputs/ cache/ logs/")

    free = shutil.disk_usage(settings.paths.cache).free
    if free >= 2 * 1024**3:
        table.add_row("7. Disk space", "PASS", f"{free / 1024**3:.1f} GiB free")
    else:
        failed = True
        table.add_row("7. Disk space", "FAIL", "free at least 2 GiB on cache volume")

    console.print(table)
    raise typer.Exit(2 if failed else 0)


@app.command()
def index(file: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Ingest/index one file, then run an interactive local retrieval REPL."""
    settings = _settings_or_exit()
    ollama = OllamaClient(settings)
    manager = ModelManager(ollama, settings)
    try:
        with pipeline_run_lock(settings):
            plan = detect_kind(file, settings)
            if plan.kind is FileKind.UNSUPPORTED:
                console.print(f"[red]{plan.reason}[/red]")
                raise typer.Exit(1)
            ollama.health_check()
            ocr_miss = any(
                page.action == "ocr"
                and read_ocr_cache(
                    ocr_cache_path(settings, plan.file_sha256, page.page_number), settings
                )
                is None
                for page in plan.pages
            )
            if ocr_miss:
                manager.swap_to(settings.ollama.ocr_model)
            ingested = ingest(file, settings, ollama)
            if ocr_miss:
                manager.evict_large_models()
            chunks = chunk_documents(ingested, settings.index)
            if not chunks:
                console.print("No indexable content.")
                raise typer.Exit(1)
            ollama.warm_embed()
            embedder = Embedder(ollama, settings)
            vectors = embedder.embed_chunks(chunks)
            store = create_vector_store(settings)
            store.add(chunks, vectors)
            store.save(settings.paths.index_cache, compute_index_key(settings, ingested.doc_sha))
            console.print(f"Indexed {len(chunks)} chunks. Enter :q to exit.")
            while True:
                query = typer.prompt("query")
                if query.strip() == ":q":
                    break
                for result in store.search(embedder.embed_query(query), settings.index.top_k):
                    console.print(
                        f"{result.score:.4f}  {result.chunk.chunk_id}  "
                        f"page {result.chunk.page_start}-{result.chunk.page_end}"
                    )
                    console.print(result.chunk.text)
    except PipelineEnvironmentError as error:
        console.print(f"[red]Environment failure:[/red] {error}")
        raise typer.Exit(2) from error
    except typer.Exit:
        raise
    except Exception as error:
        console.print(f"[red]Environment failure:[/red] {error}")
        raise typer.Exit(2) from error
    finally:
        manager.release_all()


if __name__ == "__main__":
    app()
