"""Stage-major pipeline orchestration with content caches and one run lock."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from pipeline.cloud.analysis import (
    CLOUD_PROMPT_VERSION,
    DEFAULT_TASKS,
    build_user_message,
    outbound_payload_dict,
)
from pipeline.cloud.claude_client import (
    PRICE_PER_MTOK,
    AnalysisResult,
    BudgetExceededError,
    ClaudeClient,
    CloudRefusalError,
)
from pipeline.cloud.openai_reviewer import (
    REVIEW_PROMPT_VERSION,
    OpenAIReviewer,
    PeerReviewRefusalError,
    PeerReviewResult,
    ReviewUsage,
    build_review_user_message,
    validate_peer_review,
)
from pipeline.config import Settings
from pipeline.extraction.bouncer import (
    DEFAULT_RETRIEVAL_QUERIES,
    Bouncer,
    BouncerQuarantineError,
)
from pipeline.extraction.redactor import RedactedPayload, Redactor
from pipeline.extraction.schemas import ExtractionPayload
from pipeline.indexing.chunker import Chunk, chunk_documents
from pipeline.indexing.embedder import Embedder
from pipeline.indexing.vector_store import (
    NumpyVectorStore,
    VectorStore,
    compute_index_key,
    create_vector_store,
    load_vector_store,
)
from pipeline.ingestion.excel_reader import UnreadableWorkbookError
from pipeline.ingestion.ocr_gemma import (
    DEGRADED_FILE_QUARANTINE_RATIO,
    LOW_CONFIDENCE_MARKER,
    _cache_path as ocr_cache_path,
    _read_cache as read_ocr_cache,
)
from pipeline.ingestion.router import (
    FileKind,
    IngestedFile,
    IngestPlan,
    detect_kind,
    ingest,
)
from pipeline.local_llm.model_manager import ModelManager, ModelSwapError
from pipeline.local_llm.ollama_client import OllamaClient, OllamaNotRunningError, UNLOAD_NOW
from pipeline.output.audit_log import (
    AuditLog,
    AuditRecord,
    PipelineStatus,
    TokenUsage as AuditTokenUsage,
)
from pipeline.output.excel_writer import output_filename, write_workbook

logger = logging.getLogger("pipeline.orchestrator")


class PipelineEnvironmentError(RuntimeError):
    """An environment failure maps to CLI exit code 2."""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class PipelineResult:
    input_path: Path
    content_sha256: str
    status: PipelineStatus = "success"
    artifacts: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    error: str | None = None
    quarantine_reason: str | None = None
    stages_run: list[str] = field(default_factory=list)
    raw_tokens_est: int = 0
    payload_tokens: int = 0
    payload_sha256: str | None = None
    request_sha256: str | None = None
    review_request_sha256: str | None = None
    review_verdict: str | None = None
    redaction_hits: dict[str, int] = field(default_factory=dict)
    audit_written: bool = False


@dataclass
class _FileCtx:
    path: Path
    sha256: str
    result: PipelineResult
    plan: IngestPlan | None = None
    ingested: IngestedFile | None = None
    chunks: list[Chunk] = field(default_factory=list)
    store: VectorStore | None = None
    bouncer_cache_hit: bool = False
    index_cache_hit: bool = False
    payload: ExtractionPayload | None = None
    redacted: RedactedPayload | None = None
    analysis: AnalysisResult | None = None

    @property
    def active(self) -> bool:
        return self.result.status == "success"


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(
    redacted: RedactedPayload, settings: Settings, tasks: list[str]
) -> tuple[str, str]:
    outbound = build_user_message(redacted, tasks).encode("utf-8")
    payload_sha = hashlib.sha256(outbound).hexdigest()
    request_material = (
        payload_sha
        + settings.cloud.model
        + CLOUD_PROMPT_VERSION
        + ",".join(tasks)
    )
    request_sha = hashlib.sha256(request_material.encode("utf-8")).hexdigest()
    return payload_sha, request_sha


def _atomic_json(path: Path, data: dict[str, Any], *, pretty: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            if pretty:
                json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2)
            else:
                json.dump(
                    data,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return path
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def pipeline_run_lock(settings: Settings) -> Iterator[None]:
    """Acquire the process-wide single-writer lock without waiting."""
    lock_path = settings.paths.logs / ".run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineEnvironmentError("another pipeline invocation is running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _run_logging(settings: Settings, run_id: str) -> Iterator[Path]:
    """Attach one bounded local log file for this invocation."""
    path = settings.paths.logs / f"run-{run_id}.log"
    handler = RotatingFileHandler(
        path, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    pipeline_logger = logging.getLogger("pipeline")
    prior_level = pipeline_logger.level
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.addHandler(handler)
    try:
        logger.info("run=%s started", run_id)
        yield path
    finally:
        logger.info("run=%s finished", run_id)
        pipeline_logger.removeHandler(handler)
        pipeline_logger.setLevel(prior_level)
        handler.close()


def _quarantine(
    ctx: _FileCtx,
    stage: str,
    reason: str,
    settings: Settings,
    detail: dict | None = None,
) -> None:
    ctx.result.status = "quarantined"
    ctx.result.quarantine_reason = reason
    ctx.result.error = f"DataQuarantine: {reason}"
    destination = settings.paths.quarantine_dir / (
        f"{ctx.path.stem}.{ctx.sha256[:12]}.json"
    )
    _atomic_json(
        destination,
        {
            "input_path": str(ctx.path.resolve()),
            "content_sha256": ctx.sha256,
            "stage": stage,
            "reason": reason,
            "detail": detail or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        pretty=True,
    )
    ctx.result.artifacts["quarantine_json"] = destination.resolve()
    logger.warning("file=%s quarantined reason=%s", ctx.sha256[:12], reason)


def _doc_type_hint(ctx: _FileCtx) -> str:
    suffix = ctx.path.suffix.lower().lstrip(".")
    return suffix or (ctx.plan.kind.value if ctx.plan else "unknown")


def _ocr_plan_has_miss(plan: IngestPlan, settings: Settings) -> bool:
    for page in plan.pages:
        if page.action != "ocr":
            continue
        if read_ocr_cache(
            ocr_cache_path(settings, plan.file_sha256, page.page_number), settings
        ) is None:
            return True
    return False


def _store_has_chunks(store: VectorStore | None) -> bool:
    return bool(getattr(store, "chunks", []))


def _is_memory_error(error: BaseException) -> bool:
    text = str(error).lower()
    return error.__class__.__name__ == "ResponseError" and any(
        marker in text for marker in ("memory", "out of memory", "system memory")
    )


def _is_disk_full(error: BaseException) -> bool:
    return isinstance(error, OSError) and error.errno == errno.ENOSPC


def _mark_failed(ctx: _FileCtx, error: BaseException | str, reason: str) -> None:
    ctx.result.status = "failed"
    class_name = error if isinstance(error, str) else error.__class__.__name__
    ctx.result.error = f"{class_name}: {reason}"


def _redaction_counts(redacted: RedactedPayload) -> dict[str, int]:
    return dict(Counter(event.pattern_name for event in redacted.events))


def _payload_output_path(ctx: _FileCtx, settings: Settings) -> Path:
    return settings.paths.outputs / f"{ctx.path.stem}.{ctx.sha256[:12]}.payload.json"


def _write_redacted_payload(ctx: _FileCtx, settings: Settings) -> Path:
    if ctx.redacted is None:
        raise RuntimeError("redacted payload unavailable")
    destination = _payload_output_path(ctx, settings)
    _atomic_json(destination, outbound_payload_dict(ctx.redacted), pretty=True)
    ctx.result.artifacts["payload_json"] = destination.resolve()
    return destination


def _cloud_cache_path(settings: Settings, request_sha: str) -> Path:
    return settings.paths.cloud_cache / f"{request_sha}.json"


def _read_cloud_cache(
    settings: Settings, request_sha: str, tasks: list[str]
) -> tuple[AnalysisResult, int] | None:
    try:
        data = json.loads(
            _cloud_cache_path(settings, request_sha).read_text(encoding="utf-8")
        )
        if data.get("tasks") != tasks or data.get("model") != settings.cloud.model:
            return None
        result = AnalysisResult.model_validate(data["result"])
        payload_tokens = int(data["payload_tokens"])
        if payload_tokens < 0:
            return None
        return result, payload_tokens
    except (OSError, ValueError, TypeError, KeyError, ValidationError):
        return None


def _write_cloud_cache(
    settings: Settings,
    request_sha: str,
    result: AnalysisResult,
    payload_tokens: int,
    tasks: list[str],
) -> Path:
    return _atomic_json(
        _cloud_cache_path(settings, request_sha),
        {
            "result": result.model_dump(mode="json"),
            "payload_tokens": payload_tokens,
            "tasks": tasks,
            "model": settings.cloud.model,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    )


def _review_request_sha256(
    redacted: RedactedPayload,
    analysis: AnalysisResult,
    settings: Settings,
    tasks: list[str],
) -> str:
    request = build_review_user_message(redacted, tasks, analysis)
    material = json.dumps(
        {
            "request": request,
            "model": settings.review.model,
            "reasoning_effort": settings.review.reasoning_effort,
            "max_output_tokens": settings.review.max_output_tokens,
            "prompt_version": REVIEW_PROMPT_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _review_cache_path(settings: Settings, request_sha: str) -> Path:
    return settings.paths.review_cache / f"{request_sha}.json"


def _read_review_cache(
    settings: Settings,
    request_sha: str,
    tasks: list[str],
) -> PeerReviewResult | None:
    try:
        data = json.loads(
            _review_cache_path(settings, request_sha).read_text(encoding="utf-8")
        )
        if (
            data.get("tasks") != tasks
            or data.get("model") != settings.review.model
            or data.get("reasoning_effort") != settings.review.reasoning_effort
            or data.get("prompt_version") != REVIEW_PROMPT_VERSION
        ):
            return None
        return PeerReviewResult.model_validate(data["result"])
    except (OSError, ValueError, TypeError, KeyError, ValidationError):
        return None


def _write_review_cache(
    settings: Settings,
    request_sha: str,
    review: PeerReviewResult,
    tasks: list[str],
) -> Path:
    return _atomic_json(
        _review_cache_path(settings, request_sha),
        {
            "result": review.model_dump(mode="json"),
            "tasks": tasks,
            "model": settings.review.model,
            "reasoning_effort": settings.review.reasoning_effort,
            "prompt_version": REVIEW_PROMPT_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    )


def _merge_token_usage(current: TokenUsage, usage: ReviewUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=current.input_tokens + usage.input_tokens,
        output_tokens=current.output_tokens + usage.output_tokens,
        cache_read_input_tokens=(
            current.cache_read_input_tokens + usage.cache_read_input_tokens
        ),
        cache_creation_input_tokens=(
            current.cache_creation_input_tokens + usage.cache_creation_input_tokens
        ),
        # OpenAI model pricing is not hardcoded here. The existing estimate is
        # the Anthropic amount recorded by its budget ledger.
        estimated_cost_usd=current.estimated_cost_usd,
    )


def _apply_review_result(
    ctx: _FileCtx,
    review: PeerReviewResult,
    original: AnalysisResult,
    settings: Settings,
    *,
    cache_hit: bool,
) -> None:
    if ctx.redacted is None or ctx.result.review_request_sha256 is None:
        raise RuntimeError("review context is incomplete")
    validate_peer_review(review, original, ctx.redacted)
    destination = settings.paths.review_dir / (
        f"{ctx.path.stem}.{ctx.sha256[:12]}.review.json"
    )
    _atomic_json(
        destination,
        {
            "review_request_sha256": ctx.result.review_request_sha256,
            "review_prompt_version": REVIEW_PROMPT_VERSION,
            "claude_model": settings.cloud.model,
            "review_model": settings.review.model,
            "verdict": review.verdict,
            "issues": [issue.model_dump(mode="json") for issue in review.issues],
            "claude_analysis": original.model_dump(mode="json"),
            "reviewed_analysis": review.reviewed_analysis.model_dump(mode="json"),
            "cache_hit": cache_hit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        pretty=True,
    )
    ctx.result.artifacts["review_json"] = destination.resolve()
    ctx.result.review_verdict = review.verdict
    ctx.result.stages_run.append("review")
    if review.verdict == "rejected":
        _quarantine(
            ctx,
            "review",
            "openai_peer_review_rejected",
            settings,
            {
                "review_artifact": str(destination.resolve()),
                "issues": [issue.model_dump(mode="json") for issue in review.issues],
            },
        )
        return
    ctx.analysis = review.reviewed_analysis


def _usage_from_sdk(usage: Any, cost: float) -> TokenUsage:
    def counter(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    return TokenUsage(
        input_tokens=counter("input_tokens"),
        output_tokens=counter("output_tokens"),
        cache_read_input_tokens=counter("cache_read_input_tokens"),
        cache_creation_input_tokens=counter("cache_creation_input_tokens"),
        estimated_cost_usd=cost,
    )


def _audit_context(
    ctx: _FileCtx,
    settings: Settings,
    run_id: str,
    audit: AuditLog,
) -> None:
    if ctx.result.audit_written:
        return
    status = ctx.result.status
    output_path = ctx.result.artifacts.get("xlsx")
    reduction = 0.0
    if ctx.result.payload_tokens and ctx.result.raw_tokens_est:
        reduction = 1.0 - (
            ctx.result.payload_tokens / ctx.result.raw_tokens_est
        )
    models_used = {
        "ocr": settings.ollama.ocr_model,
        "extract": settings.ollama.extract_model,
        "embed": settings.ollama.embed_model,
        "cloud": settings.cloud.model,
    }
    if settings.review.enabled:
        models_used["review"] = settings.review.model
    record = AuditRecord(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        input_file=str(ctx.path.resolve()),
        input_sha256=ctx.sha256,
        stages_run=list(ctx.result.stages_run),
        models_used=models_used,
        token_usage=AuditTokenUsage(
            input_tokens=ctx.result.token_usage.input_tokens,
            output_tokens=ctx.result.token_usage.output_tokens,
            cache_read_input_tokens=ctx.result.token_usage.cache_read_input_tokens,
            cache_creation_input_tokens=ctx.result.token_usage.cache_creation_input_tokens,
        ),
        raw_tokens_est=ctx.result.raw_tokens_est,
        payload_tokens=ctx.result.payload_tokens,
        reduction_pct=reduction if ctx.result.payload_tokens else 0.0,
        spend_usd=ctx.result.token_usage.estimated_cost_usd,
        output_path=str(output_path.resolve()) if output_path else None,
        status=status,
        error=ctx.result.error,
        redaction_enabled=settings.redaction.enabled,
        redaction_hits=ctx.result.redaction_hits,
        payload_sha256=ctx.result.payload_sha256,
        request_sha256=ctx.result.request_sha256,
    )
    audit.append(record)
    ctx.result.audit_written = True


def _audit_all(
    ctxs: list[_FileCtx], settings: Settings, run_id: str, audit: AuditLog
) -> None:
    for ctx in ctxs:
        _audit_context(ctx, settings, run_id, audit)


@contextmanager
def _audit_on_exit(
    ctxs: list[_FileCtx], settings: Settings, run_id: str, audit: AuditLog
) -> Iterator[None]:
    """Guarantee one best-effort terminal audit line per discovered file."""
    try:
        yield
    finally:
        _audit_all(ctxs, settings, run_id, audit)


def _bouncer_cache_probe(
    ctx: _FileCtx,
    ollama: OllamaClient,
    settings: Settings,
) -> bool:
    helper = Bouncer(
        ollama,
        Embedder(ollama, settings),
        NumpyVectorStore(),
        settings,
    )
    return helper.has_valid_cache(ctx.sha256, _doc_type_hint(ctx))


def _retry_ingest_memory(
    ctx: _FileCtx,
    settings: Settings,
    ollama: OllamaClient,
    manager: ModelManager,
) -> IngestedFile:
    try:
        return ingest(ctx.path, settings, ollama)
    except Exception as error:
        if not _is_memory_error(error):
            raise
        manager.release_all()
        manager.swap_to(settings.ollama.ocr_model)
        return ingest(ctx.path, settings, ollama)


def _embed_with_memory_retry(
    embedder: Embedder,
    chunks: list[Chunk],
    ollama: OllamaClient,
    manager: ModelManager,
) -> np.ndarray:
    try:
        return embedder.embed_chunks(chunks)
    except Exception as error:
        if not _is_memory_error(error):
            raise
        manager.release_all()
        try:
            ollama.warm_embed(UNLOAD_NOW)
        except Exception:
            logger.exception("Best-effort embedder eviction failed")
        ollama.warm_embed()
        return embedder.embed_chunks(chunks)


def _embed_query_with_memory_retry(
    embedder: Embedder,
    query: str,
    ollama: OllamaClient,
    manager: ModelManager,
) -> np.ndarray:
    try:
        return embedder.embed_query(query)
    except Exception as error:
        if not _is_memory_error(error):
            raise
        manager.release_all()
        try:
            ollama.warm_embed(UNLOAD_NOW)
        except Exception:
            logger.exception("Best-effort embedder eviction failed")
        ollama.warm_embed()
        return embedder.embed_query(query)


def _bounce_with_memory_retry(
    bouncer: Bouncer,
    ingested: IngestedFile,
    hint: str,
    settings: Settings,
    manager: ModelManager,
) -> Any:
    try:
        return bouncer.extract(ingested, hint)
    except Exception as error:
        if not _is_memory_error(error):
            raise
        manager.release_all()
        manager.swap_to(settings.ollama.extract_model)
        return bouncer.extract(ingested, hint)


def _environment_abort(
    ctxs: list[_FileCtx],
    settings: Settings,
    run_id: str,
    audit: AuditLog,
    reason: str,
) -> None:
    for ctx in ctxs:
        if (
            reason == "disk_full"
            and ctx.result.status == "success"
            and any(
                key in ctx.result.artifacts for key in ("xlsx", "payload_json")
            )
        ):
            # A previously published terminal artifact remains a completed
            # result even if a later file exhausts the volume.
            continue
        if ctx.payload is not None and ctx.redacted is not None:
            try:
                _write_redacted_payload(ctx, settings)
                ctx.result.status = "failed"
                ctx.result.error = f"EnvironmentFailure: {reason}_payload_preserved"
            except Exception:
                _mark_failed(ctx, "EnvironmentFailure", reason)
        elif ctx.result.status == "success":
            _mark_failed(ctx, "EnvironmentFailure", reason)
    _audit_all(ctxs, settings, run_id, audit)


@contextmanager
def _disk_full_guard(
    ctxs: list[_FileCtx], settings: Settings, run_id: str, audit: AuditLog
) -> Iterator[None]:
    """Map an unhandled ENOSPC from any atomic artifact to exit code 2."""
    try:
        yield
    except OSError as error:
        if not _is_disk_full(error):
            raise
        _environment_abort(ctxs, settings, run_id, audit, "disk_full")
        raise PipelineEnvironmentError("disk_full") from error


@contextmanager
def _terminal_failure_guard(ctxs: list[_FileCtx]) -> Iterator[None]:
    """Never let an aborted invocation leave an active file marked success."""
    try:
        yield
    except Exception as error:
        if _is_disk_full(error) or isinstance(error, PipelineEnvironmentError):
            raise
        for ctx in ctxs:
            if ctx.active:
                _mark_failed(ctx, error, "pipeline_aborted")
        raise


def run_pipeline(
    input_paths: list[Path],
    settings: Settings,
    *,
    tasks: list[str] = DEFAULT_TASKS,
    no_cloud: bool = False,
    dry_run: bool = False,
    batch: bool = False,
    console: Console | None = None,
) -> list[PipelineResult]:
    """Execute all six stages in stage-major order across every input."""
    console = console or Console()
    run_id = uuid.uuid4().hex
    audit = AuditLog(settings.paths.logs)
    ctxs: list[_FileCtx] = []
    models_touched = False

    with (
        pipeline_run_lock(settings),
        _run_logging(settings, run_id),
        _audit_on_exit(ctxs, settings, run_id, audit),
        _disk_full_guard(ctxs, settings, run_id, audit),
        _terminal_failure_guard(ctxs),
        Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", markup=False),
            BarColumn(),
            TextColumn("{task.completed:.0f}/{task.total:.0f} files"),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        total_files = len(input_paths)
        ingest_progress = progress.add_task(
            "[1/6] OCR / Parse", total=total_files
        )
        index_progress = progress.add_task("[2/6] Index", total=total_files)
        bounce_progress = progress.add_task("[3/6] Bouncer", total=total_files)
        cloud_progress = progress.add_task("[4/6] Claude", total=total_files)
        review_progress = progress.add_task("[5/6] OpenAI review", total=total_files)
        excel_progress = progress.add_task("[6/6] Excel", total=total_files)

        # A4 plan-first: no daemon call and no model load during this pass.
        for path in input_paths:
            path = Path(path)
            plan = detect_kind(path, settings)
            result = PipelineResult(path, plan.file_sha256)
            ctxs.append(_FileCtx(path, plan.file_sha256, result, plan=plan))

        ollama = OllamaClient(settings)
        manager = ModelManager(ollama, settings)
        embedder = Embedder(ollama, settings)
        redactor = Redactor(settings)

        # Probe bouncer and index caches before health/model calls.
        for ctx in ctxs:
            if ctx.plan is None or ctx.plan.kind is FileKind.UNSUPPORTED:
                continue
            ctx.bouncer_cache_hit = _bouncer_cache_probe(ctx, ollama, settings)
            if not ctx.bouncer_cache_hit:
                try:
                    ctx.store = load_vector_store(settings, ctx.sha256)
                    ctx.index_cache_hit = True
                except Exception:
                    ctx.store = None

        ocr_miss = any(
            ctx.plan is not None
            and ctx.plan.kind is not FileKind.UNSUPPORTED
            and _ocr_plan_has_miss(ctx.plan, settings)
            for ctx in ctxs
        )
        local_work_needed = ocr_miss or any(
            ctx.plan is not None
            and ctx.plan.kind is not FileKind.UNSUPPORTED
            and not ctx.bouncer_cache_hit
            for ctx in ctxs
        )

        if local_work_needed:
            try:
                ollama.health_check()
            except Exception as error:
                _environment_abort(ctxs, settings, run_id, audit, "ollama_preflight")
                raise PipelineEnvironmentError(str(error)) from error

        try:
            # ------------------------------- Stage A: ingestion / OCR
            if ocr_miss:
                try:
                    models_touched = True
                    manager.swap_to(settings.ollama.ocr_model)
                except ModelSwapError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "model_swap_timeout")
                    raise PipelineEnvironmentError("model_swap_timeout") from error

            for ctx in ctxs:
                progress.update(
                    ingest_progress,
                    description=f"[1/6] OCR / Parse {ctx.path.name}",
                )
                if not ctx.active or ctx.plan is None:
                    progress.advance(ingest_progress)
                    continue
                started = time.monotonic()
                if ctx.plan.kind is FileKind.UNSUPPORTED:
                    _quarantine(ctx, "ingest", ctx.plan.reason, settings)
                    ctx.result.timings["ingest"] = time.monotonic() - started
                    progress.advance(ingest_progress)
                    continue
                try:
                    ctx.ingested = _retry_ingest_memory(
                        ctx, settings, ollama, manager
                    )
                    ctx.result.stages_run.append("ingestion")
                    ctx.result.raw_tokens_est = sum(
                        len(document.text) for document in ctx.ingested.documents
                    ) // 4
                    ocr_documents = [
                        document
                        for document in ctx.ingested.documents
                        if document.origin == "ocr"
                    ]
                    degraded = [
                        document.page_number
                        for document in ocr_documents
                        if LOW_CONFIDENCE_MARKER.split("{", 1)[0] in document.text
                    ]
                    if (
                        ocr_documents
                        and len(degraded) / len(ocr_documents)
                        > DEGRADED_FILE_QUARANTINE_RATIO
                    ):
                        _quarantine(
                            ctx,
                            "ingest",
                            "ocr_degraded",
                            settings,
                            {"failing_pages": degraded},
                        )
                except UnreadableWorkbookError:
                    _quarantine(ctx, "ingest", "unreadable_workbook", settings)
                except OllamaNotRunningError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "ollama_mid_run")
                    raise PipelineEnvironmentError("ollama_mid_run") from error
                except Exception as error:
                    if _is_disk_full(error):
                        _environment_abort(ctxs, settings, run_id, audit, "disk_full")
                        raise PipelineEnvironmentError("disk_full") from error
                    _mark_failed(ctx, error, "ingest_failed")
                finally:
                    ctx.result.timings["ingest"] = time.monotonic() - started
                    progress.advance(ingest_progress)
            progress.update(
                ingest_progress,
                description="[1/6] OCR / Parse",
                completed=total_files,
            )

            # Prepare chunks before deciding whether the embedder needs warming.
            index_miss_with_content = False
            for ctx in ctxs:
                if not ctx.active or ctx.ingested is None or ctx.bouncer_cache_hit:
                    continue
                if ctx.store is not None:
                    ctx.result.stages_run.append("indexing")
                    continue
                ctx.chunks = chunk_documents(ctx.ingested, settings.index)
                if ctx.chunks:
                    index_miss_with_content = True

            if ocr_miss:
                try:
                    manager.evict_large_models()
                except ModelSwapError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "model_swap_timeout")
                    raise PipelineEnvironmentError("model_swap_timeout") from error
            if index_miss_with_content:
                try:
                    ollama.warm_embed()
                except OllamaNotRunningError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "ollama_mid_run")
                    raise PipelineEnvironmentError("ollama_mid_run") from error
                except Exception as error:
                    if _is_memory_error(error):
                        try:
                            ollama.warm_embed(UNLOAD_NOW)
                        except Exception:
                            logger.exception("Best-effort embedder eviction failed")
                        ollama.warm_embed()
                    else:
                        raise

            # ------------------------------- Stage B: chunk / embed / persist
            for ctx in ctxs:
                progress.update(
                    index_progress,
                    description=f"[2/6] Index {ctx.path.name}",
                )
                if (
                    not ctx.active
                    or ctx.ingested is None
                    or ctx.bouncer_cache_hit
                    or ctx.store is not None
                ):
                    progress.advance(index_progress)
                    continue
                started = time.monotonic()
                try:
                    store = create_vector_store(settings)
                    if ctx.chunks:
                        vectors = _embed_with_memory_retry(
                            embedder, ctx.chunks, ollama, manager
                        )
                        store.add(ctx.chunks, vectors)
                        store.save(
                            settings.paths.index_cache,
                            compute_index_key(settings, ctx.sha256),
                        )
                    ctx.store = store
                    ctx.result.stages_run.append("indexing")
                except OllamaNotRunningError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "ollama_mid_run")
                    raise PipelineEnvironmentError("ollama_mid_run") from error
                except Exception as error:
                    if _is_disk_full(error):
                        _environment_abort(ctxs, settings, run_id, audit, "disk_full")
                        raise PipelineEnvironmentError("disk_full") from error
                    _mark_failed(ctx, error, "indexing_failed")
                finally:
                    ctx.result.timings["index"] = time.monotonic() - started
                    progress.advance(index_progress)
            progress.update(
                index_progress,
                description="[2/6] Index",
                completed=total_files,
            )

            bouncer_misses = [
                ctx
                for ctx in ctxs
                if ctx.active
                and ctx.ingested is not None
                and not ctx.bouncer_cache_hit
                and _store_has_chunks(ctx.store)
            ]
            query_vectors: dict[str, np.ndarray] = {}
            if bouncer_misses:
                # A2: each retrieval query is embedded once here; Stage C uses
                # only this mapping and makes zero embed calls.
                try:
                    for query in DEFAULT_RETRIEVAL_QUERIES:
                        query_vectors[query] = _embed_query_with_memory_retry(
                            embedder, query, ollama, manager
                        )
                except OllamaNotRunningError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "ollama_mid_run")
                    raise PipelineEnvironmentError("ollama_mid_run") from error
                except Exception as error:
                    for affected in bouncer_misses:
                        _mark_failed(affected, error, "query_embedding_failed")
                    bouncer_misses = []
                    query_vectors = {}
            if bouncer_misses:
                try:
                    models_touched = True
                    manager.swap_to(settings.ollama.extract_model)
                except ModelSwapError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "model_swap_timeout")
                    raise PipelineEnvironmentError("model_swap_timeout") from error

            # ------------------------------- Stage C: Bouncer + redaction
            for ctx in ctxs:
                progress.update(
                    bounce_progress,
                    description=f"[3/6] Bouncer {ctx.path.name}",
                )
                if not ctx.active or ctx.ingested is None:
                    progress.advance(bounce_progress)
                    continue
                started = time.monotonic()
                try:
                    if not ctx.bouncer_cache_hit and not _store_has_chunks(ctx.store):
                        raise BouncerQuarantineError(
                            "", "no_indexable_content", []
                        )
                    store = ctx.store or NumpyVectorStore()
                    bouncer = Bouncer(
                        ollama,
                        embedder,
                        store,
                        settings,
                        query_vectors=query_vectors,
                    )
                    bounced = _bounce_with_memory_retry(
                        bouncer,
                        ctx.ingested,
                        _doc_type_hint(ctx),
                        settings,
                        manager,
                    )
                    ctx.payload = bounced.payload
                    ctx.redacted = redactor.redact_payload(ctx.payload)
                    ctx.result.redaction_hits = _redaction_counts(ctx.redacted)
                    ctx.result.stages_run.append("bouncer")
                except BouncerQuarantineError as error:
                    reason = (
                        error.errors
                        if error.errors in {"no_indexable_content", "context_overflow"}
                        else "bouncer_schema_invalid"
                    )
                    _quarantine(
                        ctx,
                        "bounce",
                        reason,
                        settings,
                        {
                            "raw_response": error.raw_response,
                            "errors": error.errors,
                            "chunk_ids": error.chunk_ids,
                        },
                    )
                except OllamaNotRunningError as error:
                    _environment_abort(ctxs, settings, run_id, audit, "ollama_mid_run")
                    raise PipelineEnvironmentError("ollama_mid_run") from error
                except Exception as error:
                    if _is_disk_full(error):
                        raise
                    _mark_failed(ctx, error, "bouncer_failed")
                finally:
                    ctx.result.timings["bounce"] = time.monotonic() - started
                    progress.advance(bounce_progress)
            progress.update(
                bounce_progress,
                description="[3/6] Bouncer",
                completed=total_files,
            )

        finally:
            # A4 + A15: clean up iff this invocation touched resident models;
            # a fully cached rerun makes zero Ollama calls of any kind.
            if models_touched:
                manager.release_all()

        # A9 is a data gate, not merely a cloud-cost optimization: every run
        # mode reports an empty validated extraction as quarantined.
        for ctx in ctxs:
            if ctx.active and ctx.payload is not None and not ctx.payload.figures:
                _quarantine(ctx, "cloud", "no_figures_extracted", settings)

        if dry_run:
            for ctx in ctxs:
                if not ctx.active or ctx.redacted is None:
                    continue
                outbound = build_user_message(ctx.redacted, tasks)
                local_input_tokens = len(outbound) // 4
                estimate = (
                    local_input_tokens * PRICE_PER_MTOK["input"]
                    + settings.cloud.max_tokens * PRICE_PER_MTOK["output"]
                ) / 1_000_000
                console.print(f"[bold]{ctx.path.name}[/bold]")
                console.print_json(
                    json.dumps(outbound_payload_dict(ctx.redacted), ensure_ascii=False)
                )
                console.print(
                    f"Local cost preview: input≈{local_input_tokens} tokens, "
                    f"ceiling≈${estimate:.6f}"
                )
            progress.update(cloud_progress, completed=total_files)
            progress.update(review_progress, completed=total_files)
            progress.update(excel_progress, completed=total_files)
            _audit_all(ctxs, settings, run_id, audit)
            return [ctx.result for ctx in ctxs]

        if no_cloud:
            for ctx in ctxs:
                if not ctx.active or ctx.redacted is None:
                    continue
                try:
                    _write_redacted_payload(ctx, settings)
                except Exception as error:
                    if _is_disk_full(error):
                        raise
                    _mark_failed(ctx, error, "payload_write_failed")
            progress.update(cloud_progress, completed=total_files)
            progress.update(review_progress, completed=total_files)
            progress.update(excel_progress, completed=total_files)
            _audit_all(ctxs, settings, run_id, audit)
            return [ctx.result for ctx in ctxs]

        # ------------------------------- Stage D: cloud cache / calls
        cloud_misses: list[_FileCtx] = []
        for ctx in ctxs:
            if not ctx.active or ctx.redacted is None or ctx.payload is None:
                continue
            payload_sha, request_sha = _payload_sha256(ctx.redacted, settings, tasks)
            ctx.result.payload_sha256 = payload_sha
            ctx.result.request_sha256 = request_sha
            cached = _read_cloud_cache(settings, request_sha, tasks)
            if cached is not None:
                ctx.analysis, ctx.result.payload_tokens = cached
                ctx.result.stages_run.append("cloud")
            else:
                cloud_misses.append(ctx)

        progress.update(
            cloud_progress,
            completed=max(0, total_files - len(cloud_misses)),
        )

        cloud_client: ClaudeClient | None = None
        if cloud_misses:
            if settings.cloud.api_key is None:
                _environment_abort(
                    ctxs, settings, run_id, audit, "api_key_missing"
                )
                raise PipelineEnvironmentError(
                    "ANTHROPIC_API_KEY is not set; use --no-cloud or configure the key"
                )
            try:
                cloud_client = ClaudeClient(settings)
            except Exception as error:
                _environment_abort(
                    ctxs, settings, run_id, audit, "cloud_client_initialization"
                )
                raise PipelineEnvironmentError(
                    "cloud client initialization failed"
                ) from error

        if batch and cloud_misses and cloud_client is not None:
            jobs: list[tuple[str, RedactedPayload, list[str]]] = []
            by_custom_id: dict[str, _FileCtx] = {}
            for ctx in cloud_misses:
                custom_id = f"{ctx.sha256[:12]}-{ctx.result.payload_sha256[:12]}"
                jobs.append((custom_id, ctx.redacted, tasks))
                by_custom_id[custom_id] = ctx
            batch_id: str | None = None
            try:
                batch_id = cloud_client.analyze_batch(jobs)
            except BudgetExceededError:
                for custom_id, ctx in by_custom_id.items():
                    ctx.result.payload_tokens = (
                        cloud_client.last_batch_payload_tokens.get(custom_id, 0)
                    )
                    _write_redacted_payload(ctx, settings)
                    ctx.result.status = "budget_stopped"
                    ctx.result.error = "BudgetExceededError: monthly_budget_exceeded"
            except Exception as error:
                if _is_disk_full(error):
                    raise
                for custom_id, ctx in by_custom_id.items():
                    ctx.result.payload_tokens = (
                        cloud_client.last_batch_payload_tokens.get(custom_id, 0)
                    )
                    _mark_failed(ctx, error, "batch_submit_failed")
            if batch_id is not None:
                manifest_jobs: dict[str, dict[str, Any]] = {}
                for custom_id, ctx in by_custom_id.items():
                    ctx.result.status = "batch_pending"
                    ctx.result.stages_run.append("cloud")
                    ctx.result.payload_tokens = cloud_client.last_batch_payload_tokens.get(
                        custom_id, 0
                    )
                    manifest_jobs[custom_id] = {
                        "input_path": str(ctx.path.resolve()),
                        "payload_sha": ctx.result.payload_sha256,
                        "request_sha": ctx.result.request_sha256,
                        "doc_sha": ctx.sha256,
                        "tasks": tasks,
                        "payload_tokens": ctx.result.payload_tokens,
                        "raw_tokens_est": ctx.result.raw_tokens_est,
                    }
                manifest = {
                    "batch_id": batch_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": settings.cloud.model,
                    "prompt_version": CLOUD_PROMPT_VERSION,
                    "reservation_usd": cloud_client.last_batch_reservation_usd,
                    "jobs": manifest_jobs,
                    "settled": False,
                }
                _atomic_json(
                    settings.paths.batches_dir / f"{batch_id}.manifest.json",
                    manifest,
                    pretty=True,
                )
                console.print(batch_id)
            progress.update(cloud_progress, completed=total_files)

        elif cloud_misses and cloud_client is not None:
            budget_stopped = False
            for ctx in cloud_misses:
                progress.update(
                    cloud_progress,
                    description=f"[4/6] Claude {ctx.path.name}",
                )
                started = time.monotonic()
                if budget_stopped:
                    _write_redacted_payload(ctx, settings)
                    ctx.result.status = "budget_stopped"
                    ctx.result.error = "BudgetExceededError: monthly_budget_exceeded"
                    progress.advance(cloud_progress)
                    continue
                try:
                    analysis, usage = cloud_client.analyze(
                        ctx.redacted,
                        tasks,
                        file_sha12=ctx.sha256[:12],
                        payload_sha=ctx.result.payload_sha256,
                    )
                    ctx.analysis = analysis
                    ctx.result.payload_tokens = int(
                        getattr(cloud_client, "last_payload_tokens", 0)
                    )
                    ctx.result.token_usage = _usage_from_sdk(
                        getattr(cloud_client, "last_aggregate_usage", None) or usage,
                        cloud_client.last_cost_usd,
                    )
                    ctx.result.stages_run.append("cloud")
                    _write_cloud_cache(
                        settings,
                        ctx.result.request_sha256,
                        analysis,
                        ctx.result.payload_tokens,
                        tasks,
                    )
                except BudgetExceededError as error:
                    budget_stopped = True
                    _write_redacted_payload(ctx, settings)
                    ctx.result.status = "budget_stopped"
                    ctx.result.error = "BudgetExceededError: monthly_budget_exceeded"
                    ctx.result.payload_tokens = int(
                        getattr(cloud_client, "last_payload_tokens", 0)
                    )
                    ctx.result.token_usage = _usage_from_sdk(
                        getattr(cloud_client, "last_aggregate_usage", None) or {},
                        float(getattr(error, "cost_usd", 0.0)),
                    )
                except CloudRefusalError as error:
                    _quarantine(ctx, "cloud", "cloud_refusal", settings)
                    ctx.result.status = "refused"
                    ctx.result.error = "CloudRefusalError: cloud_refusal"
                    ctx.result.payload_tokens = int(
                        getattr(cloud_client, "last_payload_tokens", 0)
                    )
                    ctx.result.token_usage = _usage_from_sdk(
                        getattr(cloud_client, "last_aggregate_usage", None) or {},
                        error.cost_usd,
                    )
                except Exception as error:
                    if _is_disk_full(error):
                        raise
                    _mark_failed(ctx, error, "cloud_stage_failed")
                    ctx.result.payload_tokens = cloud_client.last_payload_tokens
                    ctx.result.token_usage = _usage_from_sdk(
                        getattr(cloud_client, "last_aggregate_usage", None) or {},
                        float(
                            getattr(
                                error,
                                "cost_usd",
                                getattr(cloud_client, "last_cost_usd", 0.0),
                            )
                        ),
                    )
                finally:
                    ctx.result.timings["cloud"] = time.monotonic() - started
                    progress.advance(cloud_progress)

        progress.update(
            cloud_progress,
            description="[4/6] Claude",
            completed=total_files,
        )

        # ------------------------------- Stage E: independent OpenAI review
        reviewer: OpenAIReviewer | None = None
        for ctx in ctxs:
            progress.update(
                review_progress,
                description=f"[5/6] OpenAI review {ctx.path.name}",
            )
            if (
                not settings.review.enabled
                or not ctx.active
                or ctx.analysis is None
                or ctx.redacted is None
            ):
                progress.advance(review_progress)
                continue
            started = time.monotonic()
            original = ctx.analysis
            try:
                review_sha = _review_request_sha256(
                    ctx.redacted, original, settings, tasks
                )
                ctx.result.review_request_sha256 = review_sha
                reviewed = _read_review_cache(settings, review_sha, tasks)
                cache_hit = reviewed is not None
                if reviewed is None:
                    if settings.review.api_key is None:
                        _environment_abort(
                            ctxs,
                            settings,
                            run_id,
                            audit,
                            "openai_api_key_missing",
                        )
                        raise PipelineEnvironmentError(
                            "OPENAI_API_KEY is not set; add it to .env.local or "
                            "set review.enabled=false"
                        )
                    if reviewer is None:
                        try:
                            reviewer = OpenAIReviewer(settings)
                        except Exception as error:
                            _environment_abort(
                                ctxs,
                                settings,
                                run_id,
                                audit,
                                "review_client_initialization",
                            )
                            raise PipelineEnvironmentError(
                                "OpenAI review client initialization failed"
                            ) from error
                    reviewed, review_usage = reviewer.review(
                        ctx.redacted, tasks, original
                    )
                    validate_peer_review(reviewed, original, ctx.redacted)
                    ctx.result.token_usage = _merge_token_usage(
                        ctx.result.token_usage, review_usage
                    )
                    _write_review_cache(settings, review_sha, reviewed, tasks)
                _apply_review_result(
                    ctx,
                    reviewed,
                    original,
                    settings,
                    cache_hit=cache_hit,
                )
            except PipelineEnvironmentError:
                raise
            except PeerReviewRefusalError:
                _quarantine(ctx, "review", "openai_peer_review_refusal", settings)
                ctx.result.status = "refused"
                ctx.result.error = "PeerReviewRefusalError: openai_peer_review_refusal"
            except Exception as error:
                if _is_disk_full(error):
                    raise
                _mark_failed(ctx, error, "openai_peer_review_failed")
            finally:
                ctx.result.timings["review"] = time.monotonic() - started
                progress.advance(review_progress)

        progress.update(
            review_progress,
            description="[5/6] OpenAI review",
            completed=total_files,
        )

        # ------------------------------- Stage F: workbook then audit
        for ctx in ctxs:
            progress.update(
                excel_progress,
                description=f"[6/6] Excel {ctx.path.name}",
            )
            if not ctx.active or ctx.analysis is None or ctx.redacted is None:
                progress.advance(excel_progress)
                continue
            started = time.monotonic()
            try:
                destination = settings.paths.outputs / output_filename(
                    ctx.path.stem, ctx.sha256[:12]
                )
                write_workbook(ctx.analysis, ctx.redacted.payload, destination)
                ctx.result.artifacts["xlsx"] = destination.resolve()
                ctx.result.stages_run.append("excel")
            except Exception as error:
                if _is_disk_full(error):
                    raise
                _mark_failed(ctx, error, "workbook_write_failed")
            finally:
                ctx.result.timings["output"] = time.monotonic() - started
                progress.advance(excel_progress)

        progress.update(
            excel_progress,
            description="[6/6] Excel",
            completed=total_files,
        )

        _audit_all(ctxs, settings, run_id, audit)
        return [ctx.result for ctx in ctxs]


def _load_manifest_payload(
    settings: Settings, job: dict[str, Any]
) -> tuple[ExtractionPayload, RedactedPayload]:
    cache_path = settings.paths.bouncer_cache / f"{job['doc_sha']}.payload.json"
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    payload = ExtractionPayload.model_validate(data["payload"])
    redacted = Redactor(settings).redact_payload(payload)
    payload_sha, _ = _payload_sha256(redacted, settings, job["tasks"])
    if payload_sha != job["payload_sha"]:
        raise ValueError("batch payload hash no longer matches its submitted request")
    return payload, redacted


def _message_text(message: Any) -> str:
    content = _field(message, "content", [])
    pieces: list[str] = []
    for block in content:
        block_type = _field(block, "type")
        if block_type == "text":
            pieces.append(str(_field(block, "text", "")))
    return "".join(pieces)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def poll_batch(
    batch_id: str,
    settings: Settings,
    *,
    console: Console | None = None,
    poll_interval_s: float = 5.0,
) -> list[PipelineResult]:
    """Collect and dispatch an ended Anthropic batch one result at a time."""
    console = console or Console()
    manifest_path = settings.paths.batches_dir / f"{batch_id}.manifest.json"
    if not manifest_path.is_file():
        raise PipelineEnvironmentError(f"batch manifest not found: {manifest_path}")
    run_id = uuid.uuid4().hex
    with pipeline_run_lock(settings), _run_logging(settings, run_id):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("settled"):
            statuses = manifest.get("terminal_statuses", {})
            return [
                PipelineResult(
                    input_path=Path(manifest["jobs"][custom_id]["input_path"]),
                    content_sha256=manifest["jobs"][custom_id]["doc_sha"],
                    status=status,
                )
                for custom_id, status in statuses.items()
            ]
        if settings.cloud.api_key is None:
            raise PipelineEnvironmentError("ANTHROPIC_API_KEY is required for batch-poll")
        client = ClaudeClient(settings)
        reviewer: OpenAIReviewer | None = None
        while True:
            state = client.retrieve_batch(batch_id)
            status = _field(state, "processing_status")
            if status == "ended":
                break
            console.print(f"Batch {batch_id}: {status or 'in_progress'}")
            time.sleep(poll_interval_s)
        responses = client.batch_results(batch_id)
        response_by_id: dict[str, Any] = {}
        for item in responses:
            custom_id = _field(item, "custom_id")
            if isinstance(custom_id, str):
                response_by_id[custom_id] = item
        audit = AuditLog(settings.paths.logs)
        ctxs: list[_FileCtx] = []
        terminal_statuses: dict[str, PipelineStatus] = {}
        for custom_id, job in manifest["jobs"].items():
            path = Path(job["input_path"])
            result = PipelineResult(path, job["doc_sha"])
            result.raw_tokens_est = int(job.get("raw_tokens_est", 0))
            result.payload_tokens = int(job.get("payload_tokens", 0))
            result.payload_sha256 = job["payload_sha"]
            result.request_sha256 = job["request_sha"]
            ctx = _FileCtx(path, job["doc_sha"], result)
            ctxs.append(ctx)
            try:
                ctx.payload, ctx.redacted = _load_manifest_payload(settings, job)
                ctx.result.redaction_hits = _redaction_counts(ctx.redacted)
            except Exception as error:
                _mark_failed(ctx, error, "batch_payload_cache_unavailable")
                terminal_statuses[custom_id] = ctx.result.status
                continue
            item = response_by_id.get(custom_id)
            if item is None:
                _mark_failed(ctx, "BatchResult", "batch_result_missing")
                terminal_statuses[custom_id] = ctx.result.status
                continue
            try:
                batch_result = _field(item, "result")
                result_type = _field(batch_result, "type")
                if result_type == "succeeded":
                    message = _field(batch_result, "message")
                    usage = _field(message, "usage")
                    if message is None or usage is None:
                        raise ValueError("succeeded batch result is missing message usage")
                    with client.ledger.reserve():
                        cost = client.ledger.record(
                            usage,
                            file_sha12=ctx.sha256[:12],
                            payload_sha12=job["payload_sha"][:12],
                            batch=True,
                        )
                    ctx.result.token_usage = _usage_from_sdk(usage, cost)
                    stop_reason = _field(message, "stop_reason")
                    if stop_reason == "end_turn":
                        ctx.analysis = AnalysisResult.model_validate_json(_message_text(message))
                        ctx.result.stages_run.append("cloud")
                        _write_cloud_cache(
                            settings,
                            job["request_sha"],
                            ctx.analysis,
                            ctx.result.payload_tokens,
                            job["tasks"],
                        )
                        if settings.review.enabled:
                            original = ctx.analysis
                            review_sha = _review_request_sha256(
                                ctx.redacted, original, settings, job["tasks"]
                            )
                            ctx.result.review_request_sha256 = review_sha
                            reviewed = _read_review_cache(
                                settings, review_sha, job["tasks"]
                            )
                            cache_hit = reviewed is not None
                            if reviewed is None:
                                if settings.review.api_key is None:
                                    raise PipelineEnvironmentError(
                                        "OPENAI_API_KEY is required to peer-review "
                                        "completed batch results"
                                    )
                                if reviewer is None:
                                    try:
                                        reviewer = OpenAIReviewer(settings)
                                    except Exception as error:
                                        raise PipelineEnvironmentError(
                                            "OpenAI review client initialization failed"
                                        ) from error
                                reviewed, review_usage = reviewer.review(
                                    ctx.redacted, job["tasks"], original
                                )
                                validate_peer_review(
                                    reviewed, original, ctx.redacted
                                )
                                ctx.result.token_usage = _merge_token_usage(
                                    ctx.result.token_usage, review_usage
                                )
                                _write_review_cache(
                                    settings, review_sha, reviewed, job["tasks"]
                                )
                            _apply_review_result(
                                ctx,
                                reviewed,
                                original,
                                settings,
                                cache_hit=cache_hit,
                            )
                        if ctx.active:
                            destination = settings.paths.outputs / output_filename(
                                path.stem, ctx.sha256[:12]
                            )
                            write_workbook(
                                ctx.analysis, ctx.redacted.payload, destination
                            )
                            ctx.result.artifacts["xlsx"] = destination.resolve()
                            ctx.result.stages_run.append("excel")
                    elif stop_reason == "refusal":
                        _quarantine(ctx, "cloud", "cloud_refusal", settings)
                        ctx.result.status = "refused"
                        ctx.result.error = "CloudRefusalError: cloud_refusal"
                    elif stop_reason == "max_tokens":
                        _mark_failed(ctx, "BatchMaxTokens", "batch_max_tokens")
                    else:
                        _mark_failed(
                            ctx, "BatchStopReason", "batch_unexpected_stop_reason"
                        )
                elif result_type == "errored":
                    _mark_failed(ctx, "BatchErrored", "batch_request_errored")
                elif result_type in {"expired", "canceled"}:
                    _mark_failed(
                        ctx, "BatchTerminal", f"batch_{result_type}_resubmit"
                    )
                else:
                    _mark_failed(ctx, "BatchResult", "batch_unknown_result")
            except PeerReviewRefusalError:
                _quarantine(
                    ctx, "review", "openai_peer_review_refusal", settings
                )
                ctx.result.status = "refused"
                ctx.result.error = (
                    "PeerReviewRefusalError: openai_peer_review_refusal"
                )
            except PipelineEnvironmentError:
                raise
            except OSError as error:
                if _is_disk_full(error):
                    raise
                _mark_failed(ctx, error, "batch_result_dispatch_failed")
            except Exception as error:
                _mark_failed(ctx, error, "batch_result_dispatch_failed")
            finally:
                terminal_statuses[custom_id] = ctx.result.status

        with client.ledger.reserve():
            client.ledger.record_amount(
                -float(manifest.get("reservation_usd", 0.0)),
                kind="batch_settlement",
                batch_id=batch_id,
            )
        _audit_all(ctxs, settings, run_id, audit)
        manifest["settled"] = True
        manifest["settled_at"] = datetime.now(timezone.utc).isoformat()
        manifest["terminal_statuses"] = terminal_statuses
        _atomic_json(manifest_path, manifest, pretty=True)
        return [ctx.result for ctx in ctxs]
