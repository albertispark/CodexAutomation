"""Token-aware, table-safe chunking with hard document boundaries."""
from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.config import IndexConfig
from pipeline.ingestion.router import IngestedFile

CHARS_PER_TOKEN: int = 4
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
SENTENCE_END_RE = re.compile(r"[.!?](?:[\"')\]]*)\s+")
ROW_SPLIT_MARKER = "[ROW SPLIT]"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_sha: str
    source_path: str
    origin: str
    page_start: int
    page_end: int
    text: str
    token_estimate: int
    contains_table: bool


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _is_separator(line: str) -> bool:
    return bool(SEPARATOR_RE.fullmatch(line))


def split_blocks(page_text: str) -> list[tuple[str, bool]]:
    """Split prose on blank lines and retain table-row runs as atomic blocks."""
    blocks: list[tuple[str, bool]] = []
    prose: list[str] = []
    table: list[str] = []

    def flush_prose() -> None:
        if prose:
            text = "\n".join(prose).strip("\n")
            if text:
                blocks.append((text, False))
            prose.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("\n".join(table), True))
            table.clear()

    for line in page_text.splitlines():
        if TABLE_ROW_RE.fullmatch(line):
            flush_prose()
            table.append(line)
        else:
            flush_table()
            if line.strip():
                prose.append(line)
            else:
                flush_prose()
    flush_table()
    flush_prose()
    return blocks


def _table_prefix(rows: list[str]) -> tuple[list[str], list[str]]:
    if len(rows) >= 2 and _is_separator(rows[1]):
        return rows[:2], rows[2:]
    if rows and not any(char.isdigit() for char in rows[0]):
        return rows[:1], rows[1:]
    return [], rows


def _split_large_row(row: str, available_chars: int) -> list[str]:
    """Last-resort row split while keeping every emitted line table-shaped."""
    inner = row.strip().strip("|").strip()
    usable = max(8, available_chars - len(ROW_SPLIT_MARKER) - 6)
    pieces: list[str] = []
    while inner:
        piece = inner[:usable]
        inner = inner[usable:]
        pieces.append(f"| {piece} {ROW_SPLIT_MARKER} |")
    return pieces or [f"| {ROW_SPLIT_MARKER} |"]


def split_oversized_table(table_block: str, max_tokens: int) -> list[str]:
    """Split tables between rows, repeating only a trustworthy header prefix."""
    if estimate_tokens(table_block) <= max_tokens:
        return [table_block]
    rows = [line for line in table_block.splitlines() if line.strip()]
    if not rows:
        return []
    prefix, data_rows = _table_prefix(rows)
    prefix_text = "\n".join(prefix)
    budget_chars = max_tokens * CHARS_PER_TOKEN
    available = max(8, budget_chars - len(prefix_text) - (1 if prefix else 0))

    expanded: list[str] = []
    for row in data_rows:
        if len(row) > available:
            expanded.extend(_split_large_row(row, available))
        else:
            expanded.append(row)

    # A header-only oversized table still needs a bounded representation.
    if not expanded:
        if estimate_tokens(prefix_text) <= max_tokens:
            return [prefix_text]
        return _split_large_row(prefix_text, budget_chars)

    fragments: list[str] = []
    current = list(prefix)
    data_count = 0
    for row in expanded:
        candidate = "\n".join([*current, row])
        if data_count and estimate_tokens(candidate) > max_tokens:
            fragments.append("\n".join(current))
            current = [*prefix, row]
            data_count = 1
        else:
            current.append(row)
            data_count += 1
    if data_count or current:
        fragments.append("\n".join(current))
    return fragments


def _split_oversized_prose(text: str, max_tokens: int) -> list[str]:
    max_chars = max_tokens * CHARS_PER_TOKEN
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        cut = window.rfind("\n", 1, max_chars + 1)
        if cut <= 0:
            sentence_ends = [match.end() for match in SENTENCE_END_RE.finditer(window)]
            cut = sentence_ends[-1] if sentence_ends else max_chars
        piece = remaining[:cut].rstrip()
        if not piece:
            piece = remaining[:max_chars]
            cut = max_chars
        pieces.append(piece)
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _prose_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    target_chars = overlap_tokens * CHARS_PER_TOKEN
    if len(text) <= target_chars:
        return text
    start = len(text) - target_chars
    # Move forward to the first whitespace boundary so a numeral is never bisected.
    while start < len(text) and not text[start].isspace():
        start += 1
    return text[start:].lstrip()


def chunk_documents(ingested: IngestedFile, cfg: IndexConfig) -> list[Chunk]:
    """Chunk every document independently, preserving provenance and table rows."""
    chunks: list[Chunk] = []
    sequence = 1

    for document in ingested.documents:
        units: list[tuple[str, bool]] = []
        for block, is_table in split_blocks(document.text):
            if is_table:
                units.extend(
                    (piece, True)
                    for piece in split_oversized_table(block, cfg.chunk_tokens)
                )
            elif estimate_tokens(block) > cfg.chunk_tokens:
                units.extend(
                    (piece, False)
                    for piece in _split_oversized_prose(block, cfg.chunk_tokens)
                )
            else:
                units.append((block, False))

        current: list[tuple[str, bool]] = []

        def emit() -> tuple[str, bool]:
            nonlocal current, sequence
            if not current:
                return "", False
            text = "\n\n".join(piece for piece, _ in current)
            contains_table = any(flag for _, flag in current)
            chunks.append(
                Chunk(
                    chunk_id=f"{ingested.doc_sha}:{sequence:06d}",
                    doc_sha=ingested.doc_sha,
                    source_path=str(ingested.source_path),
                    origin=document.origin,
                    page_start=document.page_number,
                    page_end=document.page_number,
                    text=text,
                    token_estimate=estimate_tokens(text),
                    contains_table=contains_table,
                )
            )
            sequence += 1
            trailing_text, trailing_is_table = current[-1]
            current = []
            return trailing_text, trailing_is_table

        for unit_text, is_table in units:
            if not current:
                current = [(unit_text, is_table)]
                continue
            candidate = "\n\n".join(
                [*(piece for piece, _ in current), unit_text]
            )
            if estimate_tokens(candidate) <= cfg.chunk_tokens:
                current.append((unit_text, is_table))
                continue
            trailing, trailing_is_table = emit()
            carry = "" if trailing_is_table else _prose_overlap(trailing, cfg.chunk_overlap)
            if carry:
                with_carry = f"{carry}\n\n{unit_text}"
                if estimate_tokens(with_carry) <= cfg.chunk_tokens:
                    current = [(carry, False), (unit_text, is_table)]
                else:
                    current = [(unit_text, is_table)]
            else:
                current = [(unit_text, is_table)]
        emit()
        # current and overlap intentionally reset here: Documents are hard boundaries.

    return chunks
