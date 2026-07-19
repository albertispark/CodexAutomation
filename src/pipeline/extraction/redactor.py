"""Generic outbound PII scrub and the only producer of RedactedPayload."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from pipeline.config import Settings
from pipeline.extraction.schemas import ExtractionPayload, RedactionEvent

logger = logging.getLogger("pipeline.extraction.redactor")

BUILTIN_PATTERNS: dict[str, str] = {
    "account_number": r"\b\d{8,17}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "ein": r"\b\d{2}-\d{7}\b",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    "card_number": r"\b(?:\d{4}[ \-]?){3}\d{2,4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
}
REDACTION_TOKEN: str = "[REDACTED:{name}]"


@dataclass(frozen=True)
class RedactedPayload:
    """The only value type accepted by cloud message construction."""

    payload: ExtractionPayload
    events: list[RedactionEvent]


class Redactor:
    def __init__(self, cfg: Settings) -> None:
        self._enabled = cfg.redaction.enabled
        effective = dict(BUILTIN_PATTERNS)
        effective.update(cfg.redaction.patterns)
        # PANs are a more specific form of long digit sequence, so run that
        # classifier before the broad account-number fallback. The canonical
        # IBAN constant remains contract-identical while its built-in matcher
        # also accepts conventional groups separated by spaces (A14).
        order = ["card_number", "iban", *(
            name for name in effective if name not in {"card_number", "iban"}
        )]
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            (
                name,
                re.compile(
                    r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b"
                    if name == "iban" and pattern == BUILTIN_PATTERNS["iban"]
                    else pattern
                ),
            )
            for name in order
            if (pattern := effective[name])
        ]

    def redact_payload(self, payload: ExtractionPayload) -> RedactedPayload:
        if not isinstance(payload, ExtractionPayload):
            raise TypeError("redact_payload requires ExtractionPayload")
        scrubbed = payload.model_copy(deep=True)
        if not self._enabled:
            logger.warning("Outbound redaction is disabled by configuration")
            return RedactedPayload(scrubbed, [])
        events: list[RedactionEvent] = []
        self._walk(scrubbed, "", events)
        return RedactedPayload(scrubbed, events)

    def _walk(self, value: Any, path: str, events: list[RedactionEvent]) -> Any:
        if isinstance(value, BaseModel):
            for field_name in value.__class__.model_fields:
                if field_name == "figure_id":
                    continue
                field_value = getattr(value, field_name)
                field_path = f"{path}.{field_name}" if path else field_name
                replacement = self._walk(field_value, field_path, events)
                if replacement is not field_value:
                    setattr(value, field_name, replacement)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                item_path = f"{path}[{index}]"
                value[index] = self._walk(item, item_path, events)
            return value
        if isinstance(value, tuple):
            return tuple(
                self._walk(item, f"{path}[{index}]", events)
                for index, item in enumerate(value)
            )
        if isinstance(value, dict):
            for key, item in list(value.items()):
                item_path = f"{path}.{key}" if path else str(key)
                value[key] = self._walk(item, item_path, events)
            return value
        if isinstance(value, Enum):
            return value
        if isinstance(value, str):
            return self._redact_string(value, path, events)
        return value

    def _redact_string(
        self, text: str, field_path: str, events: list[RedactionEvent]
    ) -> str:
        output = text
        for name, pattern in self._patterns:
            def replace(match: re.Match[str], *, pattern_name: str = name) -> str:
                events.append(
                    RedactionEvent(
                        pattern_name=pattern_name,
                        field_path=field_path,
                        match_preview=match.group(0)[:4] + "...",
                    )
                )
                return REDACTION_TOKEN.format(name=pattern_name)

            output = pattern.sub(replace, output)
        return output
