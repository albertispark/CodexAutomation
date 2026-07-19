"""Byte-stable cloud user-message construction and analysis task registry."""
from __future__ import annotations

import json

from pipeline.extraction.redactor import RedactedPayload

CLOUD_PROMPT_VERSION: str = "1"
DEFAULT_TASKS: list[str] = ["metrics", "variance", "adjustments"]
TASK_DESCRIPTIONS: dict[str, str] = {
    "metrics": "Compute all standard metrics derivable from the provided figures.",
    "variance": "Compute period-over-period variance for every metric present in >=2 periods.",
    "adjustments": "Propose normalization adjustments for one-time/non-recurring items.",
}


def outbound_payload_dict(redacted: RedactedPayload) -> dict:
    """Return the exact cloud-eligible payload dictionary."""
    if not isinstance(redacted, RedactedPayload):
        raise TypeError("cloud message construction requires RedactedPayload")
    data = redacted.payload.model_dump(mode="json")
    for figure in data.get("figures", []):
        figure.pop("verbatim_context", None)
    return data


def build_user_message(redacted: RedactedPayload, tasks: list[str]) -> str:
    """Build the only document-derived outbound text, without source snippets."""
    if not isinstance(redacted, RedactedPayload):
        raise TypeError("build_user_message requires RedactedPayload")
    unknown = [task for task in tasks if task not in TASK_DESCRIPTIONS]
    if unknown:
        raise ValueError(
            f"Unknown analysis task(s): {', '.join(unknown)}; valid tasks: "
            f"{', '.join(TASK_DESCRIPTIONS)}"
        )
    task_lines = ["REQUESTED ANALYSIS TASKS:"] + [
        f"- {task}: {TASK_DESCRIPTIONS[task]}" for task in tasks
    ]
    payload_json = json.dumps(
        outbound_payload_dict(redacted),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "\n".join([*task_lines, "", "PAYLOAD:", payload_json])
