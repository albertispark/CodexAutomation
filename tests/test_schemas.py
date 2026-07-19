from __future__ import annotations

import json
from pathlib import Path

from pipeline.extraction.schemas import (
    SCHEMA_PATH,
    bouncer_schema,
    payload_json_schema,
)


def test_checked_in_schema_is_exact_export() -> None:
    assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) == payload_json_schema()


def test_bouncer_schema_has_no_figure_id() -> None:
    encoded = json.dumps(bouncer_schema(), sort_keys=True)
    assert "figure_id" not in encoded
