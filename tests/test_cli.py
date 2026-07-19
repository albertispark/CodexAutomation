from pathlib import Path

import pytest

from pipeline.cli import _expand_paths, _parse_tasks


def test_parse_tasks_validates_registry() -> None:
    assert _parse_tasks("metrics, variance") == ["metrics", "variance"]
    with pytest.raises(Exception, match="valid names"):
        _parse_tasks("magic")


def test_expand_paths_uses_router_extensions(settings, tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("a,b\n1,2\n")
    (tmp_path / "ignore.txt").write_text("x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.pdf").write_bytes(b"not important for expansion")
    expanded = _expand_paths([tmp_path], settings)
    assert [path.name for path in expanded] == ["a.csv", "b.pdf"]
