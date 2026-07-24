from __future__ import annotations

import pytest

from kimi_cli.wiki.context import render_compact_index

CHINESE_INDEX = """\
# Wiki Index

## Concepts
- [[concepts/普通]] — 普通知识
- [[concepts/并发]] — 并发写入
- [[concepts/恢复]] — 故障恢复
"""


def test_compact_index_preserves_utf8_entries_and_marker() -> None:
    rendered = render_compact_index(CHINESE_INDEX, max_bytes=120, max_entries=2)

    assert len(rendered.encode("utf-8")) <= 120
    assert rendered.endswith("<!-- Wiki index truncated -->")
    rendered.encode("utf-8").decode("utf-8")
    assert rendered.count("- [[") == 2


def test_compact_index_prioritizes_hinted_whole_entries() -> None:
    rendered = render_compact_index(
        CHINESE_INDEX,
        max_bytes=96,
        max_entries=1,
        hints=("并发",),
    )

    assert "[[concepts/并发]]" in rendered
    assert "[[concepts/普通]]" not in rendered
    assert "- [[concepts/并发]] — 并发写入" in rendered.splitlines()


def test_compact_index_returns_small_index_without_reformatting() -> None:
    rendered = render_compact_index(CHINESE_INDEX, max_bytes=512, max_entries=10)

    assert rendered == CHINESE_INDEX.strip()
    assert "truncated" not in rendered


@pytest.mark.parametrize(
    ("max_bytes", "max_entries"),
    [
        (0, 1),
        (20, 1),
        (120, -1),
        (True, 1),
        (120, True),
    ],
)
def test_compact_index_rejects_invalid_limits(max_bytes: int, max_entries: int) -> None:
    with pytest.raises(ValueError):
        render_compact_index(CHINESE_INDEX, max_bytes=max_bytes, max_entries=max_entries)
