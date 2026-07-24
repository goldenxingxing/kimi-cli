from __future__ import annotations

import pytest

from kimi_cli.wiki.context import (
    WIKI_BLOCK_END,
    WIKI_BLOCK_START,
    WIKI_PROMPT_MAX_BYTES,
    build_wiki_context,
    refresh_wiki_prompt_block,
    render_compact_index,
)

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


def test_refresh_inserts_wiki_block_into_pre_upgrade_prompt() -> None:
    old_prompt = "System prefix.\n\n# Skills\n\nKeep this user configuration."
    context = build_wiki_context(CHINESE_INDEX)

    refreshed = refresh_wiki_prompt_block(old_prompt, context)

    assert refreshed.startswith("System prefix.")
    assert refreshed.endswith("Keep this user configuration.")
    assert refreshed.count(WIKI_BLOCK_START) == 1
    assert refreshed.count(WIKI_BLOCK_END) == 1
    assert context in refreshed
    prefix, managed_and_suffix = refreshed.split(WIKI_BLOCK_START, 1)
    _managed, suffix = managed_and_suffix.split(WIKI_BLOCK_END, 1)
    assert prefix == "System prefix.\n\n"
    assert suffix == "\n\n# Skills\n\nKeep this user configuration."


def test_refresh_replaces_old_block_without_changing_other_prompt_content() -> None:
    old = f"Before.\n\n{WIKI_BLOCK_START}\n# Global Wiki\nstale index\n{WIKI_BLOCK_END}\n\nAfter."
    current = build_wiki_context(CHINESE_INDEX)

    refreshed = refresh_wiki_prompt_block(old, current)

    assert refreshed == (
        f"Before.\n\n{WIKI_BLOCK_START}\n# Global Wiki\n{current}\n{WIKI_BLOCK_END}\n\nAfter."
    )
    assert "stale index" not in refreshed


def test_refresh_collapses_duplicate_blocks_and_removes_stale_block_when_unavailable() -> None:
    block = f"{WIKI_BLOCK_START}\n# Global Wiki\nold\n{WIKI_BLOCK_END}"
    duplicated = f"Before.\n\n{block}\n\nMiddle.\n\n{block}\n\nAfter."

    refreshed = refresh_wiki_prompt_block(duplicated, build_wiki_context(CHINESE_INDEX))
    unavailable = refresh_wiki_prompt_block(duplicated, "")

    assert refreshed.count(WIKI_BLOCK_START) == 1
    assert refreshed.count(WIKI_BLOCK_END) == 1
    assert "Before." in refreshed and "Middle." in refreshed and "After." in refreshed
    assert WIKI_BLOCK_START not in unavailable
    assert "Before." in unavailable and "Middle." in unavailable and "After." in unavailable


def test_wiki_prompt_block_preserves_utf8_within_total_budget() -> None:
    large_index = "# Wiki Index\n\n" + "\n".join(
        f"- [[concepts/中文-{number}]] — 并发写入与故障恢复" for number in range(300)
    )

    context = build_wiki_context(large_index, hints=("中文-299",))
    prompt = refresh_wiki_prompt_block("Existing prompt.", context)
    block = prompt[prompt.index(WIKI_BLOCK_START) : prompt.index(WIKI_BLOCK_END)] + WIKI_BLOCK_END

    assert len(block.encode("utf-8")) <= WIKI_PROMPT_MAX_BYTES
    assert "中文-299" in block
    block.encode("utf-8").decode("utf-8")
