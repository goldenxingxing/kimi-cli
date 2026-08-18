"""Retrieval over stored memory.

The opening snapshot lists facts one line each — enough to recognise something
already anticipated, not enough to find something half-remembered.
"""

from __future__ import annotations

from pathlib import Path

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.search import MemorySearchIndex


def _entries(*bodies: str) -> list[MemoryEntry]:
    return [
        MemoryEntry(kind="project", scope="persistent", content=b, key=f"p/{i}")
        for i, b in enumerate(bodies)
    ]


def _index(tmp_path: Path) -> tuple[MemorySearchIndex, Path]:
    source = tmp_path / "persistent.jsonl"
    source.write_text("seed", encoding="utf-8")
    return MemorySearchIndex(tmp_path / "search.sqlite3", source), source


def test_it_finds_a_phrase_inside_chinese_without_spaces(tmp_path: Path) -> None:
    """The reason the index is trigram-tokenised.

    Chinese is written without spaces, so the default tokeniser makes a whole
    sentence one token and every phrase inside it is unfindable.
    """
    index, _ = _index(tmp_path)
    hits = index.search("仓库路径", _entries("acls 项目：真实仓库路径为 /Users/x/acls"))
    assert [h.handle for h in hits] == ["p/0"]


def test_a_two_character_query_falls_back_rather_than_missing(tmp_path: Path) -> None:
    """Two characters is a whole word in Chinese and below what trigram indexes."""
    index, _ = _index(tmp_path)
    hits = index.search("邮箱", _entries("126 邮箱已接入"))
    assert [h.handle for h in hits] == ["p/0"]


def test_punctuation_in_a_query_is_not_read_as_syntax(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)
    entries = _entries("the file core.py lives under src/")
    for query in ("core.py", 'a "quoted" thing', "src/"):
        index.search(query, entries)  # must not raise


def test_an_empty_or_whitespace_query_returns_nothing(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)
    assert index.search("", _entries("something")) == []
    assert index.search("   ", _entries("something")) == []


def test_the_index_refreshes_when_the_store_changes(tmp_path: Path) -> None:
    """It is a cache over a file the user can edit by hand."""
    index, source = _index(tmp_path)
    assert index.search("alpha", _entries("alpha fact")) != []

    source.write_text("changed", encoding="utf-8")
    hits = index.search("bravo", _entries("bravo fact"))
    assert [h.handle for h in hits] == ["p/0"]


def test_a_corrupt_index_degrades_to_no_results(tmp_path: Path) -> None:
    """Search is an aid; an unusable index must not break the tool call."""
    db = tmp_path / "search.sqlite3"
    db.write_bytes(b"this is not a database")
    source = tmp_path / "persistent.jsonl"
    source.write_text("seed", encoding="utf-8")

    assert MemorySearchIndex(db, source).search("anything", _entries("x y z")) == []


def test_results_are_capped(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)
    hits = index.search("common", _entries(*[f"common token {i}" for i in range(40)]), limit=5)
    assert len(hits) == 5
