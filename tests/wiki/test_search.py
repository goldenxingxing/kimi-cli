from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

from kimi_cli.wiki.models import SourceRef, WikiPage
from kimi_cli.wiki.schema import content_hash, render_page


def _page(
    logical_path: str,
    title: str,
    tags: list[str],
    body: str,
    *,
    revision: int = 1,
) -> WikiPage:
    created = datetime.fromisoformat("2026-07-24T12:00:00+08:00")
    return WikiPage(
        logical_path=logical_path,
        title=title,
        created=created,
        updated=created,
        tags=tags,
        sources=[
            SourceRef(
                kind="workspace-file",
                workspace_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
                path="docs/source.md",
                content_hash="sha256:" + "a" * 64,
            )
        ],
        revision=revision,
        body=body,
    )


@pytest.fixture
def pages() -> list[WikiPage]:
    return [
        _page(
            "concepts/atomic-writes.md",
            "原子写入与锁",
            ["并发", "锁"],
            "并发写入需要原子替换，避免读者看见半成品。",
        ),
        _page(
            "concepts/cache.md",
            "Search cache",
            ["sqlite", "search"],
            "SQLite FTS5 keeps a disposable cache for fast English search.",
        ),
    ]


@pytest.fixture
def index(tmp_path: Path):
    from kimi_cli.wiki.search import WikiSearchIndex

    result = WikiSearchIndex.open(tmp_path / "search.sqlite3", wal=True)
    try:
        yield result
    finally:
        result.close()


def test_trigram_finds_chinese_substring(index, pages: list[WikiPage]) -> None:
    index.rebuild(pages)

    assert index.search("并发写入", 5)[0].logical_path == "concepts/atomic-writes.md"


def test_english_search_returns_bounded_deterministic_results(index, pages: list[WikiPage]) -> None:
    index.rebuild(pages)

    results = index.search("search", 100)

    assert [result.logical_path for result in results] == ["concepts/cache.md"]
    assert results[0].revision == 1
    assert "search" in results[0].snippet.casefold()
    assert len(index.search("search", 0)) == 1


def test_short_query_uses_title_tag_fallback(index, pages: list[WikiPage]) -> None:
    index.rebuild(pages)

    assert {result.logical_path for result in index.search("锁", 5)} == {
        "concepts/atomic-writes.md"
    }


def test_sync_replaces_changed_hash_and_removes_deleted_rows(index, pages: list[WikiPage]) -> None:
    index.rebuild(pages)
    changed = _page(
        "concepts/cache.md",
        "Search cache",
        ["sqlite", "search"],
        "The durable index is Markdown, not an old cache row.",
        revision=2,
    )

    index.sync([changed])

    assert not index.search("atomic", 5)
    result = index.search("durable", 5)[0]
    assert result.logical_path == changed.logical_path
    assert result.revision == 2
    assert result.content_hash == content_hash(render_page(changed).encode("utf-8"))


def test_revision_cas_never_allows_older_snapshot_to_replace_newer_cache(
    tmp_path: Path, pages: list[WikiPage]
) -> None:
    from kimi_cli.wiki.search import WikiSearchIndex

    database = tmp_path / "search.sqlite3"
    older = WikiSearchIndex.open(database, wal=False)
    newer = WikiSearchIndex.open(database, wal=False)
    revised = pages[0].model_copy(
        update={"revision": 2, "body": "Newest authoritative cache content.\n"}
    )
    try:
        assert newer.sync([revised], revision=2) is True
        assert older.sync(pages, revision=1) is False
        result = older.search("Newest authoritative", 5)
        assert result[0].revision == 2
    finally:
        older.close()
        newer.close()


def test_non_trigram_build_uses_title_tag_then_escaped_like(
    tmp_path: Path, pages: list[WikiPage], monkeypatch: pytest.MonkeyPatch
) -> None:
    import kimi_cli.wiki.search as search

    monkeypatch.setattr(search, "_create_fts", lambda connection: False)
    index = search.WikiSearchIndex.open(tmp_path / "search.sqlite3", wal=False)
    try:
        index.rebuild(pages)

        assert not index.trigram
        assert index.search("锁", 5)[0].logical_path == "concepts/atomic-writes.md"
        assert index.search("disposable", 5)[0].logical_path == "concepts/cache.md"
        assert not index.search("%", 5)
    finally:
        index.close()


def test_fts_query_failure_uses_bounded_markdown_fallback(
    index, pages: list[WikiPage], monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    index.rebuild(pages)

    def fail_fts(query: str, limit: int):
        raise sqlite3.DatabaseError("cache unavailable")

    monkeypatch.setattr(index, "_fts_search", fail_fts)

    assert index.search("disposable", 5)[0].logical_path == "concepts/cache.md"
