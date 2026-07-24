from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

from kimi_cli.wiki.models import SourceRef, WikiPage


def _page() -> WikiPage:
    created = datetime.fromisoformat("2026-07-24T12:00:00+08:00")
    return WikiPage(
        logical_path="concepts/atomic-writes.md",
        title="Atomic writes",
        created=created,
        updated=created,
        tags=["atomic"],
        sources=[
            SourceRef(
                kind="workspace-file",
                workspace_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
                path="docs/source.md",
                content_hash="sha256:" + "a" * 64,
            )
        ],
        revision=1,
        body="Atomic replacement protects readers from incomplete files.",
    )


def test_corrupt_database_is_quarantined_then_rebuilt_from_pages(tmp_path: Path) -> None:
    from kimi_cli.wiki.search import WikiSearchIndex

    database = tmp_path / "search.sqlite3"
    database.write_bytes(b"not sqlite")

    index = WikiSearchIndex.open(database, wal=False)
    try:
        index.rebuild([_page()])
        assert index.search("atomic", 5)[0].logical_path == "concepts/atomic-writes.md"
    finally:
        index.close()

    assert database.exists()
    assert not list(tmp_path.glob("search.sqlite3.corrupt-*"))


def test_stale_cache_schema_is_replaced_without_touching_markdown(tmp_path: Path) -> None:
    from kimi_cli.wiki.search import WikiSearchIndex

    database = tmp_path / "search.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE pages (obsolete TEXT)")
    connection.commit()
    connection.close()

    with WikiSearchIndex.open(database, wal=False) as index:
        index.rebuild([_page()])
        assert index.search("incomplete", 5)[0].logical_path == "concepts/atomic-writes.md"


def test_open_honors_wal_configuration(tmp_path: Path) -> None:
    from kimi_cli.wiki.search import WikiSearchIndex

    database = tmp_path / "cache.sqlite3"
    with WikiSearchIndex.open(database, wal=True) as index:
        assert index._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    with WikiSearchIndex.open(database, wal=False) as index:
        assert index._connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_bounded_markdown_search_never_needs_sqlite_and_limits_results() -> None:
    from kimi_cli.wiki.search import bounded_markdown_search

    page = _page()

    result = bounded_markdown_search([page], "replacement", 999)

    assert [entry.logical_path for entry in result] == [page.logical_path]
    assert len(result) == 1
