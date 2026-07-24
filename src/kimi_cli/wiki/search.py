"""Disposable SQLite search cache for authoritative global Wiki Markdown.

Markdown remains the source of truth.  This module deliberately accepts parsed
pages from its caller rather than discovering files itself, so an unusable cache
can always be discarded and rebuilt without touching Wiki content.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from kimi_cli.wiki.models import WikiPage
from kimi_cli.wiki.schema import content_hash, render_page

_MAX_RESULTS = 20
_SNIPPET_LENGTH = 240
_PAGES_COLUMNS = (
    "logical_path",
    "content_hash",
    "revision",
    "title",
    "tags",
    "summary",
    "body",
)
_FTS_COLUMNS = ("logical_path", "title", "tags", "summary", "body")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A bounded, safe-to-return projection of one indexed Wiki page."""

    logical_path: str
    title: str
    summary: str
    snippet: str
    score: float
    revision: int
    content_hash: str


class WikiSearchIndex:
    """A rebuildable FTS5 cache whose rows are derived from validated pages."""

    def __init__(self, database: Path, connection: sqlite3.Connection, *, trigram: bool) -> None:
        self.database = database
        self._connection = connection
        self.trigram = trigram
        self._fts_available = _has_fts_table(connection)
        self._markdown_pages: tuple[WikiPage, ...] = ()

    @classmethod
    def open(cls, database: Path, *, wal: bool) -> WikiSearchIndex:
        """Open a cache, replacing a corrupt database with an empty disposable one."""
        path = database.expanduser()
        if path.is_symlink():
            raise ValueError("Wiki search cache must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection, trigram = _open_database(path, wal=wal)
        except sqlite3.DatabaseError as exc:
            if not _is_rebuildable_cache_error(exc):
                raise
            diagnostic = _quarantine_corrupt_database(path)
            try:
                connection, trigram = _open_database(path, wal=wal)
            except Exception:
                _remove_cache_artifacts(path)
                raise
            else:
                diagnostic.unlink(missing_ok=True)
        return cls(path, connection, trigram=trigram)

    def close(self) -> None:
        """Close the derivative cache connection."""
        self._connection.close()

    def __enter__(self) -> WikiSearchIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def rebuild(self, pages: Iterable[WikiPage]) -> None:
        """Replace every cache row with the supplied authoritative Markdown pages."""
        materialized = _validated_pages(pages)
        try:
            with self._connection:
                self._connection.execute("DELETE FROM pages")
                if self._fts_available:
                    self._connection.execute("DELETE FROM pages_fts")
                self._insert_pages(materialized)
        except sqlite3.Error:
            # The caller still has Markdown and can use bounded_markdown_search.
            self._markdown_pages = materialized
            raise
        self._markdown_pages = materialized

    def sync(self, pages: Iterable[WikiPage]) -> None:
        """Synchronize changed and deleted content-hash rows in one SQLite transaction."""
        materialized = _validated_pages(pages)
        try:
            existing = {
                row[0]: row[1]
                for row in self._connection.execute("SELECT logical_path, content_hash FROM pages")
            }
            incoming = {page.logical_path: _page_hash(page) for page in materialized}
            removed_or_changed = {
                path for path, page_hash in existing.items() if incoming.get(path) != page_hash
            }
            additions = tuple(
                page for page in materialized if existing.get(page.logical_path) != _page_hash(page)
            )
            with self._connection:
                for logical_path in sorted(removed_or_changed):
                    if self._fts_available:
                        self._connection.execute(
                            "DELETE FROM pages_fts WHERE logical_path = ?", (logical_path,)
                        )
                    self._connection.execute(
                        "DELETE FROM pages WHERE logical_path = ?", (logical_path,)
                    )
                self._insert_pages(additions)
        except sqlite3.Error:
            self._markdown_pages = materialized
            raise
        self._markdown_pages = materialized

    def search(self, query: str, limit: int) -> list[SearchResult]:
        """Search cache rows, falling back to bounded Markdown kept by this process."""
        normalized = query.strip()
        if not normalized:
            return []
        bounded_limit = _bounded_limit(limit)
        try:
            if len(normalized) < 3 or not self.trigram:
                title_tag = self._title_tag_search(normalized, bounded_limit)
                if title_tag:
                    return title_tag
            if self._fts_available:
                fts_results = self._fts_search(normalized, bounded_limit)
                if fts_results:
                    return fts_results
            return self._like_search(normalized, bounded_limit)
        except sqlite3.DatabaseError:
            return bounded_markdown_search(self._markdown_pages, normalized, bounded_limit)

    def _insert_pages(self, pages: Iterable[WikiPage]) -> None:
        for page in pages:
            page_hash = _page_hash(page)
            summary = _summary(page.body)
            self._connection.execute(
                """
                INSERT INTO pages(logical_path, content_hash, revision, title, tags, summary, body)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    page.logical_path,
                    page_hash,
                    page.revision,
                    page.title,
                    " ".join(page.tags),
                    summary,
                    page.body,
                ),
            )
            if self._fts_available:
                self._connection.execute(
                    """
                    INSERT INTO pages_fts(logical_path, title, tags, summary, body)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (page.logical_path, page.title, " ".join(page.tags), summary, page.body),
                )

    def _title_tag_search(self, query: str, limit: int) -> list[SearchResult]:
        needle = f"%{_escape_like(query.casefold())}%"
        rows = self._connection.execute(
            """
            SELECT logical_path, title, summary, body, revision, content_hash
            FROM pages
            WHERE lower(title) LIKE ? ESCAPE '\\' OR lower(tags) LIKE ? ESCAPE '\\'
            ORDER BY logical_path
            LIMIT ?
            """,
            (needle, needle, limit),
        ).fetchall()
        return [_result_from_row(row, score=0.0, query=query) for row in rows]

    def _fts_search(self, query: str, limit: int) -> list[SearchResult]:
        # A quoted phrase makes punctuation data rather than FTS query syntax.
        fts_query = f'"{query.replace('"', '""')}"'
        rows = self._connection.execute(
            """
            SELECT pages.logical_path, pages.title, pages.summary, pages.body,
                   pages.revision, pages.content_hash, bm25(pages_fts) AS score
            FROM pages_fts
            JOIN pages ON pages.logical_path = pages_fts.logical_path
            WHERE pages_fts MATCH ?
            ORDER BY score, pages.logical_path
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [_result_from_row(row[:6], score=float(row[6]), query=query) for row in rows]

    def _like_search(self, query: str, limit: int) -> list[SearchResult]:
        needle = f"%{_escape_like(query.casefold())}%"
        rows = self._connection.execute(
            """
            SELECT logical_path, title, summary, body, revision, content_hash
            FROM pages
            WHERE lower(logical_path) LIKE ? ESCAPE '\\'
               OR lower(title) LIKE ? ESCAPE '\\'
               OR lower(tags) LIKE ? ESCAPE '\\'
               OR lower(summary) LIKE ? ESCAPE '\\'
               OR lower(body) LIKE ? ESCAPE '\\'
            ORDER BY logical_path
            LIMIT ?
            """,
            (needle, needle, needle, needle, needle, limit),
        ).fetchall()
        return [_result_from_row(row, score=0.0, query=query) for row in rows]


def bounded_markdown_search(
    pages: Iterable[WikiPage], query: str, limit: int
) -> list[SearchResult]:
    """Search validated pages directly when SQLite is unavailable or unhealthy."""
    normalized = query.strip()
    if not normalized:
        return []
    needle = normalized.casefold()
    results: list[SearchResult] = []
    for page in _validated_pages(pages):
        haystack = "\n".join(
            (page.logical_path, page.title, " ".join(page.tags), page.body)
        ).casefold()
        if needle in haystack:
            results.append(
                SearchResult(
                    logical_path=page.logical_path,
                    title=page.title,
                    summary=_summary(page.body),
                    snippet=_snippet(page.body, normalized),
                    score=0.0,
                    revision=page.revision,
                    content_hash=_page_hash(page),
                )
            )
    return sorted(results, key=lambda result: result.logical_path)[: _bounded_limit(limit)]


def _open_database(path: Path, *, wal: bool) -> tuple[sqlite3.Connection, bool]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        if wal:
            connection.execute("PRAGMA journal_mode=WAL")
        else:
            connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                logical_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                revision INTEGER NOT NULL,
                title TEXT NOT NULL,
                tags TEXT NOT NULL,
                summary TEXT NOT NULL,
                body TEXT NOT NULL
            )
            """
        )
        trigram = _create_fts(connection)
        _validate_cache_schema(connection)
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection, trigram


def _create_fts(connection: sqlite3.Connection) -> bool:
    """Create the best available FTS5 index, returning trigram support."""
    existing = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'pages_fts'"
    ).fetchone()
    if existing is not None:
        definition = str(existing[0] or "")
        if "VIRTUAL TABLE" not in definition.upper() or "USING FTS5" not in definition.upper():
            raise sqlite3.DatabaseError("Wiki search cache has an invalid FTS table")
        return (
            re.search(r"tokenize\s*=\s*['\"]trigram", definition, flags=re.IGNORECASE) is not None
        )
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                logical_path UNINDEXED, title, tags, summary, body, tokenize='trigram'
            )
            """
        )
        return True
    except sqlite3.OperationalError:
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                    logical_path UNINDEXED, title, tags, summary, body
                )
                """
            )
        except sqlite3.OperationalError:
            return False
        return False


def _has_fts_table(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pages_fts'"
        ).fetchone()
        is not None
    )


def _validate_cache_schema(connection: sqlite3.Connection) -> None:
    pages_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(pages)"))
    if pages_columns != _PAGES_COLUMNS:
        raise sqlite3.DatabaseError("Wiki search cache has an invalid pages table")
    if _has_fts_table(connection):
        fts_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(pages_fts)"))
        if fts_columns != _FTS_COLUMNS:
            raise sqlite3.DatabaseError("Wiki search cache has an invalid FTS table")


def _quarantine_corrupt_database(path: Path) -> Path:
    diagnostic = path.with_name(f"{path.name}.corrupt-{uuid4().hex}")
    if path.exists():
        os.replace(path, diagnostic)
    _remove_cache_artifacts(path)
    return diagnostic


def _is_rebuildable_cache_error(error: sqlite3.DatabaseError) -> bool:
    """Only discard an identified bad derivative cache, never an unrelated DB failure."""
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "file is not a database",
            "file is encrypted",
            "database disk image is malformed",
            "invalid pages table",
            "invalid fts table",
        )
    )


def _remove_cache_artifacts(path: Path) -> None:
    for candidate in (path, path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")):
        with contextlib.suppress(FileNotFoundError):
            candidate.unlink()


def _validated_pages(pages: Iterable[WikiPage]) -> tuple[WikiPage, ...]:
    materialized = tuple(sorted(pages, key=lambda page: page.logical_path))
    paths = [page.logical_path for page in materialized]
    if len(paths) != len(set(paths)):
        raise ValueError("Wiki search pages must have distinct logical paths")
    # Rendering also repeats the strict page-body safety validation before cache input.
    for page in materialized:
        render_page(page)
    return materialized


def _page_hash(page: WikiPage) -> str:
    return content_hash(render_page(page).encode("utf-8"))


def _summary(body: str) -> str:
    return " ".join(body.split())[:_SNIPPET_LENGTH]


def _snippet(body: str, query: str) -> str:
    compact = " ".join(body.split())
    start = compact.casefold().find(query.casefold())
    if start < 0:
        return compact[:_SNIPPET_LENGTH]
    left = max(0, start - (_SNIPPET_LENGTH // 3))
    right = min(len(compact), start + len(query) + (2 * _SNIPPET_LENGTH // 3))
    prefix = "…" if left else ""
    suffix = "…" if right < len(compact) else ""
    return f"{prefix}{compact[left:right]}{suffix}"


def _result_from_row(
    row: sqlite3.Row | tuple[object, ...], *, score: float, query: str
) -> SearchResult:
    logical_path, title, summary, body, revision, page_hash = row
    return SearchResult(
        logical_path=str(logical_path),
        title=str(title),
        summary=str(summary),
        snippet=_snippet(str(body), query),
        score=score,
        revision=int(str(revision)),
        content_hash=str(page_hash),
    )


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _bounded_limit(limit: int) -> int:
    return max(1, min(int(limit), _MAX_RESULTS))
