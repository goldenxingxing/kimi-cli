"""Full-text search over stored memory.

The index is a cache, not the record. ``persistent.jsonl`` stays the source of
truth — plain text, hand-editable, greppable — and this rebuilds from it
whenever it has changed. Losing the database costs a rebuild and nothing else,
which is the property that lets memory keep being a file you can open.

SQLite's FTS5 ships with Python, so retrieval arrives without a service, a
daemon, or an embedding model.

Tokenisation is ``trigram`` rather than the default. ``unicode61`` splits on
non-alphanumerics, and Chinese is written without spaces, so a whole sentence
becomes a single token and every query for a phrase inside it misses. Measured
on a real entry: searching "仓库路径" returns nothing under ``unicode61`` and
finds it under ``trigram``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.utils.logging import logger

#: Below this, FTS5 cannot help: a trigram index has nothing to match on. Two
#: characters is a whole word in Chinese ("邮箱", "配置"), so those queries fall
#: back to a substring scan rather than returning nothing. Linear, but over a
#: store this size it is immeasurable, and a wrong answer of "no matches" is
#: worse than a scan.
FTS_MIN_QUERY_LEN = 3

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS entries USING fts5(
    entry_id UNINDEXED,
    handle,
    kind UNINDEXED,
    body,
    tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


@dataclass(frozen=True, slots=True)
class SearchHit:
    entry_id: str
    handle: str
    kind: str
    snippet: str


def _fingerprint(path: Path) -> str:
    """Cheap "has the file changed" stamp. Size plus mtime is enough here.

    A hash would be exact, but the file is rewritten wholesale on every edit,
    so a changed size or timestamp already covers every real mutation.
    """
    try:
        st = path.stat()
    except OSError:
        return "missing"
    return f"{st.st_size}:{st.st_mtime_ns}"


def _quote(query: str) -> str:
    """Turn free text into an FTS5 MATCH expression that cannot be a syntax error.

    Callers pass what a model wrote, which routinely contains `.`, `/`, `-` and
    quotes — all of which mean something to the FTS5 grammar. Everything is
    wrapped as a quoted phrase, so the query is matched literally rather than
    interpreted.
    """
    return '"' + query.replace('"', '""') + '"'


def _scan(query: str, entries: Sequence[MemoryEntry], *, limit: int) -> list[SearchHit]:
    """Case-insensitive substring match, for queries too short to index."""
    needle = query.casefold()
    hits: list[SearchHit] = []
    for entry in entries:
        position = entry.content.casefold().find(needle)
        if position < 0:
            continue
        start = max(0, position - 12)
        snippet = entry.content[start : position + len(query) + 12].replace("\n", " ")
        hits.append(SearchHit(entry.id, entry.handle, entry.kind, snippet.strip()))
        if len(hits) >= limit:
            break
    return hits


class MemorySearchIndex:
    """A rebuildable FTS5 index over memory entries."""

    def __init__(self, db_path: Path, source: Path) -> None:
        self._db_path = db_path
        self._source = source

    def search(
        self, query: str, entries: Sequence[MemoryEntry], *, limit: int = 8
    ) -> list[SearchHit]:
        """Return hits for *query*, refreshing the index if the store moved.

        Failure returns nothing rather than raising: search is an aid, and an
        unusable index should degrade to "no results", not break the tool call.
        """
        query = (query or "").strip()
        if not query:
            return []
        if len(query) < FTS_MIN_QUERY_LEN:
            return _scan(query, entries, limit=limit)
        try:
            with sqlite3.connect(self._db_path) as db:
                db.executescript(_SCHEMA)
                self._sync(db, entries)
                rows = db.execute(
                    "SELECT entry_id, handle, kind, snippet(entries, 3, '', '', '…', 24) "
                    "FROM entries WHERE entries MATCH ? ORDER BY rank LIMIT ?",
                    (_quote(query), limit),
                ).fetchall()
        except sqlite3.Error:
            logger.warning("memory search failed; treating as no results", exc_info=True)
            return []
        return [SearchHit(*row) for row in rows]

    def _sync(self, db: sqlite3.Connection, entries: Sequence[MemoryEntry]) -> None:
        """Rebuild the index when the source file has changed since last time."""
        stamp = _fingerprint(self._source)
        current = db.execute("SELECT v FROM meta WHERE k = 'fingerprint'").fetchone()
        if current and current[0] == stamp:
            return
        db.execute("DELETE FROM entries")
        db.executemany(
            "INSERT INTO entries (entry_id, handle, kind, body) VALUES (?, ?, ?, ?)",
            [(e.id, e.handle, e.kind, e.content) for e in entries],
        )
        db.execute(
            "INSERT INTO meta (k, v) VALUES ('fingerprint', ?) "
            "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (stamp,),
        )
