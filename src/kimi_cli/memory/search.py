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

import re
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


#: Words that carry no signal in a question and would match half the store.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "about",
        "from",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "why",
        "how",
        "which",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "his",
        "her",
        "their",
        "they",
        "them",
        "he",
        "she",
        "you",
        "your",
        "i",
        "me",
        "my",
        "we",
        "our",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "if",
        "then",
        "than",
        "so",
        "as",
        "such",
        "not",
        "no",
        "nor",
        "but",
    }
)

#: Below this a term cannot be indexed by trigram and is left to the scan.
_MIN_TERM_LEN = 3


#: A CJK run longer than this is a sentence, not a phrase, and matching it
#: whole finds nothing.
_CJK_PHRASE_MAX = 4

#: Two characters is the commonest Chinese word length and below what a trigram
#: index can hold, so these are for the scan rather than for FTS5.
_CJK_BIGRAM_LEN = 2

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def _cjk_terms(run: str) -> list[str]:
    """Break a Chinese run into things that might actually appear in an entry.

    Chinese is written without spaces and there is no segmenter here, so a
    whole run is either the phrase someone meant or an entire sentence — and a
    sentence never appears verbatim. Sliding character n-grams cover both: the
    run itself when it is short enough to be a phrase, and otherwise every
    3-character window, which is the shortest unit the trigram index can match.

    Measured: "邮箱是怎么配置的" as one phrase found nothing; windowed, it finds
    the entry about mailbox configuration.
    """
    if len(run) <= _CJK_PHRASE_MAX:
        return [run] if len(run) >= _MIN_TERM_LEN else []
    return [run[i : i + _MIN_TERM_LEN] for i in range(len(run) - _MIN_TERM_LEN + 1)]


def _cjk_bigrams(query: str) -> list[str]:
    """Every two-character window of the query's Chinese runs.

    Two characters is the commonest Chinese word length — 邮箱, 配置, 路径 — and
    is below what the trigram index can match, so these exist for the scan.
    Measured: "邮箱是怎么配置的" finds nothing through the index at any window
    size, and finds the right entry on the bigrams 邮箱 and 配置.
    """
    out: list[str] = []
    for run in _CJK_RUN.findall(query):
        for i in range(len(run) - _CJK_BIGRAM_LEN + 1):
            gram = run[i : i + _CJK_BIGRAM_LEN]
            if gram not in out:
                out.append(gram)
    return out


def _terms(query: str) -> list[str]:
    """Split a query into searchable terms.

    Latin words are split and stripped of stopwords: a question is mostly
    grammar, and matching on "when" or "the" returns the whole store. Chinese
    goes through :func:`_cjk_terms`.
    """
    out: list[str] = []
    for piece in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9._\'-]*", query):
        candidates = _cjk_terms(piece) if _CJK_RUN.fullmatch(piece) else [piece]
        for term in candidates:
            if len(term) < _MIN_TERM_LEN or term.casefold() in _STOPWORDS:
                continue
            if term not in out:
                out.append(term)
    return out


def _match_expression(query: str) -> str | None:
    """Build an FTS5 MATCH expression, or ``None`` if nothing is searchable.

    Every term is quoted — models write `core.py` and `src/`, and FTS5 reads
    `.`, `/` and `-` as syntax — then joined with OR so that any one of them
    can hit.

    The OR matters more than it sounds. Matching the whole query as a single
    phrase, which is what this did at first, means a natural question can only
    match a turn that contains that question verbatim. Measured on LoCoMo, that
    scored exactly zero: 1,982 questions, no hits at any depth.
    """
    terms = _terms(query)
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _scan(needles: Sequence[str], entries: Sequence[MemoryEntry], *, limit: int) -> list[SearchHit]:
    """Substring match on any of *needles*, best-covered entry first.

    Linear, and deliberately: it runs only when the index cannot help, over a
    store small enough that the cost is not measurable. Ranking by how many
    needles an entry contains keeps a scan over many fragments from returning
    whichever entry happened to be written first.
    """
    folded = [n.casefold() for n in needles if n]
    if not folded:
        return []
    scored: list[tuple[int, int, MemoryEntry, int]] = []
    for order, entry in enumerate(entries):
        body = entry.content.casefold()
        positions = [body.find(n) for n in folded]
        matched = [p for p in positions if p >= 0]
        if matched:
            scored.append((len(matched), -order, entry, min(matched)))
    scored.sort(reverse=True)

    hits: list[SearchHit] = []
    for _, _, entry, position in scored[:limit]:
        start = max(0, position - 12)
        snippet = entry.content[start : position + 40].replace("\n", " ")
        hits.append(SearchHit(entry.id, entry.handle, entry.kind, snippet.strip()))
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
        expression = _match_expression(query)
        if expression is None:
            # Nothing long enough to index — two-character Chinese, a bare
            # number, a stopword. The scan still finds those.
            return _scan([query, *_cjk_bigrams(query)], entries, limit=limit)
        try:
            with sqlite3.connect(self._db_path) as db:
                db.executescript(_SCHEMA)
                self._sync(db, entries)
                rows = db.execute(
                    "SELECT entry_id, handle, kind, snippet(entries, 3, '', '', '…', 24) "
                    "FROM entries WHERE entries MATCH ? ORDER BY rank LIMIT ?",
                    (expression, limit),
                ).fetchall()
        except sqlite3.Error:
            logger.warning("memory search failed; treating as no results", exc_info=True)
            return []
        if rows:
            return [SearchHit(*row) for row in rows]
        # The index found nothing. For Chinese that is expected rather than
        # conclusive: the words that carry the meaning are usually two
        # characters, which no trigram index holds.
        bigrams = _cjk_bigrams(query)
        return _scan(bigrams, entries, limit=limit) if bigrams else []

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
