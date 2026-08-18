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

import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest
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


def _document_frequency(needles: Sequence[str], bodies: Sequence[str]) -> dict[str, int]:
    """How many entries contain each needle.

    One pass over the corpus for all needles together rather than one pass per
    needle: with twenty needles that is the difference between twenty full
    sweeps and one. The inner loop is a substring test, which is C-speed, so
    the cost is dominated by how many times the corpus is walked.
    """
    counts = dict.fromkeys(needles, 0)
    for body in bodies:
        for needle in needles:
            if needle in body:
                counts[needle] += 1
    return counts


class _FoldedBodies:
    """Case-folded entry text, reused across searches over the same entries.

    ``str.casefold`` on every entry on every query allocates a second copy of
    the whole store per search. At the sizes memory was designed for that is
    nothing; measured at twenty thousand entries it is most of a 63 ms Chinese
    query. Entries are immutable and rebuilt wholesale when the file changes,
    so identity of the list plus its length is a sound cache key.
    """

    def __init__(self) -> None:
        self._key: tuple[int, int] | None = None
        self._bodies: list[str] = []

    def of(self, entries: Sequence[MemoryEntry]) -> list[str]:
        key = (id(entries), len(entries))
        if key != self._key:
            self._bodies = [e.content.casefold() for e in entries]
            self._key = key
        return self._bodies


def _needles(query: str) -> list[str]:
    """Everything in *query* worth matching a substring on.

    Chinese contributes two-character windows, which is the commonest word
    length and below what the trigram index can hold. Latin contributes whole
    terms: dropping them would make "CodeGraph 有什么限制" search only its
    Chinese half. Chinese *trigrams* are deliberately excluded even though
    :func:`_terms` produces them — counting a word in both its bigram and
    trigram forms weights it twice and measurably dilutes the ranking.
    """
    latin = [t.casefold() for t in _terms(query) if not _CJK_RUN.search(t)]
    return list(dict.fromkeys(_cjk_bigrams(query) + latin))


def _scan(
    needles: Sequence[str],
    entries: Sequence[MemoryEntry],
    *,
    limit: int,
    cache: _FoldedBodies | None = None,
) -> list[SearchHit]:
    """Substring match on *needles*, ranked by how much each one narrows things.

    Counting matched needles equally — which this did at first — makes a
    question's grammar worth as much as its subject: an entry containing "什么"
    outranked one containing "互助小组". Weighting by inverse document frequency
    fixes that without a stopword list for every language, and on a Chinese
    LoCoMo it moved recall@8 from 47.7% to 57.9%.

    Linear over the store, and deliberately so: it runs against entries already
    held in memory, and the alternative to a scan is an embedding model.
    """
    folded = [n.casefold() for n in needles if n]
    if not folded:
        return []

    bodies = cache.of(entries) if cache is not None else [e.content.casefold() for e in entries]
    n_docs = len(bodies) or 1
    df = _document_frequency(folded, bodies)
    kept = folded

    # Weight, rather than a cutoff. Dropping needles above a frequency
    # threshold scored marginally better on a store of two thousand entries and
    # was destructive on a real one: a memory file holds tens of entries, five
    # percent of which rounds below one, so every needle that actually matched
    # something was the one discarded. Down-weighting has no such cliff.
    weight = {n: math.log(1 + n_docs / (1 + df[n])) for n in kept}

    scored: list[tuple[float, int, MemoryEntry, int]] = []
    for order, (entry, body) in enumerate(zip(entries, bodies, strict=True)):
        positions = [(body.find(n), n) for n in kept]
        matched = [(p, n) for p, n in positions if p >= 0]
        if matched:
            total = sum(weight[n] for _, n in matched)
            scored.append((total, -order, entry, min(p for p, _ in matched)))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)

    hits: list[SearchHit] = []
    for _, _, entry, position in scored[:limit]:
        start = max(0, position - 12)
        snippet = entry.content[start : position + 40].replace("\n", " ")
        hits.append(SearchHit(entry.id, entry.handle, entry.kind, snippet.strip()))
    return hits


def _merge(
    primary: Sequence[SearchHit], secondary: Sequence[SearchHit], *, limit: int
) -> list[SearchHit]:
    """Interleave two result lists, keeping each side's order and dropping dupes.

    Interleaved rather than concatenated because neither side deserves the top
    of the list by default: one query can be mostly English and the next mostly
    Chinese, and appending would bury whichever half the store happened to
    answer better.
    """
    out: list[SearchHit] = []
    seen: set[str] = set()
    for pair in zip_longest(primary, secondary):
        for hit in pair:
            if hit is not None and hit.entry_id not in seen:
                seen.add(hit.entry_id)
                out.append(hit)
                if len(out) >= limit:
                    return out
    return out


def _merge_tail(
    primary: Sequence[SearchHit], filler: Sequence[SearchHit], *, limit: int
) -> list[SearchHit]:
    """*primary* in full, then whatever of *filler* is new, up to *limit*.

    Interleaving instead — which is right when neither side is known to be
    better — costs four points at depth 8 on Chinese, because every second slot
    goes to the ranker that was measured to be the weaker one there.
    """
    out = list(primary)
    seen = {h.entry_id for h in out}
    for hit in filler:
        if len(out) >= limit:
            break
        if hit.entry_id not in seen:
            seen.add(hit.entry_id)
            out.append(hit)
    return out[:limit]


class MemorySearchIndex:
    """A rebuildable FTS5 index over memory entries."""

    def __init__(self, db_path: Path, source: Path) -> None:
        self._db_path = db_path
        self._source = source
        self._folded = _FoldedBodies()

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

        # Chinese is where the index is weakest and the scan is strongest, so
        # the two swap roles depending on the script. Measured on LoCoMo, on
        # the same 302 questions in both languages: for Chinese, FTS5 alone
        # reaches 40.1% at depth 8 and the weighted scan alone 57.9%, so the
        # scan leads and the index only fills what it leaves empty. For Latin
        # the index is the better ranker and keeps the lead it has always had.
        # Choosing per query rather than globally is what makes the Chinese
        # gain cost the English path nothing.
        if _CJK_RUN.search(query):
            hits = _scan(_needles(query), entries, limit=limit, cache=self._folded)
            if len(hits) >= limit:
                return hits
            return _merge_tail(hits, self._fts(query, entries, limit=limit), limit=limit)

        expression = _match_expression(query)
        if expression is None:
            # Nothing long enough to index — a bare number, a stopword. The
            # scan still finds those.
            return _scan(_needles(query) or [query], entries, limit=limit, cache=self._folded)
        return self._fts(query, entries, limit=limit, expression=expression)

    def _fts(
        self,
        query: str,
        entries: Sequence[MemoryEntry],
        *,
        limit: int,
        expression: str | None = None,
    ) -> list[SearchHit]:
        """Index-side results, or nothing if the index cannot serve the query.

        Failure returns nothing rather than raising: search is an aid, and an
        unusable index should degrade to "no results", not break the tool call.
        """
        if expression is None:
            expression = _match_expression(query)
            if expression is None:
                return []
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
