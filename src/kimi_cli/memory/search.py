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
from collections.abc import Mapping, Sequence
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

-- Which entries are indexed and at what content, so a write that appends one
-- entry costs one insert rather than a rebuild of everything.
CREATE TABLE IF NOT EXISTS indexed (
    num INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    stamp TEXT NOT NULL
);

-- Posting list for Chinese bigrams. The trigram FTS index cannot hold them —
-- two characters is below its window and is also the commonest Chinese word
-- length — so matching them meant a substring pass over the whole store for
-- every needle. Here a query reaches only the entries that can match.
-- Keyed by the integer from `indexed`, not by the entry's 32-character id:
-- there is one row per (gram, entry) and a store of twenty thousand entries
-- produces most of a million of them, where the difference is a hundred
-- megabytes of index against sixteen.
CREATE TABLE IF NOT EXISTS grams (
    gram TEXT NOT NULL,
    num INTEGER NOT NULL,
    PRIMARY KEY (gram, num)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS grams_by_entry ON grams (num);
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


def _content_stamp(entry: MemoryEntry) -> str:
    """Cheap "has this entry changed" marker for incremental indexing."""
    return f"{len(entry.content)}:{hash(entry.content) & 0xFFFFFFFF:08x}"


def _hit(entry: MemoryEntry, position: int) -> SearchHit:
    start = max(0, position - 12)
    snippet = entry.content[start : position + 40].replace("\n", " ")
    return SearchHit(entry.id, entry.handle, entry.kind, snippet.strip())


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


def _entry_bigrams(text: str) -> set[str]:
    """Every distinct two-character Chinese window in *text*, for the posting list.

    Indexed rather than searched: this is the same extraction the query side
    uses, so a gram found here is exactly a substring the scan would have
    found, and the posting list can replace the sweep without changing which
    entries match.
    """
    out: set[str] = set()
    for run in _CJK_RUN.findall(text):
        for i in range(len(run) - _CJK_BIGRAM_LEN + 1):
            out.add(run[i : i + _CJK_BIGRAM_LEN])
    return out


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


def _rank(
    candidates: Sequence[tuple[int, MemoryEntry, str]],
    needles: Sequence[str],
    weight: Mapping[str, float],
    *,
    limit: int,
) -> list[SearchHit]:
    """Score *candidates* against *needles*, best first.

    Shared by the sweep and the posting-list path so that the only difference
    between them is which entries get looked at. Keeping two copies of this
    produced results that differed on 96 of 197 questions: the totals were sums
    of the same floats in a different order, which is enough to reorder ties,
    and one of the ties it reordered was the correct answer falling out of the
    top eight. `order` is each entry's position in the store, so it has to be
    passed in rather than recomputed over a subset.
    """
    scored: list[tuple[float, int, MemoryEntry, int]] = []
    for order, entry, body in candidates:
        positions = [(body.find(n), n) for n in needles]
        matched = [(at, n) for at, n in positions if at >= 0]
        if matched:
            total = sum(weight[n] for _, n in matched)
            scored.append((total, -order, entry, min(at for at, _ in matched)))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [_hit(entry, at) for _, _, entry, at in scored[:limit]]


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

    # Weight, rather than a cutoff. Dropping needles above a frequency
    # threshold scored marginally better on a store of two thousand entries and
    # was destructive on a real one: a memory file holds tens of entries, five
    # percent of which rounds below one, so every needle that actually matched
    # something was the one discarded. Down-weighting has no such cliff.
    weight = {n: math.log(1 + n_docs / (1 + df[n])) for n in folded}

    return _rank(
        [(i, e, b) for i, (e, b) in enumerate(zip(entries, bodies, strict=True))],
        folded,
        weight,
        limit=limit,
    )


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
            hits = self._scan_via_postings(query, entries, limit=limit)
            if hits is None:
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
        """Bring the index in line with *entries*, touching only what changed.

        The whole index used to be dropped and rebuilt whenever the source file
        moved, and the source file moves on every memory write — so adding one
        entry to a large store cost a full reindex before the next search could
        run (measured at 690 ms for twenty thousand entries). Comparing per
        entry costs one cheap query and makes the common case, an append, cost
        one insert.
        """
        stamp = _fingerprint(self._source)
        current = db.execute("SELECT v FROM meta WHERE k = 'fingerprint'").fetchone()
        if current and current[0] == stamp:
            return

        want = {e.id: (e, _content_stamp(e)) for e in entries}
        have = {
            row[0]: (row[1], row[2])
            for row in db.execute("SELECT entry_id, stamp, num FROM indexed")
        }

        stale = [(i, have[i][1]) for i in have if i not in want or want[i][1] != have[i][0]]
        fresh = [i for i, (_, st) in want.items() if i not in have or have[i][0] != st]

        if stale:
            db.executemany("DELETE FROM entries WHERE entry_id = ?", [(i,) for i, _ in stale])
            db.executemany("DELETE FROM grams WHERE num = ?", [(n,) for _, n in stale])
            db.executemany("DELETE FROM indexed WHERE entry_id = ?", [(i,) for i, _ in stale])

        if fresh:
            rows = [want[i][0] for i in fresh]
            db.executemany(
                "INSERT INTO entries (entry_id, handle, kind, body) VALUES (?, ?, ?, ?)",
                [(e.id, e.handle, e.kind, e.content) for e in rows],
            )
            db.executemany(
                "INSERT INTO indexed (entry_id, stamp) VALUES (?, ?)",
                [(e.id, want[e.id][1]) for e in rows],
            )
            nums = dict(
                db.execute(
                    "SELECT entry_id, num FROM indexed WHERE entry_id IN "
                    f"({','.join('?' * len(rows))})",  # noqa: S608
                    [e.id for e in rows],
                )
            )
            db.executemany(
                "INSERT OR IGNORE INTO grams (gram, num) VALUES (?, ?)",
                [(g, nums[e.id]) for e in rows for g in _entry_bigrams(e.content.casefold())],
            )

        db.execute(
            "INSERT INTO meta (k, v) VALUES ('fingerprint', ?) "
            "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (stamp,),
        )

    def _scan_via_postings(
        self, query: str, entries: Sequence[MemoryEntry], *, limit: int
    ) -> list[SearchHit] | None:
        """The weighted scan, but reaching only entries that can match.

        Returns ``None`` when the index is unusable, so the caller can fall back
        to sweeping the store rather than silently returning nothing.

        Document frequency comes from the posting list by counting rows, which
        is the same number the sweep computed by testing every entry, so the
        ranking is unchanged — only the number of entries touched is. Latin
        needles keep substring semantics ("deploy" matching "deployment"), which
        a token posting list cannot express, so they are applied to the
        candidates the Chinese half turned up rather than to the whole store.
        """
        needles = [n.casefold() for n in _needles(query)]
        grams = [g.casefold() for g in _cjk_bigrams(query)]
        if not grams:
            return None
        latin = [n for n in needles if n not in set(grams)]

        try:
            with sqlite3.connect(self._db_path) as db:
                db.executescript(_SCHEMA)
                self._sync(db, entries)
                placeholders = ",".join("?" * len(grams))
                pairs = db.execute(
                    "SELECT g.gram, i.entry_id FROM grams g JOIN indexed i ON i.num = g.num "
                    f"WHERE g.gram IN ({placeholders})",  # noqa: S608
                    grams,
                ).fetchall()
                n_docs = db.execute("SELECT COUNT(*) FROM indexed").fetchone()[0] or 1
        except sqlite3.Error:
            logger.warning("posting lookup failed; falling back to a scan", exc_info=True)
            return None

        df: dict[str, int] = dict.fromkeys(grams, 0)
        candidate_ids: set[str] = set()
        for gram, entry_id in pairs:
            df[gram] += 1
            candidate_ids.add(entry_id)

        bodies = self._folded.of(entries)
        if len(bodies) != n_docs:
            # The index is describing a different store than the caller holds.
            # Sweeping is correct where guessing is not.
            return None

        # Latin needles keep substring semantics — "deploy" matching
        # "deployment" — which a token posting list cannot express, so their
        # frequency comes from a pass over the bodies. That pass also collects
        # the entries the posting list cannot see: one matching a Latin term
        # and no Chinese gram is a candidate the sweep would have scored.
        for needle in latin:
            df[needle] = 0
        if latin:
            for entry, body in zip(entries, bodies, strict=True):
                for needle in latin:
                    if needle in body:
                        df[needle] += 1
                        candidate_ids.add(entry.id)

        if not candidate_ids:
            return []

        weight = {n: math.log(1 + n_docs / (1 + df[n])) for n in needles}
        return _rank(
            [
                (i, e, b)
                for i, (e, b) in enumerate(zip(entries, bodies, strict=True))
                if e.id in candidate_ids
            ],
            needles,
            weight,
            limit=limit,
        )
