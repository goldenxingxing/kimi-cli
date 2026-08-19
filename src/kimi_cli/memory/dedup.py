"""Deciding whether a memory write says something the store already holds.

Cross-session memory is injected in full at the start of every session, so a
fact recorded twice is paid for on every subsequent conversation. Nothing used
to prevent that but a line in the tool description asking the model not to.

The hard part is not finding duplicates — it is refusing to merge two things
that merely *look* alike. ``difflib`` measures character overlap, and character
overlap cannot tell a rephrasing from a correction:

    "Merge freeze begins 2026-03-05"  vs  "Merge freeze begins on 2026-03-05"   0.967
    "Merge freeze begins 2026-03-05"  vs  "Merge freeze begins 2026-04-05"      0.967

Identical scores; the first pair must merge and the second must never. So the
ratio alone decides nothing. A merge additionally requires passing three guards
(:func:`may_merge`) that look at what actually changed rather than how much.
The guards and the threshold fail in different directions, which is the point:
a mis-set threshold cannot on its own destroy a fact.

A merge overwrites the older wording, so it is the only lossy operation here.
It is applied to the incoming write only — never to records already on disk,
which are folded on exact normalized equality alone.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from kimi_cli.memory.entry import MemoryEntry, MemoryKind
from kimi_cli.utils.string import fold_text

if TYPE_CHECKING:
    # recent.py imports this module for its own dedup, so the summary model can
    # only be referenced as a type here.
    from kimi_cli.memory.recent import SessionSummary

# Ratio at or above which a near-identical entry is merged, provided the guards
# also pass. Normalization has already removed case, width, and whitespace, so
# at 0.95 a 100-character entry may differ by ~5 characters — punctuation, an
# article, a plural, a preposition, a typo. Every *semantic* difference that
# fits in 5 characters is a number, an identifier, or a negation, and those are
# what the guards below remove independently of this number.
AUTO_MERGE_RATIO = 0.95

# Ratio at or above which an entry is reported to the caller as a possible
# duplicate without being touched. Turns "probably the same" into information
# rather than action.
ADVISORY_RATIO = 0.85

# Guard G1. Below this length ratio the shorter text cannot be a restatement of
# the longer one, only a lossier version of it.
MIN_LENGTH_RATIO = 0.60

# Above this length, compare whitespace-separated tokens instead of characters.
# Character-level ratio() on a 4000-character pair costs ~147ms and would put
# seconds onto the session-end path; the token-level ratio costs ~11ms and
# lands within 0.001 of the same number on real prose.
CHAR_LEVEL_MAX_LEN = 2_000

# How many possible duplicates to report back. The caller shows these to a
# model or a user; past a handful the list stops being actionable.
MAX_ADVISORIES = 3

# Cross-session summary similarity is deliberately not used. Two sessions
# producing near-identical summaries are still two distinct events at distinct
# times, and the FIFO cap already bounds growth. Named rather than omitted so
# the decision is visible. If it is ever wanted, the right measure is word-set
# Jaccard, not a character ratio.
CROSS_SESSION_JACCARD: float | None = None

DuplicateAction = Literal["merge", "advise", "create"]
SummaryPolicy = Literal["supersede", "skip_if_session_present"]

# Numbers, including dotted/dashed/colon-joined runs, so a date, a version, a
# time, or a path segment is one token rather than several.
_NUMERIC_RE = re.compile(r"[0-9]+(?:[.:/_-][0-9]+)*")

# Negation markers in both languages the product ships in. CJK markers cannot
# use \b, which is defined on word characters.
_NEGATION_RE = re.compile(
    r"\bnot\b|\bno\b|\bnever\b|\bnone\b|\bdon'?t\b|\bdoesn'?t\b|\bdidn'?t\b"
    r"|\bwon'?t\b|\bcan'?t\b|\bcannot\b|\bshouldn'?t\b|\bavoid\b|\bwithout\b"
    r"|不|别|勿|禁止|无需|不要"
)

# NFKC leaves the typographic apostrophe and quotes alone, so "don't" and
# "don't" would otherwise never match. Safe to fold here because memory
# matching is a convenience boundary, unlike the Wiki's intent hashing.
_QUOTE_TRANSLATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


@dataclass(frozen=True, slots=True)
class DuplicateVerdict:
    """What to do with an incoming entry, and what the caller should be told."""

    action: DuplicateAction
    target_id: str | None
    """The id of the older entry to merge into. Set only when action is merge."""
    score: float
    advisories: tuple[tuple[str, str, float], ...] = ()
    """``(id, content, score)`` for entries the caller may have meant instead."""


CREATE_VERDICT = DuplicateVerdict(action="create", target_id=None, score=0.0)


@dataclass(frozen=True, slots=True)
class SummaryPlacement:
    """The result of fitting one summary into the kept list."""

    items: list[SessionSummary]
    stored: SessionSummary | None
    """``None`` when the incoming summary added nothing and was dropped."""
    superseded_id: str | None


def normalize_content(raw: str) -> str:
    """Fold the axes that never change what a memory means."""
    return fold_text(raw.translate(_QUOTE_TRANSLATION))


def numeric_tokens(text: str) -> list[str]:
    """Every number in *text*, as a sorted multiset."""
    return sorted(_NUMERIC_RE.findall(text))


def has_negation(text: str) -> bool:
    return _NEGATION_RE.search(text) is not None


def _matcher(a: str, b: str) -> difflib.SequenceMatcher[str]:
    """A matcher over characters, or over tokens once the text gets long.

    ``autojunk`` is off in both cases and must stay off: it engages on any
    sequence of 200 or more elements and marks anything appearing in over 1%
    of positions as noise, which on character sequences silently returns a
    different number. Measured on one 4000-character pair: 0.5375 with it off,
    0.4630 with it on. Left on, a threshold means two different things above
    and below 200 characters.
    """
    if max(len(a), len(b)) > CHAR_LEVEL_MAX_LEN:
        return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False)
    return difflib.SequenceMatcher(None, a, b, autojunk=False)


def similarity(a: str, b: str) -> float:
    """Similarity of two already-normalized strings, in ``[0.0, 1.0]``.

    Returns 0.0 rather than the true ratio for pairs below
    :data:`ADVISORY_RATIO`. ``real_quick_ratio`` and ``quick_ratio`` are exact
    upper bounds on ``ratio``, so short-circuiting on them cannot discard a
    pair that would have scored above the floor.
    """
    if not a or not b:
        return 1.0 if a == b else 0.0
    matcher = _matcher(a, b)
    if matcher.real_quick_ratio() < ADVISORY_RATIO:
        return 0.0
    if matcher.quick_ratio() < ADVISORY_RATIO:
        return 0.0
    return matcher.ratio()


def raw_similarity(a: str, b: str, *, floor: float = 0.0) -> float:
    """The true ratio, with the caller choosing where it stops being interesting.

    `similarity` short-circuits anything below :data:`ADVISORY_RATIO` to 0.0,
    which is sound for its callers — they only care whether a pair is close
    enough to merge or to mention — and silently useless to a caller working at
    a lower threshold, which then reads every pair as completely dissimilar.

    `floor` is not decoration. `ratio` is quadratic in the length of both
    strings, so a caller comparing every pair in a store pays it n² times: with
    two hundred entries a session's opening context took nineteen seconds to
    assemble. `quick_ratio` and `real_quick_ratio` are exact upper bounds, so
    stopping on them cannot discard a pair that would have scored above the
    floor — it only refuses to price the ones that were never candidates.
    """
    if not a or not b:
        return 1.0 if a == b else 0.0
    matcher = _matcher(a, b)
    if floor > 0.0:
        if matcher.real_quick_ratio() < floor:
            return 0.0
        if matcher.quick_ratio() < floor:
            return 0.0
    return matcher.ratio()


def may_merge(a: str, b: str) -> bool:
    """Whether two normalized strings are allowed to merge at all.

    Independent of the similarity score, and checked in addition to it. Each
    guard covers a class where near-identical characters mean a different fact:

    G1 length ratio — a merge only ever discards the *older* text, so the
    dangerous shape is a long, detailed entry being replaced by a terse
    restatement. Also a free prefilter.

    G2 numeric multiset — dates, versions, ports, quotas, counts, ticket ids.
    The highest-precision facts in memory and the ones where a silent overwrite
    is hardest to notice. Compared as a multiset, not positionally, so
    "begins 2026-03-05" and "begins on 2026-03-05" still merge.

    G3 negation parity — polarity inversion. Parity rather than absence, so two
    variants of "with no trailing summary" are not blocked by their own "no".
    """
    lo, hi = sorted((len(a), len(b)))
    if hi == 0 or lo / hi < MIN_LENGTH_RATIO:
        return False
    if numeric_tokens(a) != numeric_tokens(b):
        return False
    return has_negation(a) == has_negation(b)


def classify_entry(
    content: str,
    kind: MemoryKind,
    existing: Sequence[MemoryEntry],
) -> DuplicateVerdict:
    """Decide whether *content* duplicates something already stored.

    ``kind`` partitions the comparison: fuzzy matching never crosses it, since
    the same sentence filed as ``user`` and as ``feedback`` records different
    things (a standing preference versus the user having corrected the agent),
    and merging would silently reclassify one of them. An *exact* match under a
    different kind is still reported as an advisory, so the duplicate is
    visible without being acted on.
    """
    folded = normalize_content(content)
    advisories: list[tuple[str, str, float]] = []
    candidates: list[tuple[float, MemoryEntry]] = []

    for entry in existing:
        entry_folded = normalize_content(entry.content)
        if entry_folded == folded:
            if entry.kind == kind:
                # Identical after normalization: merging loses nothing.
                return DuplicateVerdict(action="merge", target_id=entry.id, score=1.0)
            advisories.append((entry.id, entry.content, 1.0))
            continue
        if entry.kind != kind:
            continue
        score = similarity(folded, entry_folded)
        if score < ADVISORY_RATIO:
            continue
        if score >= AUTO_MERGE_RATIO and may_merge(folded, entry_folded):
            candidates.append((score, entry))
        else:
            advisories.append((entry.id, entry.content, score))

    if candidates:
        # Highest score wins; oldest breaks a tie, so the merge target is
        # stable no matter what order the file happens to be in.
        candidates.sort(key=lambda c: (-c[0], c[1].created_at))
        score, target = candidates[0]
        advisories.extend((e.id, e.content, s) for s, e in candidates[1:])
        return DuplicateVerdict(
            action="merge",
            target_id=target.id,
            score=score,
            advisories=_rank(advisories),
        )

    if advisories:
        return DuplicateVerdict(
            action="advise",
            target_id=None,
            score=0.0,
            advisories=_rank(advisories),
        )

    return CREATE_VERDICT


def _rank(advisories: list[tuple[str, str, float]]) -> tuple[tuple[str, str, float], ...]:
    advisories.sort(key=lambda a: (-a[2], a[0]))
    return tuple(advisories[:MAX_ADVISORIES])


def merge_entry(
    older: MemoryEntry,
    newer_content: str,
    *,
    now: float,
    newer_key: str | None = None,
) -> MemoryEntry:
    """Fold a restatement into the entry it restates.

    Keeps ``id`` so a caller holding it can still update or delete the entry,
    and ``created_at`` so it keeps meaning "when this fact was first learned".

    An existing ``key`` is kept for the same reason as the id — something may
    already refer to it. But an entry written before keys existed has none, and
    a restatement that supplies one is the only chance it will ever get to
    acquire a readable handle, so in that case the new key is adopted.
    """
    update: dict[str, object] = {"content": newer_content, "updated_at": now}
    if older.key is None and newer_key:
        update["key"] = newer_key
    return older.model_copy(update=update)


def compact_entries(entries: Sequence[MemoryEntry]) -> tuple[list[MemoryEntry], int]:
    """Fold exactly-repeated entries already on disk. Returns ``(kept, dropped)``.

    Exact normalized equality only — deliberately no fuzzy matching. A fuzzy
    merge here would rewrite entries the user was never shown an approval
    prompt for, at file-rewrite time, with nothing to review. Exact folding
    loses nothing: the surviving text is byte-identical apart from case, width,
    and whitespace.
    """
    kept: list[MemoryEntry] = []
    positions: dict[tuple[str, str], int] = {}
    dropped = 0

    for entry in entries:
        key = (entry.kind, normalize_content(entry.content))
        position = positions.get(key)
        if position is None:
            positions[key] = len(kept)
            kept.append(entry)
            continue
        older = kept[position]
        stamp = max(
            entry.updated_at or entry.created_at,
            older.updated_at or older.created_at,
        )
        kept[position] = merge_entry(older, entry.content, now=stamp)
        dropped += 1

    return kept, dropped


# --------------------------------- summaries ---------------------------------
#
# Summary dedup asks a different question from entry dedup: not "is this the
# same fact" but "does this record supersede that one". Similarity is the wrong
# tool for it — too slow on 4000-character prose, and unable to separate two
# summaries of one session from two summaries of unrelated sessions in the same
# repository, which share just as much vocabulary.
#
# The duplication is structural instead, and exactly knowable. A session writes
# one summary per compaction plus one at session end, and compaction summaries
# are cumulative: SimpleCompaction feeds the previous summary back into the
# history it re-summarizes, so summary N+1 of a session subsumes summary N.


def _merge_summary(older: SessionSummary, newer: SessionSummary) -> SessionSummary:
    """Take the newer record's content under the older record's id.

    ``created_at`` comes from the newer record because it is rendered to the
    model alongside the text and must describe what is actually shown.
    """
    return older.model_copy(
        update={
            "summary": newer.summary,
            "trigger": newer.trigger,
            "work_dir": newer.work_dir,
            "created_at": newer.created_at,
        }
    )


def place_summary(
    items: Sequence[SessionSummary],
    incoming: SessionSummary,
    *,
    policy: SummaryPolicy = "supersede",
) -> SummaryPlacement:
    """Fit *incoming* into *items*, superseding this session's previous record.

    ``policy="skip_if_session_present"`` is for a degraded summary — the raw
    tail of the conversation, written when the summarizer was unavailable.
    Both paths record ``trigger="session_end"``, so nothing in the record
    distinguishes them, and superseding blindly would let a raw transcript tail
    overwrite a good summary of the same session.
    """
    kept = list(items)
    folded = normalize_content(incoming.summary)

    for existing in kept:
        if normalize_content(existing.summary) == folded:
            # Identical text, whichever session produced it: nothing to add.
            return SummaryPlacement(items=kept, stored=None, superseded_id=None)

    same_session = [i for i, s in enumerate(kept) if s.session_id == incoming.session_id]
    if same_session and policy == "skip_if_session_present":
        return SummaryPlacement(items=kept, stored=None, superseded_id=None)

    if same_session:
        # Collapse this session's records into their oldest — the id stays put
        # across repeated supersessions — and move the result to the tail,
        # since the injection reads the last few and position carries recency.
        older = kept[same_session[0]]
        merged = _merge_summary(older, incoming)
        for index in reversed(same_session):
            kept.pop(index)
        kept.append(merged)
        return SummaryPlacement(items=kept, stored=merged, superseded_id=older.id)

    kept.append(incoming)
    return SummaryPlacement(items=kept, stored=incoming, superseded_id=None)


def compact_summaries(items: Sequence[SessionSummary]) -> tuple[list[SessionSummary], int]:
    """Apply the same rules to summaries already on disk. ``(kept, dropped)``."""
    kept: list[SessionSummary] = []
    for item in items:
        placement = place_summary(kept, item)
        kept = placement.items
    return kept, len(items) - len(kept)
