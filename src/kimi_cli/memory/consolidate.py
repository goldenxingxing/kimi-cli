"""Which behavioural entries a newer one has made redundant.

Behavioural memory is injected whether or not anyone asks for it, and the
budget holds roughly a hundred and twenty entries. Past that the oldest stop
arriving — silently, and a standing instruction that stops arriving stops being
followed. Retirement gives that an intentional form; this finds the entries
worth retiring.

Nothing here decides anything. It ranks pairs where a newer entry appears to
cover an older one and hands them to the user, because the cost of being wrong
is asymmetric: retiring a rule that still holds changes how the agent behaves
with no signal that anything happened, while leaving a stale one costs a line
of context. That asymmetry is why supersession is not detected with a threshold
and applied automatically, even though the machinery to do so is right here.

The similarity floor is deliberately below the one that merges duplicates
outright — two instructions about the same subject rarely share wording — and
the guards from `dedup` are reused unchanged, because the failure they prevent
is the same one: a precise instruction being displaced by a vaguer restatement.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from kimi_cli.memory.dedup import (
    MIN_LENGTH_RATIO,
    has_negation,
    normalize_content,
    numeric_tokens,
)
from kimi_cli.memory.entry import MemoryEntry

#: Chosen from the measured gap, not from intuition — and it had to be, because
#: token-level scores sit well below the character-level ones the merge
#: thresholds were tuned for, so carrying a number over from there rejects real
#: pairs. On eight hand-checked pairs in both languages:
#:
#: - rules that restate each other scored 0.545 to 0.743
#:   ("日报要简洁" against "日报要简洁，聚焦产出而不是过程" is the weakest)
#: - rules about different things scored 0.143 to 0.222
#:
#: 0.50 sits in the empty middle. The one unrelated pair that scores high —
#: "always run the migration" against "never run the migration", at 0.833 — is
#: not a threshold problem and is refused by the negation guard instead.
SUPERSEDED_RATIO = 0.50

#: Shared subject matter, as a Jaccard overlap of distinctive terms.
#:
#: Sequence similarity was calibrated on one-sentence rules and does not survive
#: contact with real entries: on a live store the two obviously-overlapping
#: versions of one procedure scored 0.198 against a 0.50 floor, and the detector
#: found nothing at all. Two long procedures about the same thing share their
#: vocabulary and little of their wording, which is what this measures instead —
#: the same pair scores 0.391 and 0.213, against 0.117 for the closest unrelated
#: pair.
#:
#: Calibrated on one cluster in one store, which is not enough to trust the
#: number; it is set where that cluster's gap is widest and should be revisited
#: against more stores before it is treated as settled.
TOPIC_OVERLAP_RATIO = 0.20

#: Wording by which a newer entry announces that it replaces something.
#:
#: A far better signal than similarity, and it was being ignored. An entry
#: reading "2026-05-06 起，TDI 改用 EWMA，不再维护 48 小时窗口" is not merely
#: similar to the entry about the 48-hour window — it says outright that it
#: supersedes it. Measured on an incremental corpus, every revision produced
#: wording of this kind, while similarity scored the pairs too low to pair and
#: the numeric guard vetoed them anyway.
#:
#: Announcement alone is not enough: an entry can announce a change to
#: something else entirely, so it still has to be about the same subject.
_SUPERSEDES = re.compile(
    r"不再|已改为|改为|改成|改用|替代|取代|自\s*\d|起改|废弃|不再使用"
    r"|no longer|replaced by|superseded|instead of|changed from|switched (?:to|from)"
    r"|moved to|renamed to|as of \d",
    re.I,
)


def announces_supersession(text: str) -> bool:
    """Whether *text* says it replaces something, rather than merely resembling it."""
    return _SUPERSEDES.search(text) is not None


#: Shared subject required of an entry that announces a replacement. Lower
#: than the silent case: the announcement carries most of the evidence, and
#: demanding the usual overlap on top of it rejects a terse revision of a long
#: original — which is the ordinary shape of "X 改为 Y".
_ANNOUNCED_OVERLAP = 0.08

#: Never propose retiring more than this at once. A long list invites approving
#: it wholesale, which is the outcome this whole design exists to prevent.
MAX_PROPOSALS = 5


#: Latin runs as words, CJK as single characters.
#:
#: Comparing character by character is quadratic in the length of both strings,
#: and this runs over every pair in the store on the way into a session:
#: measured on sixty entries of four hundred characters, 20.1 seconds. The same
#: pairs compared as token sequences take 15 ms. Splitting on whitespace alone
#: would buy that only for languages that use it — a Chinese rule is one token,
#: so every pair scores 0.0 or 1.0 — so CJK is split per character, which is
#: both meaningful there and short enough to stay cheap.
_TOKEN = re.compile(r"[\u4e00-\u9fff]|[^\s\u4e00-\u9fff]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _similarity(a: str, b: str) -> float:
    """How much of one instruction the other repeats, over tokens."""
    if not a or not b:
        return 1.0 if a == b else 0.0
    return difflib.SequenceMatcher(None, _tokens(a), _tokens(b), autojunk=False).ratio()


def _without_announcement(text: str) -> str:
    """*text* with its replacement wording removed, for the negation check.

    The two collide by construction. Announcing a replacement is usually done
    by negating what it replaces — "改用 EWMA，不再维护 48 小时窗口", "switched
    to AdaGrad, no longer uses SLSQP" — and the negation guard then reads that
    as an inversion and refuses the pair. Two of three real revisions were lost
    to exactly this.

    The guard is still worth having: it is what stops "always run the
    migration" being retired by "never run the migration". That pair announces
    nothing, so removing announcement wording leaves it untouched.
    """
    return _SUPERSEDES.sub(" ", text)


def _may_supersede(older: str, newer: str) -> bool:
    """Whether *newer* may be treated as replacing *older*, both normalized.

    `dedup.may_merge` answers the neighbouring question and answers it
    symmetrically, because a merge does not know which text will survive. Here
    it is known, and that changes one of the three guards.

    Length is directional: losing detail is the harm, so a newer entry that is
    substantially *shorter* is refused, while a newer entry that is longer is
    the ordinary shape of a rule being made more precise — and the symmetric
    guard rejected exactly that case.

    Negation parity is kept as a veto: an instruction must never be proposed
    for retirement by its own inversion, and no amount of shared subject makes
    that safe.

    Matching numbers is *not* a veto here, which is where this parts company
    with `dedup.may_merge`. That guard governs an automatic, destructive merge,
    where "under 80 characters" quietly replacing "under 100" is exactly the
    silent loss it exists to prevent. This governs a proposal the user reads,
    so the asymmetry runs the other way: a wrong veto hides a real duplicate
    forever, a wrong proposal costs one line to decline. Measured on a live
    store, the veto was the whole reason nothing was ever found — the two
    versions of one procedure differ in step numbers and section references,
    and were refused for it. The difference is reported instead.
    """
    if not older or not newer:
        return False
    if len(newer) < len(older) * MIN_LENGTH_RATIO:
        return False
    if announces_supersession(newer):
        return has_negation(_without_announcement(older)) == has_negation(
            _without_announcement(newer)
        )
    return has_negation(older) == has_negation(newer)


@dataclass(frozen=True, slots=True)
class Supersession:
    """A newer entry that appears to cover an older one."""

    older: MemoryEntry
    newer: MemoryEntry
    score: float
    numbers_differ: bool = False

    def render(self) -> str:
        caution = (
            " — their numbers differ, so check they are one rule and not two"
            if self.numbers_differ
            else ""
        )
        return (
            f"- {self.older.handle} appears superseded by {self.newer.handle} "
            f"({self.score:.2f}){caution}: {self.older.content[:80]}"
        )


def find_superseded(entries: Sequence[MemoryEntry]) -> list[Supersession]:
    """Pairs where a later behavioural entry seems to replace an earlier one.

    Ordered by how confident the match is. Only behavioural entries are
    considered: a project fact going stale costs a wrong lookup, while a stale
    instruction is followed.
    """
    live = [e for e in entries if e.is_behavioural and e.retired_at is None and e.content.strip()]
    found: list[Supersession] = []

    topics = [topic_terms(e.content) for e in live]

    for index, older in enumerate(live):
        best: Supersession | None = None
        for offset, newer in enumerate(live[index + 1 :]):
            if older.kind != newer.kind:
                continue
            a, b = normalize_content(older.content), normalize_content(newer.content)

            # Two ways of being the same rule, because entries come in two
            # shapes. A one-line instruction restated more precisely is caught
            # by sequence similarity; a multi-step procedure rewritten is not —
            # it keeps its subject and changes its wording, so it is caught by
            # how much vocabulary the two share.
            sequence = _similarity(a, b)
            ta, tb = topics[index], topics[index + 1 + offset]
            overlap = len(ta & tb) / len(ta | tb) if ta and tb else 0.0

            if not _may_supersede(a, b):
                continue

            # Three ways of being the same rule again, in descending order of
            # how much they are guessing. A newer entry that says it replaces
            # something, about the same subject, is stating the relationship
            # rather than resembling it — so it needs far less overlap than a
            # silent rewrite does.
            announced = announces_supersession(newer.content) and overlap >= _ANNOUNCED_OVERLAP
            score = max(sequence, overlap)
            if announced:
                score = max(score, 1.0 - (1.0 - overlap) / 4)
            elif sequence < SUPERSEDED_RATIO and overlap < TOPIC_OVERLAP_RATIO:
                continue
            if best is None or score > best.score:
                best = Supersession(
                    older=older,
                    newer=newer,
                    score=score,
                    numbers_differ=numeric_tokens(a) != numeric_tokens(b),
                )
        if best is not None:
            found.append(best)

    found.sort(key=lambda s: s.score, reverse=True)
    return found[:MAX_PROPOSALS]


#: What the injected behavioural section holds.
#:
#: Lives here rather than beside the renderer because the write path needs it
#: too: an entry's cost is only meaningful as a share of the space there is,
#: and the moment to say so is when it is being written.
BEHAVIOURAL_BUDGET_CHARS = 8_000

#: A behavioural entry above this is worth remarking on. Measured on a real
#: store: eleven entries averaging 576 characters filled 79% of the budget, and
#: three of them were versions of one procedure that already existed as a file.
#: At 150 characters the same budget holds fifty-odd entries instead of
#: fourteen, which is the difference between the ceiling being a concern and
#: being the concern.
LONG_ENTRY_CHARS = 400

#: Warn while there is still room to act.
#:
#: The pressure signal fired when the store had already outgrown the budget —
#: by which point entries were being dropped from every session. A store at 79%
#: is where consolidation is still cheap and nothing has been lost yet.
PRESSURE_WARN_AT = 0.75

#: How long a subject has to stay away before the rule about it is worth
#: asking about. Long enough that a quiet fortnight on one part of a project
#: means nothing, short enough to matter within the life of a store.
DORMANT_AFTER_DAYS = 90

#: A term this common says nothing about what a conversation was about.
_TOPIC_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "be",
        "not",
        "no",
        "do",
        "does",
        "use",
        "used",
        "using",
        "user",
        "should",
        "must",
        "always",
        "never",
        "when",
        "if",
        "it",
        "this",
        "that",
        "你",
        "我",
        "的",
        "了",
        "是",
        "在",
        "和",
        "要",
        "不",
        "有",
        "个",
        "这",
        "那",
        "会",
        "能",
        "就",
        "都",
        "把",
        "被",
        "给",
        "让",
        "用",
    }
)

_PUNCTUATION = ".,;:!?()[]{}\"'`，。；：！？（）【】「」、…—/\\"

#: Terms shared with a conversation before its subject counts as having come
#: up. One word in common is a coincidence; two is a topic.
_TOPIC_HITS = 2


def topic_terms(text: str) -> set[str]:
    """The distinctive terms of a piece of text, for judging what it is about.

    Deliberately crude — this decides which entries to *ask* about, never what
    to remove, so a wrong answer costs a question rather than a rule.
    """
    # Punctuation is stripped here and not in `_tokens`: supersession compares
    # two entries, where a trailing full stop lands on both sides and cancels,
    # while this compares an entry against a conversation, where "branch." and
    # "branch" are the difference between a rule looking live and looking
    # abandoned.
    terms = set()
    for token in _tokens(text.casefold()):
        term = token.strip(_PUNCTUATION)
        if len(term) > 1 and term not in _TOPIC_STOPWORDS:
            terms.add(term)
    return terms


def mark_relevant(
    entries: Sequence[MemoryEntry], conversation: str, *, now: float
) -> list[MemoryEntry]:
    """Stamp behavioural entries whose subject appears in *conversation*.

    Returns the entries that changed, so the caller can decide whether a write
    is worth it.
    """
    seen = topic_terms(conversation)
    if not seen:
        return []
    touched: list[MemoryEntry] = []
    for entry in entries:
        if not entry.is_behavioural or entry.retired_at is not None:
            continue
        if len(topic_terms(entry.content) & seen) >= _TOPIC_HITS:
            entry.last_relevant_at = now
            touched.append(entry)
    return touched


def find_dormant(
    entries: Sequence[MemoryEntry], *, now: float, after_days: int = DORMANT_AFTER_DAYS
) -> list[MemoryEntry]:
    """Behavioural entries whose subject has not come up in a long time.

    Oldest silence first. An entry never stamped at all falls back to when it
    was written, so a store that predates this measurement ages in rather than
    reporting everything as dormant on the first run.
    """
    cutoff = now - after_days * 86_400
    dormant = [
        e
        for e in entries
        if e.is_behavioural
        and e.retired_at is None
        and (e.last_relevant_at or e.updated_at or e.created_at) < cutoff
    ]
    dormant.sort(key=lambda e: e.last_relevant_at or e.updated_at or e.created_at)
    return dormant[:MAX_PROPOSALS]


def pressure(entries: Sequence[MemoryEntry], budget_chars: int) -> tuple[int, int]:
    """``(entries that fit, entries there are)`` for behavioural memory.

    The ceiling is invisible from inside a session — the model is shown what
    fits and has no way to know what did not — so it is measured here and
    reported to the user, who is the one who can do something about it.
    """
    live = [e for e in entries if e.is_behavioural and e.retired_at is None]
    used = 0
    fit = 0
    for entry in reversed(live):
        used += len(entry.render()) + 1
        if used > budget_chars:
            break
        fit += 1
    return fit, len(live)
