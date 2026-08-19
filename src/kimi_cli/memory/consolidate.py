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


def _may_supersede(older: str, newer: str) -> bool:
    """Whether *newer* may be treated as replacing *older*, both normalized.

    `dedup.may_merge` answers the neighbouring question and answers it
    symmetrically, because a merge does not know which text will survive. Here
    it is known, and that changes one of the three guards.

    Length is directional: losing detail is the harm, so a newer entry that is
    substantially *shorter* is refused, while a newer entry that is longer is
    the ordinary shape of a rule being made more precise — and the symmetric
    guard rejected exactly that case.

    The other two are kept as they are. Numbers must match, so "keep the
    changelog under 80 characters" never proposes retiring the version that
    said 100: those are two claims, and deciding between them is not a
    similarity question. Negation parity must match, so an instruction is never
    proposed for retirement by its own inversion.
    """
    if not older or not newer:
        return False
    if len(newer) < len(older) * MIN_LENGTH_RATIO:
        return False
    if numeric_tokens(older) != numeric_tokens(newer):
        return False
    return has_negation(older) == has_negation(newer)


@dataclass(frozen=True, slots=True)
class Supersession:
    """A newer entry that appears to cover an older one."""

    older: MemoryEntry
    newer: MemoryEntry
    score: float

    def render(self) -> str:
        return (
            f"- {self.older.handle} appears superseded by {self.newer.handle} "
            f"({self.score:.2f}): {self.older.content[:80]}"
        )


def find_superseded(entries: Sequence[MemoryEntry]) -> list[Supersession]:
    """Pairs where a later behavioural entry seems to replace an earlier one.

    Ordered by how confident the match is. Only behavioural entries are
    considered: a project fact going stale costs a wrong lookup, while a stale
    instruction is followed.
    """
    live = [e for e in entries if e.is_behavioural and e.retired_at is None and e.content.strip()]
    found: list[Supersession] = []

    for index, older in enumerate(live):
        best: Supersession | None = None
        for newer in live[index + 1 :]:
            if older.kind != newer.kind:
                continue
            a, b = normalize_content(older.content), normalize_content(newer.content)
            score = _similarity(a, b)
            if score < SUPERSEDED_RATIO or not _may_supersede(a, b):
                continue
            if best is None or score > best.score:
                best = Supersession(older=older, newer=newer, score=score)
        if best is not None:
            found.append(best)

    found.sort(key=lambda s: s.score, reverse=True)
    return found[:MAX_PROPOSALS]


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
