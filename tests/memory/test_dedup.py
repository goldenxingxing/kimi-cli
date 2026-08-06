"""The matching rules, which is where a bug costs the user a fact.

The false-merge cases assert *which* guard refused, not just that something
did. A test that only checks ``action != "merge"`` still passes if a guard is
removed and the threshold happens to cover the case that day.
"""

from __future__ import annotations

import difflib
import time

import pytest

from kimi_cli.memory.dedup import (
    ADVISORY_RATIO,
    AUTO_MERGE_RATIO,
    classify_entry,
    compact_entries,
    has_negation,
    may_merge,
    merge_entry,
    normalize_content,
    numeric_tokens,
    similarity,
)
from kimi_cli.memory.entry import MemoryEntry, MemoryKind
from kimi_cli.utils.string import fold_text
from kimi_cli.wiki.intent import normalize_intent_text


def entry(content: str, *, kind: MemoryKind = "user", created_at: float = 1000.0) -> MemoryEntry:
    return MemoryEntry(kind=kind, scope="persistent", content=content, created_at=created_at)


def verdict_for(new: str, *old: str, kind: MemoryKind = "user"):
    return classify_entry(new, kind, [entry(o, kind=kind) for o in old])


# --------------------------------- normalization ---------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ｒemember  THIS\n rule", "remember this rule"),
        ("记住 A/B", "记住 a/b"),
        ("  padded   out  ", "padded out"),
    ],
)
def test_normalization_folds_width_case_and_whitespace(raw: str, expected: str) -> None:
    assert normalize_content(raw) == expected


def test_normalization_unifies_typographic_apostrophes() -> None:
    # NFKC leaves U+2019 alone, so without this fold the two spellings of the
    # same word never match.
    assert normalize_content("Don’t mock the db") == normalize_content("Don't mock the db")


@pytest.mark.parametrize(
    "raw",
    ["Ｒemember  THIS\n rule", "记住 A/B", "plain text", "", "   "],
)
def test_fold_text_still_backs_the_wiki_normalizer(raw: str) -> None:
    # normalize_intent_text is an authorization boundary; extracting the shared
    # helper must not have changed what it folds.
    assert normalize_intent_text(raw) == fold_text(raw)


# ------------------------------- must merge -------------------------------


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "User prefers terse responses with no trailing summary",
            "User prefers terse responses, with no trailing summary.",
            id="punctuation",
        ),
        pytest.param(
            "user prefers terse  replies",
            "User prefers terse replies",
            id="case-and-whitespace",
        ),
        pytest.param(
            "Merge freeze begins 2026-03-05 for all repos",
            "Merge freeze begins on 2026-03-05 for all repos",
            id="inserted-preposition-same-date",
        ),
        pytest.param(
            "Oncall dashboard: grafana.internal/d/api",
            "Oncall dashboard is grafana.internal/d/api",
            id="rephrased-copula",
        ),
    ],
)
def test_restatements_merge(old: str, new: str) -> None:
    result = verdict_for(new, old)
    assert result.action == "merge"
    assert result.target_id is not None


def test_exact_match_after_normalization_scores_one() -> None:
    result = verdict_for("user prefers terse replies", "User Prefers  TERSE replies")
    assert result.action == "merge"
    assert result.score == 1.0


# --------------------------- must NOT merge (the guards) ---------------------------


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "Merge freeze begins 2026-03-05",
            "Merge freeze begins 2026-04-05",
            id="date",
        ),
        pytest.param("Retry budget is 3 attempts", "Retry budget is 5 attempts", id="count"),
        pytest.param("Pin lxml to v2.1.0", "Pin lxml to v2.1.1", id="version"),
    ],
)
def test_numeric_change_is_never_a_restatement(old: str, new: str) -> None:
    a, b = normalize_content(old), normalize_content(new)
    assert similarity(a, b) >= ADVISORY_RATIO
    assert numeric_tokens(a) != numeric_tokens(b)
    assert may_merge(a, b) is False
    assert verdict_for(new, old).action == "advise"


def test_the_threshold_alone_cannot_separate_a_correction_from_a_restatement() -> None:
    """Why the numeric guard has to exist.

    A changed date and an inserted preposition score identically. Whatever the
    threshold is set to, it either merges both or neither — only looking at
    *what* changed can tell them apart.
    """
    restated = similarity(
        normalize_content("Merge freeze begins 2026-03-05 for all repos"),
        normalize_content("Merge freeze begins on 2026-03-05 for all repos"),
    )
    corrected = similarity(
        normalize_content("Merge freeze begins 2026-03-05"),
        normalize_content("Merge freeze begins 2026-04-05"),
    )
    assert restated == pytest.approx(corrected, abs=0.01)
    assert restated >= AUTO_MERGE_RATIO
    assert corrected >= AUTO_MERGE_RATIO

    assert (
        verdict_for(
            "Merge freeze begins on 2026-03-05 for all repos",
            "Merge freeze begins 2026-03-05 for all repos",
        ).action
        == "merge"
    )
    assert (
        verdict_for("Merge freeze begins 2026-04-05", "Merge freeze begins 2026-03-05").action
        == "advise"
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "Don't mock the database in integration tests",
            "Mock the database in integration tests",
            id="english",
        ),
        pytest.param("提交前不要跑 lint", "提交前要跑 lint", id="chinese"),
    ],
)
def test_polarity_flip_is_never_a_restatement(old: str, new: str) -> None:
    a, b = normalize_content(old), normalize_content(new)
    assert similarity(a, b) >= ADVISORY_RATIO
    assert has_negation(a) != has_negation(b)
    assert may_merge(a, b) is False
    assert verdict_for(new, old).action == "advise"


def test_negation_parity_does_not_block_a_shared_negation() -> None:
    # Both sides contain "no", so parity holds and the merge proceeds. An
    # absence check rather than a parity check would wrongly refuse this.
    a = normalize_content("User prefers terse responses with no trailing summary")
    b = normalize_content("User prefers terse responses, with no trailing summary.")
    assert has_negation(a) and has_negation(b)
    assert may_merge(a, b) is True


def test_terse_restatement_of_a_detailed_fact_is_refused() -> None:
    detailed = (
        "The deploy runbook lives in ops/runbooks/deploy.md and covers rollback, "
        "canary, and the on-call escalation ladder"
    )
    terse = "Deploy runbook is in ops/"
    a, b = normalize_content(detailed), normalize_content(terse)
    # A merge only ever discards the older text, so this is the shape that
    # loses information.
    assert may_merge(a, b) is False
    assert verdict_for(terse, detailed).action == "create"

    stored = compact_entries([entry(detailed)])[0]
    assert stored[0].content == detailed


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "Pipeline bugs tracked in Linear project INGEST",
            "Pipeline bugs tracked in Linear project EGRESS",
            id="identifier",
        ),
        pytest.param("Use tabs for indentation", "Use spaces for indentation", id="tabs-spaces"),
        pytest.param(
            "Prefers responses in Chinese", "Prefers responses in Japanese", id="language"
        ),
        pytest.param("Use pytest for unit tests", "Use pytest for e2e tests", id="test-kind"),
    ],
)
def test_distinct_facts_below_the_threshold_are_not_merged(old: str, new: str) -> None:
    assert similarity(normalize_content(old), normalize_content(new)) < AUTO_MERGE_RATIO
    assert verdict_for(new, old).action != "merge"


def test_reordering_is_not_a_restatement() -> None:
    assert verdict_for("Deploy to prod before staging", "Deploy to staging before prod").action == (
        "create"
    )


def test_guards_and_similarity_are_symmetric() -> None:
    pairs = [
        ("Merge freeze begins 2026-03-05", "Merge freeze begins 2026-04-05"),
        ("Use tabs for indentation", "Use spaces for indentation"),
        ("User prefers terse replies", "User prefers terse replies."),
    ]
    for old, new in pairs:
        a, b = normalize_content(old), normalize_content(new)
        assert may_merge(a, b) == may_merge(b, a)
        assert similarity(a, b) == pytest.approx(similarity(b, a))


# ----------------------------------- kind -----------------------------------


def test_fuzzy_matching_never_crosses_kind() -> None:
    existing = [entry("User prefers terse responses with no trailing summary", kind="user")]
    result = classify_entry(
        "User prefers terse responses, with no trailing summary.", "feedback", existing
    )
    assert result.action != "merge"


def test_exact_match_under_another_kind_is_reported_not_merged() -> None:
    content = "User prefers terse replies"
    result = classify_entry(content, "feedback", [entry(content, kind="user")])
    assert result.action == "advise"
    assert result.target_id is None
    assert [(a[1], a[2]) for a in result.advisories] == [(content, 1.0)]


# --------------------------------- advisories ---------------------------------


def test_near_miss_is_reported_without_being_touched() -> None:
    old = "Pipeline bugs tracked in Linear project INGEST"
    result = verdict_for("Pipeline bugs tracked in Linear project EGRESS", old)
    assert result.action == "advise"
    assert result.target_id is None
    assert len(result.advisories) == 1
    _, content, score = result.advisories[0]
    assert content == old
    assert ADVISORY_RATIO <= score < AUTO_MERGE_RATIO


def test_unrelated_content_produces_no_advisories() -> None:
    result = verdict_for("Deploy target is fly.io", "User prefers terse replies")
    assert result.action == "create"
    assert result.advisories == ()


# ---------------------------- similarity mechanics ----------------------------


def test_long_text_is_compared_without_autojunk() -> None:
    # difflib turns autojunk on at 200 elements, which on characters marks every
    # common letter as noise. Left on, a one-typo difference in this pair scores
    # 0.04 instead of 0.999 and would never merge.
    base = normalize_content("the deploy runbook covers rollback and canary and escalation. " * 16)
    typo = base.replace("canary", "canery", 1)
    assert len(base) > 200

    assert similarity(base, typo) == pytest.approx(
        difflib.SequenceMatcher(None, base, typo, autojunk=False).ratio()
    )
    assert similarity(base, typo) != pytest.approx(
        difflib.SequenceMatcher(None, base, typo, autojunk=True).ratio()
    )


def test_long_text_comparison_stays_fast() -> None:
    # Character-level ratio() on a pair this size costs ~150ms, which would put
    # seconds onto a write that compares against a whole file.
    a = normalize_content("lorem ipsum dolor sit amet consectetur " * 120)
    b = a[:2500] + " changed " + a[2509:]

    started = time.perf_counter()
    score = similarity(a, b)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert score > AUTO_MERGE_RATIO
    assert elapsed_ms < 50


def test_dissimilar_pairs_short_circuit_to_zero() -> None:
    assert similarity("completely unrelated", "nothing alike at all here") == 0.0


# -------------------------------- compaction --------------------------------


def test_compaction_folds_exact_repeats_keeping_the_oldest_identity() -> None:
    first = entry("User prefers terse replies", created_at=100.0)
    second = entry("user prefers  TERSE replies", created_at=200.0)
    third = entry("User prefers terse replies.", created_at=300.0)

    kept, dropped = compact_entries([first, second, third])

    assert dropped == 1
    assert [e.id for e in kept] == [first.id, third.id]
    survivor = kept[0]
    assert survivor.created_at == 100.0
    assert survivor.content == second.content
    assert survivor.updated_at == 200.0


def test_compaction_never_fuzzy_merges() -> None:
    # A fuzzy merge here would rewrite entries nobody was shown a prompt for.
    pair = [
        entry("User prefers terse responses with no trailing summary"),
        entry("User prefers terse responses, with no trailing summary."),
    ]
    assert verdict_for(pair[1].content, pair[0].content).action == "merge"

    kept, dropped = compact_entries(pair)

    assert dropped == 0
    assert len(kept) == 2


def test_compaction_preserves_order() -> None:
    entries = [entry("alpha"), entry("beta"), entry("alpha "), entry("gamma")]
    kept, _ = compact_entries(entries)
    assert [e.content for e in kept] == ["alpha ", "beta", "gamma"]


def test_merge_keeps_identity_and_takes_the_new_wording() -> None:
    older = entry("User prefers terse replies", created_at=100.0)
    merged = merge_entry(older, "User prefers terse replies.", now=555.0)

    assert merged.id == older.id
    assert merged.kind == older.kind
    assert merged.created_at == 100.0
    assert merged.content == "User prefers terse replies."
    assert merged.updated_at == 555.0
