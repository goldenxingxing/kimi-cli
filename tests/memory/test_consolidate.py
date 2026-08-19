"""Finding what a newer instruction has replaced — and refusing to act on it.

Behavioural memory is injected unasked and the budget holds about a hundred and
twenty entries, so a store that only grows eventually drops the oldest without
saying so. Consolidation is the intended way out. What it must not become is a
threshold that retires instructions on its own: being wrong costs a rule the
user is relying on, silently, while being cautious costs a line of context.
"""

from __future__ import annotations

from kimi_cli.memory.consolidate import (
    MAX_PROPOSALS,
    SUPERSEDED_RATIO,
    find_superseded,
    pressure,
)
from kimi_cli.memory.entry import MemoryEntry


def _entry(kind: str, content: str, *, retired: bool = False) -> MemoryEntry:
    entry = MemoryEntry(kind=kind, scope="persistent", content=content)  # type: ignore[arg-type]
    if retired:
        entry.retired_at = 1_700_000_000.0
    return entry


class TestFindingSupersession:
    def test_a_restatement_of_the_same_rule_is_paired(self) -> None:
        entries = [
            _entry("feedback", "Never force-push to the main branch."),
            _entry("feedback", "Never force-push to the main branch, use --force-with-lease."),
        ]

        found = find_superseded(entries)

        assert len(found) == 1
        assert found[0].older is entries[0]
        assert found[0].newer is entries[1]

    def test_two_rules_about_different_things_are_not_paired(self) -> None:
        entries = [
            _entry("feedback", "Never force-push to the main branch."),
            _entry("feedback", "Write commit messages in the imperative mood."),
        ]

        assert find_superseded(entries) == []

    def test_the_newer_entry_is_the_survivor(self) -> None:
        """Order in the file is chronological, so position decides which is which."""
        entries = [
            _entry("feedback", "Deploy from the release branch only."),
            _entry("feedback", "Deploy from the release branch only, after CI is green."),
        ]

        assert find_superseded(entries)[0].newer.content.endswith("after CI is green.")

    def test_project_facts_are_left_alone(self) -> None:
        """A stale project fact costs a wrong lookup; a stale rule gets followed."""
        entries = [
            _entry("project", "The API base path is /v1."),
            _entry("project", "The API base path is /v1 for all public routes."),
        ]

        assert find_superseded(entries) == []

    def test_a_retired_entry_is_not_proposed_again(self) -> None:
        entries = [
            _entry("feedback", "Never force-push to the main branch.", retired=True),
            _entry("feedback", "Never force-push to the main branch, use --force-with-lease."),
        ]

        assert find_superseded(entries) == []

    def test_a_contradicting_pair_is_never_paired(self) -> None:
        """The guard that matters most: identical wording, opposite meaning.

        Character similarity cannot see the difference, and merging these would
        silently invert an instruction.
        """
        entries = [
            _entry("feedback", "Always run the migration before deploying."),
            _entry("feedback", "Never run the migration before deploying."),
        ]

        assert find_superseded(entries) == []

    def test_a_precise_rule_is_not_displaced_by_a_vague_one(self) -> None:
        """A merge discards the older text, so the dangerous direction is losing detail."""
        entries = [
            _entry(
                "feedback",
                "Never force-push to main; on shared branches use --force-with-lease "
                "and tell the channel first.",
            ),
            _entry("feedback", "Never force-push."),
        ]

        assert find_superseded(entries) == []

    def test_the_list_is_short_enough_to_read(self) -> None:
        """A long list gets approved wholesale, which is what this design prevents."""
        entries = [
            _entry("feedback", f"Rule {i}: keep the changelog entry under one line.")
            for i in range(40)
        ]

        assert len(find_superseded(entries)) <= MAX_PROPOSALS

    def test_the_floor_sits_below_the_merge_threshold(self) -> None:
        """Entries similar enough to merge outright never reach this code."""
        from kimi_cli.memory.dedup import AUTO_MERGE_RATIO

        assert SUPERSEDED_RATIO < AUTO_MERGE_RATIO


class TestPressure:
    def test_it_reports_what_did_not_fit(self) -> None:
        entries = [_entry("feedback", "x" * 200) for _ in range(50)]

        fit, held = pressure(entries, budget_chars=1_000)

        assert held == 50
        assert 0 < fit < 50

    def test_a_store_within_budget_reports_no_pressure(self) -> None:
        entries = [_entry("feedback", "Keep commits small.")]

        assert pressure(entries, budget_chars=8_000) == (1, 1)

    def test_retired_entries_do_not_count_against_the_budget(self) -> None:
        """Retiring is how the ceiling is relieved, so it has to show up here."""
        entries = [_entry("feedback", "x" * 200, retired=True) for _ in range(50)]
        entries.append(_entry("feedback", "Keep commits small."))

        assert pressure(entries, budget_chars=8_000) == (1, 1)


class TestTheThresholdSeparatesRealCases:
    """The number was measured, so the measurements are kept.

    Token-level similarity scores markedly lower than the character-level
    scores the merge thresholds were tuned against, so a threshold carried over
    from there silently rejects every real pair — which is how this was first
    written.
    """

    def _score(self, a: str, b: str) -> float:
        from kimi_cli.memory.consolidate import _similarity
        from kimi_cli.memory.dedup import normalize_content

        return _similarity(normalize_content(a), normalize_content(b))

    def test_restatements_land_above_it(self) -> None:
        pairs = [
            (
                "Deploy from the release branch only.",
                "Deploy from the release branch only, after CI is green.",
            ),
            ("日报要简洁。", "日报要简洁，聚焦产出而不是过程。"),
            ("文档里不要出现代码函数名。", "文档里不要出现代码函数名，只保留功能化描述。"),
        ]

        for older, newer in pairs:
            assert self._score(older, newer) >= SUPERSEDED_RATIO, older

    def test_unrelated_rules_land_below_it(self) -> None:
        pairs = [
            (
                "Never force-push to the main branch.",
                "Write commit messages in the imperative mood.",
            ),
            ("文档里不要出现代码函数名。", "日报需要包含工作内容理解部分。"),
            ("Keep commits small.", "Keep the changelog under 80 characters."),
        ]

        for older, newer in pairs:
            assert self._score(older, newer) < SUPERSEDED_RATIO, older

    def test_chinese_is_compared_meaningfully(self) -> None:
        """Splitting on whitespace makes a Chinese rule one token and every score 0 or 1."""
        from kimi_cli.memory.consolidate import _tokens

        assert len(_tokens("日报要简洁")) == 5

    def test_the_comparison_stays_cheap_on_a_large_store(self) -> None:
        """This runs over every pair on the way into a session.

        Character-level comparison of sixty four-hundred-character entries took
        twenty seconds of a session's opening context.
        """
        import time

        entries = [_entry("feedback", f"rule number {i} " + "x" * 400) for i in range(60)]

        started = time.perf_counter()
        find_superseded(entries)
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0, f"took {elapsed:.1f}s to scan sixty entries"
