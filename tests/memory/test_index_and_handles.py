"""Memory that is looked up rather than recited, and handles a model can use."""

from __future__ import annotations

import pytest

from kimi_cli.memory.dedup import merge_entry
from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.soul.dynamic_injections import cross_session_memory as csm


def _entry(kind: str, content: str, key: str | None = None) -> MemoryEntry:
    return MemoryEntry(kind=kind, scope="persistent", content=content, key=key)  # type: ignore[arg-type]


class TestHandles:
    def test_a_key_is_normalised_and_becomes_the_handle(self) -> None:
        e = _entry("project", "body", key="ACLS/Repo-Path")
        assert e.key == "acls/repo-path"
        assert e.handle == "acls/repo-path"

    def test_an_entry_without_a_key_falls_back_to_its_id(self) -> None:
        """Records written before keys existed still need something to address."""
        e = _entry("project", "body")
        assert e.handle == e.id[:8]

    @pytest.mark.parametrize("bad", ["has space", "UPPER/../x", "a" * 65, "/leading", "sl/as/hes"])
    def test_a_key_that_could_not_be_typed_back_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _entry("project", "body", key=bad)

    def test_an_empty_key_is_treated_as_absent(self) -> None:
        assert _entry("project", "body", key="   ").key is None


class TestKindSplit:
    def test_behaviour_is_stated_and_facts_are_indexed(self) -> None:
        """Nothing prompts a model to go and fetch "be careful".

        A preference only changes behaviour if it is in front of the model, so
        it is quoted in full. A project fact is the opposite — worth having
        available, wasteful in every unrelated conversation.
        """
        entries = [
            _entry("feedback", "Always verify the date before writing a report."),
            _entry("user", "The user is an insulin pump manufacturer."),
            _entry(
                "project",
                "acls lives at /Users/x/acls, not the copy under output/. "
                + "Further detail that only matters once you are working on it. " * 4
                + "TAIL-MARKER",
                key="acls/repo",
            ),
            _entry("reference", "Mailbox configured at ~/mail.env", key="mail/config"),
        ]

        out = csm._render(entries, [])

        assert "Always verify the date before writing a report." in out
        assert "The user is an insulin pump manufacturer." in out
        assert "acls lives at" in out, "the index still has to identify the fact"
        assert "TAIL-MARKER" not in out, "a long fact is summarised, not recited"
        assert "(acls/repo)" in out and "(mail/config)" in out

    def test_the_index_says_how_to_read_an_entry(self) -> None:
        out = csm._render([_entry("project", "a fact", key="p/one")], [])
        assert '"op": "get"' in out

    def test_a_long_fact_is_summarised_to_one_line(self) -> None:
        entry = _entry("project", "first line is the summary\n" + "detail\n" * 50, key="p/one")
        line = entry.render_index()
        assert "\n" not in line
        assert "first line is the summary" in line

    def test_sections_are_omitted_when_empty(self) -> None:
        assert "Recorded facts" not in csm._render([_entry("feedback", "x")], [])
        assert "Persistent memory" not in csm._render([_entry("project", "x")], [])


class TestBudget:
    def test_an_oversized_snapshot_is_cut_and_says_so(self) -> None:
        """Persistent memory has no cap of its own; it only ever grows."""
        entries = [_entry("project", f"fact number {i} " + "x" * 400) for i in range(200)]

        out = csm._render(entries, [])

        assert len(out) <= csm._SNAPSHOT_BUDGET_CHARS + 100
        assert "truncated" in out

    def test_behavioural_memory_survives_the_cut(self) -> None:
        """Losing an instruction changes how the agent works, silently.

        Losing a project fact only means it has to be fetched.
        """
        entries = [_entry("feedback", "NEVER force-push to main.")]
        entries += [_entry("project", f"fact {i} " + "x" * 400) for i in range(200)]

        out = csm._render(entries, [])

        assert "NEVER force-push to main." in out


class TestMergeKeepsHandles:
    def test_merging_keeps_the_existing_key(self) -> None:
        """Something may already refer to it — same reason the id is kept."""
        older = _entry("project", "old text", key="p/one")
        merged = merge_entry(older, "new text", now=1.0, newer_key="p/two")
        assert merged.key == "p/one"
        assert merged.id == older.id

    def test_a_restatement_can_give_an_unkeyed_entry_a_key(self) -> None:
        """Its only chance to acquire a readable handle."""
        older = _entry("project", "old text")
        merged = merge_entry(older, "new text", now=1.0, newer_key="p/two")
        assert merged.key == "p/two"
