"""Memory that is looked up rather than recited, and handles a model can use."""

from __future__ import annotations

import pytest

from kimi_cli.memory.dedup import merge_entry
from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.recent import SessionSummary
from kimi_cli.soul.dynamic_injections import cross_session_memory as csm


def _entry(kind: str, content: str, key: str | None = None) -> MemoryEntry:
    return MemoryEntry(kind=kind, scope="persistent", content=content, key=key)  # type: ignore[arg-type]


def _summaries(n: int) -> list[SessionSummary]:
    return [
        SessionSummary(
            session_id=f"s{i:04d}", trigger="compaction", summary=f"recap {i} " + "y" * 300
        )
        for i in range(n)
    ]


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
        assert "(acls/repo, " in out and "(mail/config, " in out

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


class TestStaleness:
    def test_the_index_dates_each_fact(self) -> None:
        """Project facts go stale, and two entries about the same thing are
        told apart by which is more recent."""
        import re

        line = _entry("project", "a fact", key="p/one").render_index()
        assert re.search(r"\(p/one, \d{4}-\d{2}-\d{2}\)", line)


class TestBudget:
    def test_an_oversized_section_is_cut_and_says_so(self) -> None:
        """Persistent memory has no cap of its own; it only ever grows."""
        entries = [_entry("project", f"fact number {i} " + "x" * 400) for i in range(200)]

        out = csm._render(entries, [])

        assert len(out) <= csm._INDEX_BUDGET_CHARS + 500
        assert "not shown" in out, "a silent cut reads as a complete store"
        assert "search" in out, "and it has to say how to reach the rest"

    def test_the_cut_says_how_many_were_left_out(self) -> None:
        """ "Some were omitted" and "1,879 were omitted" call for different behaviour."""
        entries = [_entry("project", f"fact number {i} " + "x" * 400) for i in range(200)]

        out = csm._render(entries, [])

        shown = sum(1 for i in range(200) if f"fact number {i} " in out)
        assert f"({200 - shown} older" in out, "the count has to match what was actually dropped"

    def test_the_newest_entries_are_the_ones_kept(self) -> None:
        """The store is append-only, so cutting from the head drops the newest.

        For behavioural memory that is the correction the user gave most
        recently — the single entry least safe to lose, and the first one a
        head-truncation discarded.
        """
        entries = [_entry("feedback", f"rule number {i} " + "x" * 400) for i in range(60)]

        out = csm._render(entries, [])

        assert "rule number 59 " in out, "the most recent instruction must survive"
        assert "rule number 0 " not in out, "and the oldest is what gives way"

    def test_one_section_growing_does_not_evict_another(self) -> None:
        """A shared pool would let whichever section grows keep what it takes."""
        entries = [_entry("feedback", "NEVER force-push to main.")]
        entries += [_entry("project", f"fact {i} " + "x" * 400) for i in range(200)]
        recents = _summaries(30)

        out = csm._render(entries, recents)

        assert "NEVER force-push to main." in out
        assert "Recent session summaries" in out, "recaps must not be squeezed out"

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


class TestAddressableFromWhatIsShown:
    """The snapshot shows `id[:8]`; update and delete used to demand the full id.

    So every handle the agent could see was one it could not act on: it could
    add and never amend. Six of thirty-two entries in a real store had the
    model writing "supersedes the older record" into the prose because of it.
    """

    def _store(self, tmp_path):
        from kimi_cli.memory.storage import upsert_entry

        path = tmp_path / "persistent.jsonl"
        entry = MemoryEntry(kind="project", scope="persistent", content="a fact")
        upsert_entry(path, entry)
        return path, entry

    def test_update_accepts_the_handle_the_snapshot_showed(self, tmp_path) -> None:
        from kimi_cli.memory.storage import update_entry

        path, entry = self._store(tmp_path)
        assert update_entry(path, entry.handle, "revised") is not None

    def test_delete_accepts_the_handle_the_snapshot_showed(self, tmp_path) -> None:
        from kimi_cli.memory.storage import delete_entry

        path, entry = self._store(tmp_path)
        assert delete_entry(path, entry.handle) is True

    def test_a_key_also_works(self, tmp_path) -> None:
        from kimi_cli.memory.storage import update_entry, upsert_entry

        path = tmp_path / "persistent.jsonl"
        upsert_entry(
            path,
            MemoryEntry(kind="project", scope="persistent", content="a", key="proj/a"),
        )
        assert update_entry(path, "proj/a", "revised") is not None

    def test_an_ambiguous_prefix_is_refused_rather_than_guessed(self) -> None:
        """Editing the wrong memory silently is worse than saying which matched."""
        from kimi_cli.memory.storage import AmbiguousHandleError, resolve_handle

        a = MemoryEntry(id="abcd1111" + "0" * 24, kind="project", scope="persistent", content="a")
        b = MemoryEntry(id="abcd2222" + "0" * 24, kind="project", scope="persistent", content="b")

        assert resolve_handle([a, b], a.id) is a
        with pytest.raises(AmbiguousHandleError):
            resolve_handle([a, b], "abcd")


class TestGroundTruth:
    def test_the_header_says_the_snapshot_is_established_fact(self) -> None:
        """Without it, an agent re-derives what is already in front of it."""
        out = csm._render([_entry("feedback", "x")], [])
        lowered = out.lower()
        assert "established fact" in lowered
        assert "do not re-derive" in lowered

    def test_the_header_still_gives_the_conversation_the_last_word(self) -> None:
        out = csm._render([_entry("feedback", "x")], [])
        assert "the conversation wins" in out.lower()


class TestTruncationDirection:
    def test_recaps_keep_the_newest_not_the_oldest(self) -> None:
        """They arrive oldest-first, so cutting from the end loses today's.

        The one section where the usual direction is exactly backwards.
        """
        recents = [
            SessionSummary(
                session_id=f"s{i:04d}", trigger="compaction", summary=f"RECAP-{i} " + "y" * 900
            )
            for i in range(10)
        ]

        out = csm._render([], recents)

        assert "RECAP-9" in out, "the newest recap must survive"
        assert "RECAP-0" not in out, "the oldest is what should go"

    def test_a_section_heading_is_never_what_gets_cut(self) -> None:
        recents = [
            SessionSummary(session_id=f"s{i:04d}", trigger="compaction", summary="y" * 2000)
            for i in range(10)
        ]

        out = csm._render([_entry("project", "x" * 5000)], recents)

        assert "## Recorded facts (index)" in out
        assert "## Recent session summaries" in out


class TestSuggestions:
    def test_suggestions_are_marked_as_undecided(self) -> None:
        """They were extracted, not approved. Reading them as fact is the one
        way this feature could do harm."""
        from kimi_cli.memory.candidates import MemoryCandidate

        out = csm._render([], [], [MemoryCandidate(kind="project", content="a guessed fact")])

        assert "not saved" in out.lower()
        assert "have not been approved" in out.lower()
        assert '"op": "promote"' in out

    def test_no_suggestions_means_no_section(self) -> None:
        assert "Suggested memories" not in csm._render([_entry("feedback", "x")], [], [])
