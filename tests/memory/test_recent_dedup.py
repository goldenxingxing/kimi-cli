"""Session-summary supersession: one record per session, newest content."""

from __future__ import annotations

import time
from pathlib import Path

from kimi_cli.memory.recent import (
    DEFAULT_MAX_SUMMARIES,
    SessionSummary,
    SummaryTrigger,
    append_summary,
    read_recent_summaries,
)


def summary(
    session_id: str,
    text: str,
    *,
    trigger: SummaryTrigger = "compaction",
    created_at: float | None = None,
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        trigger=trigger,
        summary=text,
        created_at=created_at if created_at is not None else time.time(),
    )


def test_session_end_supersedes_the_sessions_compaction_summary(tmp_path: Path) -> None:
    # Compaction summaries are cumulative — the later one already contains the
    # earlier one — so keeping both spends the injection window twice.
    path = tmp_path / "recent.jsonl"
    first = summary("session-a", "Explored the auth module", created_at=100.0)
    append_summary(path, first)

    result = append_summary(
        path,
        summary(
            "session-a",
            "Explored auth, then fixed the token refresh",
            trigger="session_end",
            created_at=200.0,
        ),
    )

    stored = read_recent_summaries(path)
    assert len(stored) == 1
    assert result.superseded_id == first.id
    assert stored[0].id == first.id
    assert stored[0].summary == "Explored auth, then fixed the token refresh"
    assert stored[0].trigger == "session_end"
    assert stored[0].created_at == 200.0


def test_a_long_session_still_occupies_one_slot(tmp_path: Path) -> None:
    path = tmp_path / "recent.jsonl"
    for index in range(3):
        append_summary(path, summary("session-a", f"Round {index} of the work"))

    stored = read_recent_summaries(path)
    assert len(stored) == 1
    assert stored[0].summary == "Round 2 of the work"


def test_the_surviving_record_moves_to_the_tail(tmp_path: Path) -> None:
    # The injection reads only the last few summaries, so a session that is
    # still being written must not stay stuck behind newer ones.
    path = tmp_path / "recent.jsonl"
    append_summary(path, summary("session-a", "Older session work"))
    append_summary(path, summary("session-b", "Some other session"))

    append_summary(path, summary("session-a", "Older session work, continued"))

    assert [s.session_id for s in read_recent_summaries(path)] == ["session-b", "session-a"]


def test_different_sessions_with_similar_text_are_both_kept(tmp_path: Path) -> None:
    # Two runs of the same task in the same repo are two distinct events. The
    # FIFO cap already bounds growth, so there is nothing to gain by merging.
    path = tmp_path / "recent.jsonl"
    append_summary(path, summary("session-a", "Fixed the flaky retry test in the ingest pipeline"))
    append_summary(path, summary("session-b", "Fixed the flaky retry test in the ingest pipelines"))

    assert len(read_recent_summaries(path)) == 2


def test_byte_identical_text_from_another_session_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "recent.jsonl"
    append_summary(path, summary("session-a", "Nothing happened"))

    result = append_summary(path, summary("session-b", "nothing   happened"))

    assert result.stored is None
    assert len(read_recent_summaries(path)) == 1


def test_a_degraded_summary_never_overwrites_a_real_one(tmp_path: Path) -> None:
    # The raw conversation tail written when the summarizer is unavailable
    # records trigger="session_end" just like a real summary does, so nothing
    # in the record itself distinguishes them.
    path = tmp_path / "recent.jsonl"
    good = summary("session-a", "A careful summary of the whole session")
    append_summary(path, good)

    result = append_summary(
        path,
        summary("session-a", "user: hey\nassistant: hi", trigger="session_end"),
        policy="skip_if_session_present",
    )

    assert result.stored is None
    stored = read_recent_summaries(path)
    assert len(stored) == 1
    assert stored[0].summary == good.summary


def test_a_degraded_summary_is_kept_when_it_is_all_there_is(tmp_path: Path) -> None:
    path = tmp_path / "recent.jsonl"

    result = append_summary(
        path,
        summary("session-a", "user: hey\nassistant: hi", trigger="session_end"),
        policy="skip_if_session_present",
    )

    assert result.stored is not None
    assert len(read_recent_summaries(path)) == 1


def test_the_fifo_cap_still_applies(tmp_path: Path) -> None:
    path = tmp_path / "recent.jsonl"
    for index in range(DEFAULT_MAX_SUMMARIES + 5):
        append_summary(path, summary(f"session-{index}", f"Work item {index}"))

    stored = read_recent_summaries(path)
    assert len(stored) == DEFAULT_MAX_SUMMARIES
    assert stored[0].session_id == "session-5"


def test_dedup_runs_before_the_trim_so_nothing_is_over_trimmed(tmp_path: Path) -> None:
    path = tmp_path / "recent.jsonl"
    for index in range(DEFAULT_MAX_SUMMARIES):
        append_summary(path, summary(f"session-{index}", f"Work item {index}"))

    # A repeat of an existing session collapses rather than pushing one out.
    append_summary(path, summary("session-0", "Work item 0, revisited"))

    stored = read_recent_summaries(path)
    assert len(stored) == DEFAULT_MAX_SUMMARIES
    assert {s.session_id for s in stored} == {f"session-{i}" for i in range(DEFAULT_MAX_SUMMARIES)}


def test_appending_against_a_full_file_of_long_summaries_stays_fast(tmp_path: Path) -> None:
    # Guards against character-level similarity being reintroduced here: on
    # 20 summaries this size it would cost seconds, on the session-end path.
    path = tmp_path / "recent.jsonl"
    body = "the deploy runbook covers rollback and canary and escalation. " * 65
    for index in range(DEFAULT_MAX_SUMMARIES):
        append_summary(path, summary(f"session-{index}", f"{index} {body}"))

    started = time.perf_counter()
    append_summary(path, summary("session-new", f"new {body}"))
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert elapsed_ms < 200
