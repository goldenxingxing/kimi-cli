"""The persistent-memory write path: merging, compaction, and the merge pin."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from kimi_cli.memory.entry import MemoryEntry, MemoryKind
from kimi_cli.memory.storage import read_entries, upsert_entry


def entry(content: str, *, kind: MemoryKind = "user") -> MemoryEntry:
    return MemoryEntry(kind=kind, scope="persistent", content=content)


def write_raw(path: Path, *entries: MemoryEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(e.model_dump_json() + "\n" for e in entries),
        encoding="utf-8",
    )


def test_first_write_creates_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"

    result = upsert_entry(path, entry("User prefers terse replies"))

    assert result.merged is False
    assert result.compacted == 0
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_repeating_a_fact_updates_it_in_place(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"
    first = upsert_entry(path, entry("User prefers terse replies")).entry

    result = upsert_entry(path, entry("User prefers terse replies."))

    assert result.merged is True
    assert result.entry.id == first.id
    assert result.replaced_content == "User prefers terse replies"
    assert result.entry.content == "User prefers terse replies."
    assert result.entry.updated_at is not None

    stored = read_entries(path)
    assert len(stored) == 1
    assert stored[0].id == first.id
    assert stored[0].content == "User prefers terse replies."


def test_a_different_fact_is_kept_separately(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"
    first = upsert_entry(path, entry("Merge freeze begins 2026-03-05")).entry

    result = upsert_entry(path, entry("Merge freeze begins 2026-04-05"))

    assert result.merged is False
    assert {e.id for e in read_entries(path)} == {first.id, result.entry.id}


def test_duplicates_already_on_disk_are_folded_by_the_next_write(tmp_path: Path) -> None:
    # A file as it would look after upgrading from a version with no dedup.
    path = tmp_path / "persistent.jsonl"
    older, newer = entry("User prefers terse replies"), entry("user prefers  terse replies")
    write_raw(path, older, newer)

    result = upsert_entry(path, entry("Deploy target is fly.io"))

    assert result.merged is False
    assert result.compacted == 1
    stored = read_entries(path)
    assert [e.id for e in stored] == [older.id, result.entry.id]
    assert stored[0].content == newer.content


def test_pinned_merge_target_that_vanished_degrades_to_a_create(tmp_path: Path) -> None:
    # Between classify and write, another writer can change the file. Landing
    # on a different entry than the one approved is the outcome to prevent.
    path = tmp_path / "persistent.jsonl"
    write_raw(path, entry("User prefers terse replies"))

    result = upsert_entry(
        path, entry("User prefers terse replies."), expect_target_id="does-not-exist"
    )

    assert result.merged is False
    assert len(read_entries(path)) == 2


def test_pinned_merge_target_pointing_elsewhere_never_merges(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"
    intended = entry("User prefers terse replies")
    unrelated = entry("Deploy target is fly.io")
    write_raw(path, intended, unrelated)

    result = upsert_entry(path, entry("User prefers terse replies."), expect_target_id=unrelated.id)

    assert result.merged is False
    assert read_entries(path)[1].content == "Deploy target is fly.io"


def test_dedup_can_be_bypassed(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"
    upsert_entry(path, entry("User prefers terse replies"), dedup=False)
    upsert_entry(path, entry("User prefers terse replies"), dedup=False)

    assert len(read_entries(path)) == 2


def test_a_rewrite_drops_lines_that_cannot_be_parsed(tmp_path: Path) -> None:
    # Deliberate: the append-only path used to preserve corrupt lines forever.
    # A merge rewrites the file, and an unreadable record cannot survive that.
    path = tmp_path / "persistent.jsonl"
    keep = entry("User prefers terse replies")
    path.write_text(keep.model_dump_json() + "\n{not json at all\n", encoding="utf-8")

    upsert_entry(path, entry("User prefers terse replies."))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == keep.id


def test_advisories_reach_the_caller(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"
    existing = upsert_entry(path, entry("Pipeline bugs tracked in Linear project INGEST")).entry

    result = upsert_entry(path, entry("Pipeline bugs tracked in Linear project EGRESS"))

    assert result.merged is False
    assert [a[0] for a in result.advisories] == [existing.id]
    assert len(read_entries(path)) == 2


def test_result_matches_what_is_actually_stored(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"
    upsert_entry(path, entry("User prefers terse replies"))

    result = upsert_entry(path, entry("User prefers terse replies!"))

    assert read_entries(path) == [result.entry]


def test_concurrent_writers_produce_one_entry_and_no_torn_lines(tmp_path: Path) -> None:
    path = tmp_path / "persistent.jsonl"
    barrier = threading.Barrier(4)

    def write() -> None:
        barrier.wait()
        upsert_entry(path, entry("User prefers terse replies"))

    threads = [threading.Thread(target=write) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)
    assert len(read_entries(path)) == 1
