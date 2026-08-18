"""Retiring changes what is injected and nothing else."""

from __future__ import annotations

from pathlib import Path

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.storage import (
    read_entries,
    resolve_handle,
    set_retired,
    upsert_entry,
)


def _store(tmp_path: Path) -> Path:
    path = tmp_path / "persistent.jsonl"
    upsert_entry(
        path,
        MemoryEntry(
            kind="feedback",
            scope="persistent",
            content="Always target the 2024 API.",
            key="api/version",
        ),
    )
    return path


def test_a_retired_entry_stays_in_the_file(tmp_path: Path) -> None:
    """Deleting is for something wrong; this is for something merely over."""
    path = _store(tmp_path)

    set_retired(path, "api/version", retired=True)

    entries = read_entries(path)
    assert len(entries) == 1
    assert entries[0].content == "Always target the 2024 API."
    assert entries[0].retired_at is not None


def test_a_retired_entry_is_still_addressable(tmp_path: Path) -> None:
    """`search` and `get` are the only way back to it, so they have to work."""
    path = _store(tmp_path)
    set_retired(path, "api/version", retired=True)

    found = resolve_handle(read_entries(path), "api/version")

    assert found is not None
    assert found.retired_at is not None


def test_restoring_puts_it_back(tmp_path: Path) -> None:
    path = _store(tmp_path)
    set_retired(path, "api/version", retired=True)

    set_retired(path, "api/version", retired=False)

    assert read_entries(path)[0].retired_at is None


def test_an_unknown_handle_reports_rather_than_guesses(tmp_path: Path) -> None:
    path = _store(tmp_path)

    assert set_retired(path, "no/such-thing", retired=True) is None
