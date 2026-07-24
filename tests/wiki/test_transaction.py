from __future__ import annotations

import multiprocessing
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kimi_cli.wiki.initialize import ensure_wiki
from kimi_cli.wiki.models import PageChange, WikiPage
from kimi_cli.wiki.schema import parse_page, render_page
from kimi_cli.wiki.transaction import WikiConflictError, WikiTransaction


def _page(path: str, *, revision: int, body: str) -> WikiPage:
    now = datetime(2026, 7, 24, tzinfo=UTC)
    return WikiPage(
        logical_path=path,
        title="Atomic writes",
        created=now,
        updated=now,
        tags=["wiki"],
        sources=[],
        revision=revision,
        body=body,
    )


def _prepare_update(root: Path) -> tuple[WikiTransaction, Path]:
    layout = ensure_wiki(root)
    target = layout.root / "concepts" / "atomic-writes.md"
    target.write_text(
        render_page(_page("concepts/atomic-writes.md", revision=1, body="Before.\n")),
        encoding="utf-8",
    )
    transaction = WikiTransaction.prepare(
        layout=layout,
        changes=[
            PageChange(
                page=_page("concepts/atomic-writes.md", revision=2, body="After.\n"),
                expected_revision=1,
            )
        ],
        expected_global_revision=0,
        index_bytes=b"# Wiki Index\n\n- [[concepts/atomic-writes]]\n",
        log_bytes=b"# Wiki Log\n\n## [2026-07-24] remember | atomic writes\n",
    )
    return transaction, target


def _concurrent_commit(
    root: str,
    logical_path: str,
    barrier: Any,
    queue: multiprocessing.Queue[str],
) -> None:
    layout = ensure_wiki(Path(root))
    transaction = WikiTransaction.prepare(
        layout=layout,
        changes=[
            PageChange(
                page=_page(logical_path, revision=2, body=f"After {logical_path}.\n"),
                expected_revision=1,
            )
        ],
        expected_global_revision=0,
        index_bytes=b"# Wiki Index\n\nConcurrent result.\n",
        log_bytes=b"# Wiki Log\n\n## [2026-07-24] remember | concurrent\n",
    )
    barrier.wait(timeout=2)
    try:
        transaction.commit()
    except WikiConflictError:
        queue.put("conflict")
    else:
        queue.put("committed")


def _paused_commit(transaction: WikiTransaction, replaced: Any) -> None:
    import kimi_cli.wiki.transaction as transaction_module

    def pause(name: str) -> None:
        if name == "page_replace":
            replaced.set()
            time.sleep(0.2)

    transaction_module._hit_failpoint = pause
    transaction.commit()


def test_commit_replaces_pages_then_special_files_and_revision(tmp_path: Path) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")

    revision = transaction.commit()

    assert revision == 1
    assert parse_page(target.read_text(encoding="utf-8"), "concepts/atomic-writes.md").revision == 2
    assert transaction.layout.index.read_bytes().endswith(b"[[concepts/atomic-writes]]\n")
    assert transaction.layout.log.read_bytes().endswith(b"remember | atomic writes\n")
    assert transaction.layout.revision.read_text(encoding="ascii") == "1\n"


def test_commit_revalidates_global_revision_under_writer_lock(tmp_path: Path) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")
    transaction.layout.revision.write_text("1\n", encoding="ascii")
    before = target.read_bytes()

    with pytest.raises(WikiConflictError, match="global revision"):
        transaction.commit()

    assert target.read_bytes() == before
    assert not any(transaction.layout.metadata.joinpath("journal").iterdir())


def test_commit_revalidates_page_revision_under_writer_lock(tmp_path: Path) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")
    target.write_text(
        render_page(_page("concepts/atomic-writes.md", revision=2, body="Concurrent.\n")),
        encoding="utf-8",
    )
    before = target.read_bytes()

    with pytest.raises(WikiConflictError, match="page revision"):
        transaction.commit()

    assert target.read_bytes() == before
    assert not any(transaction.layout.metadata.joinpath("journal").iterdir())


def test_new_page_requires_revision_one_and_absence(tmp_path: Path) -> None:
    layout = ensure_wiki(tmp_path / "wiki")
    transaction = WikiTransaction.prepare(
        layout=layout,
        changes=[
            PageChange(
                page=_page("concepts/new.md", revision=1, body="New.\n"),
                expected_revision=None,
            )
        ],
        expected_global_revision=0,
        index_bytes=b"# Wiki Index\n",
        log_bytes=b"# Wiki Log\n",
    )

    assert transaction.commit() == 1

    with pytest.raises(WikiConflictError, match="already exists"):
        transaction.commit()


def test_prepare_rejects_non_monotonic_page_revision(tmp_path: Path) -> None:
    layout = ensure_wiki(tmp_path / "wiki")

    with pytest.raises(ValueError, match="increment"):
        WikiTransaction.prepare(
            layout=layout,
            changes=[
                PageChange(
                    page=_page("concepts/atomic-writes.md", revision=3, body="After.\n"),
                    expected_revision=1,
                )
            ],
            expected_global_revision=0,
            index_bytes=b"# Wiki Index\n",
            log_bytes=b"# Wiki Log\n",
        )


@pytest.mark.parametrize("revision", [False, 0.0, 1.0])
def test_expected_global_revision_is_a_strict_integer(
    tmp_path: Path,
    revision: object,
) -> None:
    layout = ensure_wiki(tmp_path / "wiki")

    with pytest.raises(ValueError, match="non-negative integer"):
        WikiTransaction.prepare(
            layout=layout,
            changes=[
                PageChange(
                    page=_page("concepts/new.md", revision=1, body="New.\n"),
                    expected_revision=None,
                )
            ],
            expected_global_revision=revision,  # type: ignore[arg-type]
            index_bytes=b"# Wiki Index\n",
            log_bytes=b"# Wiki Log\n",
        )


def test_journal_contains_only_relative_managed_targets(tmp_path: Path, monkeypatch) -> None:
    import kimi_cli.wiki.transaction as transaction_module

    transaction, _ = _prepare_update(tmp_path / "wiki")

    def fail_after_journal(name: str) -> None:
        if name == "journal_fsync":
            raise OSError("injected")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", fail_after_journal)
    with pytest.raises(OSError, match="injected"):
        transaction.commit()

    records = list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))
    assert len(records) == 1
    text = records[0].read_text(encoding="utf-8")
    assert str(transaction.layout.root) not in text
    assert '"target":"concepts/atomic-writes.md"' in text
    assert '"target":"index.md"' in text
    assert '"target":".openkimo/revision"' in text


def test_commit_exposes_all_durable_boundary_failpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kimi_cli.wiki.transaction as transaction_module

    transaction, _ = _prepare_update(tmp_path / "wiki")
    observed: list[str] = []
    monkeypatch.setattr(transaction_module, "_hit_failpoint", observed.append)

    transaction.commit()

    assert {
        "journal_directory_fsync",
        "artifact_create",
        "artifact_fsync",
        "artifact_directory_fsync",
        "prepared_record_temp_create",
        "prepared_record_temp_write",
        "prepared_record_temp_fsync",
        "prepared_record_pre_replace",
        "prepared_record_replace",
        "prepared_record_directory_fsync",
        "page_temp_create",
        "page_temp_write",
        "page_temp_fsync",
        "page_pre_replace",
        "page_replace",
        "index_temp_create",
        "index_temp_write",
        "index_temp_fsync",
        "index_pre_replace",
        "index_replace",
        "log_temp_create",
        "log_temp_write",
        "log_temp_fsync",
        "log_pre_replace",
        "log_replace",
        "revision_temp_create",
        "revision_temp_write",
        "revision_temp_fsync",
        "revision_pre_replace",
        "revision_replace",
        "commit_record_temp_create",
        "commit_record_temp_write",
        "commit_record_temp_fsync",
        "commit_record_pre_replace",
        "commit_record_replace",
        "commit_record_directory_fsync",
        "reindex_marker_temp_create",
        "reindex_marker_temp_write",
        "reindex_marker_temp_fsync",
        "reindex_marker_pre_replace",
        "reindex_marker_replace",
        "reindex_marker_directory_fsync",
        "journal_cleanup_pre_delete",
        "journal_cleanup_delete",
        "journal_cleanup_directory_fsync",
    } <= set(observed)


def test_concurrent_writers_serialize_and_one_detects_conflict(tmp_path: Path) -> None:
    layout = ensure_wiki(tmp_path / "wiki")
    logical_paths = ("concepts/first.md", "concepts/second.md")
    for logical_path in logical_paths:
        (layout.root / logical_path).write_text(
            render_page(_page(logical_path, revision=1, body="Before.\n")),
            encoding="utf-8",
        )
    barrier = multiprocessing.Barrier(2)
    queue: multiprocessing.Queue[str] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_concurrent_commit,
            args=(str(layout.root), logical_path, barrier, queue),
        )
        for logical_path in logical_paths
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(5)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(queue.get(timeout=1) for _ in processes) == ["committed", "conflict"]
    assert layout.revision.read_text(encoding="ascii") == "1\n"
    revisions = [
        parse_page(
            (layout.root / logical_path).read_text(encoding="utf-8"),
            logical_path,
        ).revision
        for logical_path in logical_paths
    ]
    assert sorted(revisions) == [1, 2]


def test_reader_waits_for_whole_commit_not_intermediate_page_state(tmp_path: Path) -> None:
    from kimi_cli.wiki.transaction import wiki_read_lock

    transaction, target = _prepare_update(tmp_path / "wiki")
    replaced = multiprocessing.Event()
    writer = multiprocessing.Process(target=_paused_commit, args=(transaction, replaced))
    writer.start()
    assert replaced.wait(timeout=2)

    with wiki_read_lock(transaction.layout, timeout=2):
        observed = (
            target.read_text(encoding="utf-8"),
            transaction.layout.index.read_bytes(),
            transaction.layout.log.read_bytes(),
            transaction.layout.revision.read_bytes(),
        )
    writer.join(5)

    assert writer.exitcode == 0
    assert observed[0].endswith("After.\n")
    assert observed[1].endswith(b"[[concepts/atomic-writes]]\n")
    assert observed[2].endswith(b"remember | atomic writes\n")
    assert observed[3] == b"1\n"
