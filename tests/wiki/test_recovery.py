from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import kimi_cli.wiki.transaction as transaction_module
from kimi_cli.wiki.initialize import ensure_wiki
from kimi_cli.wiki.models import PageChange, WikiPage
from kimi_cli.wiki.schema import render_page
from kimi_cli.wiki.transaction import (
    WikiRecoveryRequired,
    WikiTransaction,
    recover_transactions,
    wiki_read_lock,
)


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


def _snapshot(transaction: transaction_module.WikiTransaction, target: Path) -> tuple[bytes, ...]:
    return (
        target.read_bytes(),
        transaction.layout.index.read_bytes(),
        transaction.layout.log.read_bytes(),
        transaction.layout.revision.read_bytes(),
    )


@pytest.mark.parametrize(
    "failpoint",
    ["journal_fsync", "page_replace", "index_replace", "log_replace", "revision_replace"],
)
def test_recovery_never_exposes_partial_commit(
    tmp_path: Path,
    failpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, target = _prepare_update(tmp_path / failpoint)
    before = _snapshot(transaction, target)

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="injected"):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    after = _snapshot(transaction, target)

    assert after == before or after[3] == b"1\n"
    if after != before:
        assert target.read_text(encoding="utf-8").endswith("After.\n")
        assert after[1].endswith(b"[[concepts/atomic-writes]]\n")
        assert after[2].endswith(b"remember | atomic writes\n")
    assert result.writes_quarantined is False


def test_partial_transaction_rolls_back_when_forward_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")
    before = _snapshot(transaction, target)

    def inject(name: str) -> None:
        if name == "page_replace":
            raise OSError("injected")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)
    next(transaction.layout.metadata.joinpath("journal").glob("*/new/0001")).unlink()

    result = recover_transactions(transaction.layout)

    assert _snapshot(transaction, target) == before
    assert result.rolled_back_transactions == 1


def test_committed_markdown_reports_that_search_needs_rebuild(
    tmp_path: Path,
) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")
    transaction.commit()

    result = recover_transactions(transaction.layout)

    assert target.read_text(encoding="utf-8").endswith("After.\n")
    assert result.needs_reindex is True


def test_unreadable_journal_quarantines_writes_but_not_reads(tmp_path: Path) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")
    journal = transaction.layout.metadata / "journal" / "broken"
    journal.mkdir()
    (journal / "record.json").write_bytes(b"{not-json")

    result = recover_transactions(transaction.layout)

    assert result.writes_quarantined is True
    assert target.read_text(encoding="utf-8").endswith("Before.\n")
    with pytest.raises(WikiRecoveryRequired, match="quarantined"):
        transaction.commit()


def test_journal_target_escape_is_quarantined_without_touching_outside(
    tmp_path: Path,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    outside = tmp_path / "outside.txt"
    outside.write_text("safe", encoding="utf-8")
    journal = transaction.layout.metadata / "journal" / "malicious"
    journal.mkdir()
    record = {
        "version": 1,
        "transaction_id": "malicious",
        "state": "prepared",
        "expected_global_revision": 0,
        "new_revision": 1,
        "targets": [
            {
                "kind": "page",
                "target": "../../outside.txt",
                "old_hash": None,
                "new_hash": "sha256:" + "0" * 64,
                "old_artifact": None,
                "new_artifact": "new/0000",
            }
        ],
    }
    (journal / "record.json").write_text(json.dumps(record), encoding="utf-8")

    result = recover_transactions(transaction.layout)

    assert result.writes_quarantined is True
    assert outside.read_text(encoding="utf-8") == "safe"


def test_shared_read_runs_recovery_before_yielding(tmp_path: Path, monkeypatch) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")

    def inject(name: str) -> None:
        if name == "index_replace":
            raise OSError("injected")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    with wiki_read_lock(transaction.layout, timeout=1.0) as result:
        snapshot = _snapshot(transaction, target)

    assert result.writes_quarantined is False
    assert snapshot[3] in {b"0\n", b"1\n"}
    if snapshot[3] == b"1\n":
        assert snapshot[0].decode().endswith("After.\n")
        assert snapshot[1].endswith(b"[[concepts/atomic-writes]]\n")
        assert snapshot[2].endswith(b"remember | atomic writes\n")


def test_recovery_io_failure_blocks_reader_instead_of_exposing_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")

    def inject(name: str) -> None:
        if name == "page_replace":
            raise OSError("initial interruption")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="initial interruption"):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    def cannot_recover(
        _target: Path,
        _data: bytes,
        *,
        boundary: str | None = None,
    ) -> None:
        del boundary
        raise OSError("disk remains unavailable")

    monkeypatch.setattr(transaction_module, "_durable_replace", cannot_recover)
    with (
        pytest.raises(OSError, match="disk remains unavailable"),
        wiki_read_lock(transaction.layout, timeout=1.0),
    ):
        pytest.fail("reader must not observe a partial valid transaction")


def test_successful_search_rebuild_acknowledges_committed_journal(tmp_path: Path) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()

    result = recover_transactions(transaction.layout)
    acknowledgement = transaction_module.acknowledge_reindex(
        transaction.layout,
        rebuilt_revision=1,
    )

    assert result.needs_reindex is True
    assert acknowledgement.acknowledged is True
    assert not transaction.layout.metadata.joinpath("search.invalid").exists()
    assert not list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))


def test_failed_search_rebuild_retains_only_independent_cache_marker(tmp_path: Path) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()

    with pytest.raises(OSError, match="sqlite unavailable"):
        raise OSError("sqlite unavailable")

    result = recover_transactions(transaction.layout)
    assert result.needs_reindex is True
    assert transaction.layout.metadata.joinpath("search.invalid").is_file()
    assert not list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))


def test_authoritative_journal_cleanup_is_independent_of_cache_rebuild(
    tmp_path: Path,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    assert transaction.commit() == 1

    result = recover_transactions(transaction.layout)

    assert result.needs_reindex is True
    assert result.required_reindex_revision == 1
    assert not list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))
    assert transaction.layout.metadata.joinpath("search.invalid").is_file()


def test_rebuild_failure_does_not_quarantine_or_block_a_later_commit(
    tmp_path: Path,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()
    recover_transactions(transaction.layout)

    with pytest.raises(OSError, match="sqlite unavailable"):
        raise OSError("sqlite unavailable")

    second = WikiTransaction.prepare(
        layout=transaction.layout,
        changes=[
            PageChange(
                page=_page("concepts/second.md", revision=1, body="Second.\n"),
                expected_revision=None,
            )
        ],
        expected_global_revision=1,
        index_bytes=b"# Wiki Index\n\nSecond.\n",
        log_bytes=b"# Wiki Log\n\n## [2026-07-24] remember | second\n",
    )
    assert second.commit() == 2
    restarted = recover_transactions(transaction.layout)
    assert restarted.writes_quarantined is False
    assert restarted.required_reindex_revision == 2


def test_unknown_current_hash_is_preserved_and_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")

    def inject(name: str) -> None:
        if name == "page_replace":
            raise OSError("injected")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)
    unknown = b"# Human recovery note\n\nDo not overwrite.\n"
    transaction.layout.index.write_bytes(unknown)

    result = recover_transactions(transaction.layout)

    assert result.writes_quarantined is True
    assert transaction.layout.index.read_bytes() == unknown


def test_roll_forward_reads_each_verified_artifact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")

    def inject(name: str) -> None:
        if name == "page_replace":
            raise OSError("injected")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)
    original = transaction_module._read_optional_artifact
    reads: dict[Path, int] = {}

    def read_once(path: Path) -> bytes | None:
        reads[path] = reads.get(path, 0) + 1
        if reads[path] > 1:
            raise AssertionError(f"artifact read twice: {path.name}")
        return original(path)

    monkeypatch.setattr(transaction_module, "_read_optional_artifact", read_once)
    result = recover_transactions(transaction.layout)

    assert result.writes_quarantined is False
    assert reads and max(reads.values()) == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("expected_page_revision", 2),
        ("page_revision", 99),
        ("revision_artifact", 99),
    ],
)
def test_recovery_validates_recorded_page_and_revision_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: int,
) -> None:
    transaction, _ = _prepare_update(tmp_path / field)

    def inject(name: str) -> None:
        if name == "journal_fsync":
            raise OSError("injected")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)
    record_path = next(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    journal = record_path.parent
    if field == "expected_page_revision":
        record["targets"][0]["expected_page_revision"] = replacement
    elif field == "page_revision":
        artifact = journal / record["targets"][0]["new_artifact"]
        changed = _page("concepts/atomic-writes.md", revision=replacement, body="After.\n")
        data = render_page(changed).encode()
        artifact.write_bytes(data)
        record["targets"][0]["new_hash"] = transaction_module.content_hash(data)
    else:
        artifact = journal / record["targets"][-1]["new_artifact"]
        data = f"{replacement}\n".encode()
        artifact.write_bytes(data)
        record["targets"][-1]["new_hash"] = transaction_module.content_hash(data)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    result = recover_transactions(transaction.layout)

    assert result.writes_quarantined is True


def test_reindex_ack_failure_propagates_and_marker_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()
    recover_transactions(transaction.layout)

    def fail_remove(_path: Path) -> None:
        raise OSError("ack delete failed")

    monkeypatch.setattr(transaction_module, "_remove_reindex_marker", fail_remove)
    with pytest.raises(OSError, match="ack delete failed"):
        transaction_module.acknowledge_reindex(
            transaction.layout,
            rebuilt_revision=1,
        )

    assert transaction.layout.metadata.joinpath("search.invalid").is_file()
    assert recover_transactions(transaction.layout).writes_quarantined is False


@pytest.mark.parametrize(
    "failpoint",
    [
        "journal_directory_fsync",
        "artifact_create",
        "artifact_fsync",
        "artifact_directory_fsync",
        "prepared_record_replace",
        "prepared_record_directory_fsync",
        "journal_fsync",
    ],
)
def test_precommit_durability_failpoints_recover_without_authority_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    transaction, target = _prepare_update(tmp_path / failpoint)
    before = _snapshot(transaction, target)

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="injected"):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.writes_quarantined is False
    assert _snapshot(transaction, target) == before


@pytest.mark.parametrize(
    "failpoint",
    ["commit_record_replace", "commit_record_directory_fsync"],
)
def test_commit_marker_failpoints_recover_complete_new_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    transaction, target = _prepare_update(tmp_path / failpoint)

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="injected"):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.writes_quarantined is False
    assert target.read_text(encoding="utf-8").endswith("After.\n")
    assert transaction.layout.revision.read_bytes() == b"1\n"


def test_rollback_failure_is_retryable_without_exposing_unknown_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")
    before = _snapshot(transaction, target)

    def stop_after_page(name: str) -> None:
        if name == "page_replace":
            raise OSError("initial interruption")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", stop_after_page)
    with pytest.raises(OSError):
        transaction.commit()
    journal = next(transaction.layout.metadata.joinpath("journal").glob("*"))
    (journal / "new" / "0001").unlink()

    def stop_rollback(name: str) -> None:
        if name == "rollback_replace":
            raise OSError("rollback interrupted")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", stop_rollback)
    with pytest.raises(OSError, match="rollback interrupted"):
        recover_transactions(transaction.layout)
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.rolled_back_transactions == 1
    assert _snapshot(transaction, target) == before


def test_cache_marker_failure_keeps_committed_journal_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")

    def inject(name: str) -> None:
        if name == "reindex_marker_replace":
            raise OSError("marker unavailable")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    assert transaction.commit() == 1
    assert list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.writes_quarantined is False
    assert result.required_reindex_revision == 1
    assert target.read_text(encoding="utf-8").endswith("After.\n")
    assert not list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))


def test_committed_authority_does_not_depend_on_disposable_journal_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")

    def inject(name: str) -> None:
        if name == "reindex_marker_replace":
            raise OSError("marker unavailable")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    transaction.commit()
    journal = next(transaction.layout.metadata.joinpath("journal").glob("*"))
    (journal / "new" / "0000").write_bytes(b"corrupt disposable copy")
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)

    assert result.writes_quarantined is False
    assert result.required_reindex_revision == 1
    assert target.read_text(encoding="utf-8").endswith("After.\n")
    assert not journal.exists()


def test_reindex_ack_rejects_unreadable_marker_without_deleting_it(
    tmp_path: Path,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()
    marker = transaction.layout.metadata / "search.invalid"
    marker.write_bytes(b"{invalid")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        transaction_module.acknowledge_reindex(
            transaction.layout,
            rebuilt_revision=1,
        )

    assert marker.read_bytes() == b"{invalid"


def test_reindex_ack_directory_fsync_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()

    def inject(name: str) -> None:
        if name == "reindex_ack_directory_fsync":
            raise OSError("ack directory fsync failed")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="ack directory fsync failed"):
        transaction_module.acknowledge_reindex(
            transaction.layout,
            rebuilt_revision=1,
        )


def test_reindex_ack_does_not_clear_a_newer_revision_marker(tmp_path: Path) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()
    second = WikiTransaction.prepare(
        layout=transaction.layout,
        changes=[
            PageChange(
                page=_page("concepts/second.md", revision=1, body="Second.\n"),
                expected_revision=None,
            )
        ],
        expected_global_revision=1,
        index_bytes=b"# Wiki Index\n\nSecond.\n",
        log_bytes=b"# Wiki Log\n\n## [2026-07-24] remember | second\n",
    )
    second.commit()

    acknowledgement = transaction_module.acknowledge_reindex(
        transaction.layout,
        rebuilt_revision=1,
    )

    assert acknowledgement.acknowledged is False
    assert acknowledgement.required_revision == 2
    assert recover_transactions(transaction.layout).required_reindex_revision == 2


def test_pending_post_commit_cleanup_blocks_writes_but_not_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")

    def fail_cleanup(name: str) -> None:
        if name == "journal_cleanup_pre_delete":
            raise OSError("cleanup temporarily unavailable")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", fail_cleanup)
    assert transaction.commit() == 1
    with wiki_read_lock(transaction.layout) as read_result:
        assert target.read_text(encoding="utf-8").endswith("After.\n")
    assert read_result.post_commit_cleanup_pending == 1
    assert read_result.writes_quarantined is False

    with pytest.raises(WikiRecoveryRequired, match="cleanup") as error:
        WikiTransaction.prepare(
            layout=transaction.layout,
            changes=[
                PageChange(
                    page=_page("concepts/second.md", revision=1, body="Second.\n"),
                    expected_revision=None,
                )
            ],
            expected_global_revision=1,
            index_bytes=b"# Wiki Index\n\nSecond.\n",
            log_bytes=b"# Wiki Log\n\n## [2026-07-24] remember | second\n",
        )
    assert error.value.retryable is True
    assert "recover" in error.value.action.casefold()

    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)
    recovered = recover_transactions(transaction.layout)
    assert recovered.post_commit_cleanup_pending == 0
    assert recovered.writes_quarantined is False
    second = WikiTransaction.prepare(
        layout=transaction.layout,
        changes=[
            PageChange(
                page=_page("concepts/second.md", revision=1, body="Second.\n"),
                expected_revision=None,
            )
        ],
        expected_global_revision=1,
        index_bytes=b"# Wiki Index\n\nSecond.\n",
        log_bytes=b"# Wiki Log\n\n## [2026-07-24] remember | second\n",
    )
    assert second.commit() == 2
    assert recover_transactions(transaction.layout).writes_quarantined is False


def test_commit_rejects_pending_cleanup_before_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, target = _prepare_update(tmp_path / "wiki")
    before = _snapshot(transaction, target)
    pending = transaction_module.RecoveryResult(post_commit_cleanup_pending=1)
    monkeypatch.setattr(
        transaction_module,
        "_recover_transactions_locked",
        lambda _layout: pending,
    )

    with pytest.raises(WikiRecoveryRequired, match="cleanup") as error:
        transaction.commit()

    assert error.value.retryable is True
    assert _snapshot(transaction, target) == before


@pytest.mark.parametrize(
    "failpoint",
    [
        f"{boundary}_{stage}"
        for boundary in ("page", "index", "log", "revision")
        for stage in (
            "temp_create",
            "temp_write",
            "temp_fsync",
            "pre_replace",
            "replace",
            "directory_fsync",
        )
    ],
)
def test_each_target_replace_failpoint_recovers_complete_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    transaction, target = _prepare_update(tmp_path / failpoint)
    before = _snapshot(transaction, target)

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="injected"):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.writes_quarantined is False
    if failpoint in {
        "page_temp_create",
        "page_temp_write",
        "page_temp_fsync",
        "page_pre_replace",
    }:
        assert _snapshot(transaction, target) == before
        return
    assert target.read_text(encoding="utf-8").endswith("After.\n")
    assert transaction.layout.index.read_bytes().endswith(b"[[concepts/atomic-writes]]\n")
    assert transaction.layout.log.read_bytes().endswith(b"remember | atomic writes\n")
    assert transaction.layout.revision.read_bytes() == b"1\n"


@pytest.mark.parametrize(
    "failpoint",
    [
        f"{boundary}_{stage}"
        for boundary in ("prepared_record", "commit_record")
        for stage in (
            "temp_create",
            "temp_write",
            "temp_fsync",
            "pre_replace",
            "replace",
            "directory_fsync",
        )
    ],
)
def test_each_record_replace_failpoint_has_deterministic_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    transaction, target = _prepare_update(tmp_path / failpoint)
    before = _snapshot(transaction, target)

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="injected"):
        transaction.commit()
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.writes_quarantined is False
    snapshot = _snapshot(transaction, target)
    if failpoint.startswith("prepared_record"):
        assert snapshot == before
    else:
        assert snapshot[0].decode().endswith("After.\n")
        assert snapshot[3] == b"1\n"


@pytest.mark.parametrize(
    "failpoint",
    [
        "reindex_marker_temp_create",
        "reindex_marker_temp_write",
        "reindex_marker_temp_fsync",
        "reindex_marker_pre_replace",
        "reindex_marker_replace",
        "reindex_marker_directory_fsync",
        "journal_cleanup_pre_delete",
        "journal_cleanup_delete",
        "journal_cleanup_directory_fsync",
    ],
)
def test_each_post_commit_cleanup_failpoint_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    transaction, target = _prepare_update(tmp_path / failpoint)

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    assert transaction.commit() == 1
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.writes_quarantined is False
    assert result.required_reindex_revision == 1
    assert target.read_text(encoding="utf-8").endswith("After.\n")


@pytest.mark.parametrize(
    "failpoint",
    [
        "rollback_temp_create",
        "rollback_temp_write",
        "rollback_temp_fsync",
        "rollback_pre_replace",
        "rollback_replace",
        "rollback_directory_fsync",
    ],
)
def test_each_rollback_replace_failpoint_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    transaction, target = _prepare_update(tmp_path / failpoint)
    before = _snapshot(transaction, target)

    def stop_after_page(name: str) -> None:
        if name == "page_replace":
            raise OSError("initial interruption")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", stop_after_page)
    with pytest.raises(OSError):
        transaction.commit()
    journal = next(transaction.layout.metadata.joinpath("journal").glob("*"))
    (journal / "new" / "0001").unlink()

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="injected"):
        recover_transactions(transaction.layout)
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(transaction.layout)
    assert result.writes_quarantined is False
    assert _snapshot(transaction, target) == before


@pytest.mark.parametrize(
    "failpoint",
    ["rollback_remove", "rollback_directory_fsync"],
)
def test_each_rollback_remove_failpoint_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
) -> None:
    layout = ensure_wiki(tmp_path / failpoint)
    target = layout.root / "concepts" / "new.md"
    transaction = WikiTransaction.prepare(
        layout=layout,
        changes=[
            PageChange(
                page=_page("concepts/new.md", revision=1, body="New.\n"),
                expected_revision=None,
            )
        ],
        expected_global_revision=0,
        index_bytes=b"# Wiki Index\n\nNew.\n",
        log_bytes=b"# Wiki Log\n\n## [2026-07-24] remember | new\n",
    )

    def stop_after_page(name: str) -> None:
        if name == "page_replace":
            raise OSError("initial interruption")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", stop_after_page)
    with pytest.raises(OSError):
        transaction.commit()
    journal = next(layout.metadata.joinpath("journal").glob("*"))
    (journal / "new" / "0001").unlink()

    def inject(name: str) -> None:
        if name == failpoint:
            raise OSError(f"injected {name}")

    monkeypatch.setattr(transaction_module, "_hit_failpoint", inject)
    with pytest.raises(OSError, match="injected"):
        recover_transactions(layout)
    monkeypatch.setattr(transaction_module, "_hit_failpoint", lambda _name: None)

    result = recover_transactions(layout)
    assert result.writes_quarantined is False
    assert not target.exists()
    assert layout.revision.read_bytes() == b"0\n"
