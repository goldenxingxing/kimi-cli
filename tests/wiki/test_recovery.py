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

    def cannot_recover(_target: Path, _data: bytes) -> None:
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
    calls: list[str] = []

    result = recover_transactions(
        transaction.layout,
        rebuild_search=lambda: calls.append("rebuilt"),
    )

    assert result.needs_reindex is True
    assert calls == ["rebuilt"]
    assert not list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))


def test_failed_search_rebuild_retains_committed_journal(tmp_path: Path) -> None:
    transaction, _ = _prepare_update(tmp_path / "wiki")
    transaction.commit()

    def fail_rebuild() -> None:
        raise OSError("sqlite unavailable")

    with pytest.raises(OSError, match="sqlite unavailable"):
        recover_transactions(transaction.layout, rebuild_search=fail_rebuild)

    assert list(transaction.layout.metadata.joinpath("journal").glob("*/record.json"))
