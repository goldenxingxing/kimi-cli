from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from kimi_cli.wiki.models import CurrentSource, PageChange, SourceRef, WikiCandidate, WikiPage
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.transaction import WikiConflictError

_SESSION_ID = UUID("223e4567-e89b-12d3-a456-426614174000")
_NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)


def _source(marker: str = "a") -> SourceRef:
    return SourceRef(
        kind="conversation",
        session_id=_SESSION_ID,
        content_hash="sha256:" + marker * 64,
    )


def _candidate(
    *,
    path: str = "concepts/cache-mode.md",
    title: str = "Cache mode",
    body: str = "Source-a says WAL is appropriate for a local cache.\n",
    marker: str = "a",
    revision: int = 1,
    expected_revision: int | None = None,
) -> WikiCandidate:
    source = _source(marker)
    page = WikiPage(
        logical_path=path,
        title=title,
        created=_NOW,
        updated=_NOW,
        tags=["sqlite", "wiki"],
        sources=[source],
        revision=revision,
        body=body,
    )
    return WikiCandidate(
        summary=f"Record {title}",
        pages=[PageChange(page=page, expected_revision=expected_revision)],
        sources=[source],
        value="high",
    )


def _context(*, conflicts: tuple[str, ...] = ()):
    from kimi_cli.wiki.value_gate import WikiContext

    return WikiContext(
        session_id=_SESSION_ID,
        cross_turn_utility=True,
        stable=True,
        user_confirmed=True,
        conflicting_pages=conflicts,
    )


@pytest.fixture
def manager(tmp_path: Path):
    from kimi_cli.wiki.manager import WikiManager

    instance = WikiManager(tmp_path / "wiki", wal=False)
    yield instance
    instance.close()


def test_prepare_is_non_mutating_and_commit_updates_page_index_log_and_search(manager) -> None:
    from kimi_cli.wiki.manager import PreparedWikiChange

    prepared = manager.prepare(_candidate(), _context())

    assert isinstance(prepared, PreparedWikiChange)
    assert prepared.summary == "Record Cache mode"
    assert prepared.pages == ("concepts/cache-mode.md",)
    assert not (manager.layout.root / "concepts" / "cache-mode.md").exists()
    assert manager.layout.revision.read_text(encoding="ascii") == "0\n"

    result = manager.commit(prepared)

    assert result.global_revision == 1
    assert result.pages == ("concepts/cache-mode.md",)
    assert manager.read("concepts/cache-mode.md").page.revision == 1
    assert manager.search("WAL", 5)[0].logical_path == "concepts/cache-mode.md"
    assert "[[concepts/cache-mode]] — Source-a says WAL" in manager.layout.index.read_text(
        encoding="utf-8"
    )
    log = manager.layout.log.read_text(encoding="utf-8")
    assert "operation=remember" in log
    assert "revision=1" in log
    assert "pages=concepts/cache-mode.md" in log
    assert str(manager.layout.root) not in log


def test_prepare_releases_all_locks_before_external_decision(manager) -> None:
    prepared = manager.prepare(_candidate(), _context())

    with manager.lock.exclusive(timeout=0.1):
        assert prepared.pages == ("concepts/cache-mode.md",)


def test_prepared_approval_metadata_is_compact_and_contains_no_raw_content(manager) -> None:
    candidate = _candidate().model_copy(
        update={"summary": "Record cache guidance\nfor future sessions"}
    )

    prepared = manager.prepare(candidate, _context())

    assert prepared.summary == "Record cache guidance for future sessions"
    assert prepared.pages == ("concepts/cache-mode.md",)
    assert len(prepared.source_ids) == 1
    assert "Source-a says" not in repr(prepared)


def test_audit_summary_cannot_inject_machine_parseable_fields(manager) -> None:
    candidate = _candidate().model_copy(
        update={"summary": "Record cache guidance | operation=delete"}
    )

    manager.commit(manager.prepare(candidate, _context()))

    log = manager.layout.log.read_text(encoding="utf-8")
    assert log.count(" | operation=") == 1
    assert "summary=Record cache guidance %7C operation=delete" in log


def test_commit_does_not_use_manager_outer_exclusive_lock(manager) -> None:
    prepared = manager.prepare(_candidate(), _context())

    class BombLock:
        def exclusive(self, timeout: float):
            raise AssertionError("WikiManager must not wrap WikiTransaction.commit")

    manager.lock = BombLock()
    assert manager.commit(prepared).global_revision == 1


def test_conflict_preserves_both_sourced_positions(manager) -> None:
    manager.commit(manager.prepare(_candidate(), _context()))
    conflicting = _candidate(
        body="Source-b says WAL must not be shared over a network filesystem.\n",
        marker="b",
        revision=2,
        expected_revision=1,
    )

    result = manager.commit(
        manager.prepare(
            conflicting,
            _context(conflicts=("concepts/cache-mode.md",)),
        )
    )

    page = manager.read("concepts/cache-mode.md").page
    assert page.revision == 2
    assert "## Conflict" in page.body
    assert "Source-a" in page.body
    assert "Source-b" in page.body
    existing_source_id = f"conversation:{_SESSION_ID}@sha256:{'a' * 64}"
    proposed_source_id = f"conversation:{_SESSION_ID}@sha256:{'b' * 64}"
    assert (
        f"### Existing sourced position\n\nSources: {existing_source_id}\n\nSource-a" in page.body
    )
    assert (
        f"### Additional sourced position\n\nSources: {proposed_source_id}\n\nSource-b" in page.body
    )
    assert {source.content_hash for source in page.sources} == {
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    }
    assert f"revision={result.global_revision}" in manager.layout.log.read_text(encoding="utf-8")


def test_revision_change_after_prepare_is_rejected_without_overwrite(manager) -> None:
    manager.commit(manager.prepare(_candidate(), _context()))
    stale = manager.prepare(
        _candidate(body="Stale proposal.\n", marker="b", revision=2, expected_revision=1),
        _context(),
    )
    winner = manager.prepare(
        _candidate(body="Concurrent winner.\n", marker="c", revision=2, expected_revision=1),
        _context(),
    )
    manager.commit(winner)

    with pytest.raises(WikiConflictError):
        manager.commit(stale)

    assert manager.read("concepts/cache-mode.md").page.body == "Concurrent winner.\n"


def test_mixed_duplicate_and_novel_candidate_commits_only_novel_page(manager) -> None:
    manager.commit(manager.prepare(_candidate(), _context()))
    duplicate = _candidate(revision=2, expected_revision=1).pages[0]
    novel_candidate = _candidate(
        path="entities/sqlite.md",
        title="SQLite",
        body="SQLite is an embedded database.\n",
        marker="b",
    )
    mixed = novel_candidate.model_copy(
        update={
            "pages": [duplicate, novel_candidate.pages[0]],
            "sources": [_source("a"), _source("b")],
        }
    )

    prepared = manager.prepare(mixed, _context())

    assert prepared.pages == ("entities/sqlite.md",)
    assert prepared.duplicate_pages == ("concepts/cache-mode.md",)
    manager.commit(prepared)
    assert manager.read("concepts/cache-mode.md").page.revision == 1


def test_ingest_requires_current_source_provenance_on_every_page(manager) -> None:
    raw = "A stable source supplied in this user interaction."
    current = CurrentSource(kind="inline", content=raw)
    expected = SourceRef(
        kind="conversation",
        session_id=_SESSION_ID,
        content_hash=content_hash(raw.encode("utf-8")),
    )
    unrelated = _source("b")
    candidate = _candidate()
    page = candidate.pages[0].page.model_copy(update={"sources": [unrelated]})
    instructions = candidate.model_copy(
        update={
            "pages": [PageChange(page=page, expected_revision=None)],
            "sources": [expected, unrelated],
        }
    )

    discarded = manager.ingest(current, instructions, _context())

    assert discarded.reason == "ungrounded"
    assert not (manager.layout.root / "concepts" / "cache-mode.md").exists()


def test_ingest_prepares_sanitized_conclusion_without_storing_raw_source(manager) -> None:
    raw = "A long current-turn source whose raw wording must not be persisted."
    current = CurrentSource(kind="inline", content=raw)
    expected = SourceRef(
        kind="conversation",
        session_id=_SESSION_ID,
        content_hash=content_hash(raw.encode("utf-8")),
    )
    candidate = _candidate(body="Durable sanitized conclusion.\n")
    page = candidate.pages[0].page.model_copy(update={"sources": [expected]})
    instructions = candidate.model_copy(
        update={
            "pages": [PageChange(page=page, expected_revision=None)],
            "sources": [expected],
        }
    )

    prepared = manager.ingest(current, instructions, _context())

    assert prepared.pages == ("concepts/cache-mode.md",)
    manager.commit(prepared)
    stored = manager.read("concepts/cache-mode.md").content
    assert "Durable sanitized conclusion." in stored
    assert raw not in stored
    assert "operation=ingest" in manager.layout.log.read_text(encoding="utf-8")


def test_ingest_discards_raw_source_with_credential_alias(manager) -> None:
    current = CurrentSource(kind="inline", content="refresh_token=do-not-store")

    result = manager.ingest(current, _candidate(), _context())

    assert result.reason == "sensitive"
    assert not (manager.layout.metadata / "pending").exists()


def test_ingest_reuses_encoded_url_credential_scanner(manager) -> None:
    current = CurrentSource(
        kind="inline",
        content="See https://example.test/article?refresh%5Ftoken=do-not-store",
    )

    result = manager.ingest(current, _candidate(), _context())

    assert result.reason == "sensitive"


def test_cache_sync_receives_a_full_snapshot_and_runs_outside_read_lock(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.commit(manager.prepare(_candidate(), _context()))
    prepared = manager.prepare(
        _candidate(
            path="entities/sqlite.md",
            title="SQLite",
            body="SQLite supplies the disposable local search cache.\n",
            marker="b",
        ),
        _context(),
    )
    original_sync = manager.search_index.sync
    snapshots: list[set[str]] = []

    def observing_sync(pages, **kwargs):
        materialized = tuple(pages)
        snapshots.append({page.logical_path for page in materialized})
        # This would deadlock or time out if the callback ran inside a shared read lock.
        assert manager.read("concepts/cache-mode.md").page.title == "Cache mode"
        return original_sync(materialized, **kwargs)

    monkeypatch.setattr(manager.search_index, "sync", observing_sync)

    result = manager.commit(prepared)

    assert snapshots == [{"concepts/cache-mode.md", "entities/sqlite.md"}]
    assert result.search_index_current is True
    assert not (manager.layout.metadata / "search.invalid").exists()


def test_search_cache_failure_does_not_fail_authoritative_commit_and_retries(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_sync = manager.search_index.sync

    def fail_sync(_pages, **_kwargs):
        raise sqlite3.OperationalError("injected cache failure")

    monkeypatch.setattr(manager.search_index, "sync", fail_sync)
    result = manager.commit(manager.prepare(_candidate(), _context()))

    assert result.global_revision == 1
    assert result.search_index_current is False
    assert manager.read("concepts/cache-mode.md").page.revision == 1
    assert (manager.layout.metadata / "search.invalid").exists()

    monkeypatch.setattr(manager.search_index, "sync", original_sync)
    assert manager.search("WAL", 5)[0].logical_path == "concepts/cache-mode.md"
    assert not (manager.layout.metadata / "search.invalid").exists()


def test_search_uses_authoritative_markdown_while_cache_refresh_keeps_failing(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_sync(_pages, **_kwargs):
        raise sqlite3.OperationalError("persistent cache failure")

    monkeypatch.setattr(manager.search_index, "sync", fail_sync)
    result = manager.commit(manager.prepare(_candidate(), _context()))

    assert result.global_revision == 1
    assert result.search_index_current is False
    assert manager.search("WAL", 5)[0].logical_path == "concepts/cache-mode.md"


@pytest.mark.parametrize("failure", [OSError("disk error"), RuntimeError("callback error")])
def test_post_commit_cache_or_ack_error_never_reports_authority_failure(
    manager, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    import kimi_cli.wiki.manager as manager_module

    if isinstance(failure, OSError):
        monkeypatch.setattr(
            manager_module,
            "acknowledge_reindex",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            manager.search_index,
            "sync",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )

    result = manager.commit(manager.prepare(_candidate(), _context()))

    assert result.global_revision == 1
    assert result.search_index_current is False
    assert manager.read("concepts/cache-mode.md").page.revision == 1


def test_cache_cas_rejection_does_not_acknowledge_stale_snapshot(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kimi_cli.wiki.manager as manager_module

    monkeypatch.setattr(manager.search_index, "sync", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        manager_module,
        "acknowledge_reindex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale cache snapshots must not be acknowledged")
        ),
    )

    result = manager.commit(manager.prepare(_candidate(), _context()))

    assert result.global_revision == 1
    assert result.search_index_current is False


def test_initial_cache_failure_is_retried_even_without_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kimi_cli.wiki.initialize import ensure_wiki
    from kimi_cli.wiki.manager import WikiManager
    from kimi_cli.wiki.schema import render_page
    from kimi_cli.wiki.search import WikiSearchIndex

    layout = ensure_wiki(tmp_path / "wiki")
    page = _candidate().pages[0].page
    (layout.root / page.logical_path).write_text(render_page(page), encoding="utf-8")
    original_sync = WikiSearchIndex.sync
    calls = 0

    def fail_first_sync(index, pages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("injected startup cache failure")
        return original_sync(index, pages, **kwargs)

    monkeypatch.setattr(WikiSearchIndex, "sync", fail_first_sync)
    manager = WikiManager(layout.root, wal=False)
    try:
        assert manager.search("WAL", 5)[0].logical_path == "concepts/cache-mode.md"
        assert calls == 2
    finally:
        manager.close()


def test_stale_cache_acknowledgement_never_reports_current(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kimi_cli.wiki.manager as manager_module
    from kimi_cli.wiki.transaction import ReindexAcknowledgement

    prepared = manager.prepare(_candidate(), _context())
    monkeypatch.setattr(
        manager_module,
        "acknowledge_reindex",
        lambda *_args, **_kwargs: ReindexAcknowledgement(
            acknowledged=False,
            required_revision=2,
        ),
    )

    result = manager.commit(prepared)

    assert result.global_revision == 1
    assert result.search_index_current is False


def test_unrelated_concurrent_commit_is_safely_rebased(manager) -> None:
    first = manager.prepare(_candidate(), _context())
    second = manager.prepare(
        _candidate(
            path="entities/sqlite.md",
            title="SQLite",
            body="SQLite is an embedded database.\n",
            marker="b",
        ),
        _context(),
    )

    assert manager.commit(first).global_revision == 1
    assert manager.commit(second).global_revision == 2

    assert manager.read("concepts/cache-mode.md").page.revision == 1
    assert manager.read("entities/sqlite.md").page.revision == 1
    index = manager.layout.index.read_text(encoding="utf-8")
    log = manager.layout.log.read_text(encoding="utf-8")
    assert "[[concepts/cache-mode]]" in index
    assert "[[entities/sqlite]]" in index
    assert "revision=2" in log


def test_rebase_uses_immutable_approved_candidate_snapshot(manager) -> None:
    candidate = _candidate(
        path="entities/sqlite.md",
        title="SQLite",
        body="Approved SQLite knowledge.\n",
        marker="b",
    )
    prepared = manager.prepare(candidate, _context())
    candidate.pages[0].page.body = "Unapproved mutation after preparation.\n"
    manager.commit(manager.prepare(_candidate(), _context()))

    manager.commit(prepared)

    body = manager.read("entities/sqlite.md").page.body
    assert body == "Approved SQLite knowledge.\n"


def test_malformed_logical_filename_is_skipped_by_search_and_reported_by_lint(manager) -> None:
    malformed = manager.layout.root / "concepts" / "bad name.md"
    malformed.write_text("not a valid logical Wiki page\n", encoding="utf-8")

    assert manager.search("logical Wiki", 5) == []
    report = manager.lint(None)

    assert any(
        issue.code == "malformed_page" and issue.logical_path == "concepts/bad name.md"
        for issue in report.issues
    )


def test_nested_logical_page_is_skipped_by_search_and_reported_by_lint(manager) -> None:
    nested = manager.layout.root / "concepts" / "nested"
    nested.mkdir()
    (nested / "page.md").write_text("nested invalid page\n", encoding="utf-8")

    assert manager.search("nested invalid", 5) == []
    assert any(
        issue.code == "malformed_page" and issue.logical_path == "concepts/nested/page.md"
        for issue in manager.lint(None).issues
    )


def test_manager_can_cross_threads_for_construct_use_commit_search_and_close(
    tmp_path: Path,
) -> None:
    from kimi_cli.wiki.manager import PreparedWikiChange, WikiManager

    with ThreadPoolExecutor(max_workers=1) as executor:
        manager = executor.submit(WikiManager, tmp_path / "wiki", wal=False).result()
    try:
        prepared = manager.prepare(_candidate(), _context())
        assert isinstance(prepared, PreparedWikiChange)
        result = manager.commit(prepared)
        assert result.search_index_current is True
        assert manager.search("WAL", 5)[0].logical_path == "concepts/cache-mode.md"
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: manager.search("WAL", 5), range(8)))
        assert all(items[0].logical_path == "concepts/cache-mode.md" for items in results)
    finally:
        manager.close()
