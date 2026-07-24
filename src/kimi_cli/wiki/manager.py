"""Application service boundary for the shared, authoritative global Wiki."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kimi_cli.wiki.initialize import CATEGORY_DIRS, WikiLayout, ensure_wiki
from kimi_cli.wiki.lint import LintReport, lint_snapshot
from kimi_cli.wiki.locking import WikiLock
from kimi_cli.wiki.models import (
    CurrentSource,
    PageChange,
    SourceRef,
    WikiCandidate,
    WikiPage,
)
from kimi_cli.wiki.schema import (
    content_hash,
    parse_page,
    resolve_page_path,
    validate_logical_page,
)
from kimi_cli.wiki.search import SearchResult, WikiSearchIndex
from kimi_cli.wiki.transaction import (
    WikiTransaction,
    acknowledge_reindex,
    wiki_read_lock,
)
from kimi_cli.wiki.value_gate import (
    DiscardedCandidate,
    WikiContext,
    contains_sensitive_text,
    duplicate_candidate_paths,
    evaluate_candidate,
)
from kimi_cli.wiki.workspaces import WorkspaceRegistry

_LOCK_TIMEOUT_SECONDS = 5.0
_CATEGORY_LABELS = {
    "entities": "Entities",
    "concepts": "Concepts",
    "comparisons": "Comparisons",
    "sources": "Sources",
    "queries": "Queries",
    "lint": "Lint",
}


@dataclass(frozen=True, slots=True)
class WikiReadResult:
    page: WikiPage
    content: str
    global_revision: int


@dataclass(frozen=True, slots=True)
class PreparedWikiChange:
    summary: str
    pages: tuple[str, ...]
    source_ids: tuple[str, ...]
    duplicate_pages: tuple[str, ...]
    conflict_pages: tuple[str, ...]
    transaction: WikiTransaction = field(repr=False)


@dataclass(frozen=True, slots=True)
class CommitResult:
    global_revision: int
    pages: tuple[str, ...]
    search_index_current: bool


@dataclass(frozen=True, slots=True)
class _AuthoritySnapshot:
    global_revision: int
    raw_pages: Mapping[str, str]
    pages: tuple[WikiPage, ...]
    log_text: str
    needs_reindex: bool


class WikiManager:
    """Own initialization, safe reads, admission, transactions, cache, and lint."""

    def __init__(self, root: Path | None = None, *, wal: bool = True) -> None:
        self.layout = ensure_wiki(root)
        self.registry = WorkspaceRegistry(self.layout.metadata / "workspaces.json")
        # Exposed for lock-state diagnostics and Task 10 approval-boundary tests.
        # Commit deliberately delegates locking to WikiTransaction.
        self.lock = WikiLock(self.layout.metadata / "locks" / "writer.lock")
        self.search_index = WikiSearchIndex.open(self.layout.database, wal=wal)
        self._search_index_current = self._refresh_search_index(force=True)

    def close(self) -> None:
        self.search_index.close()

    def ensure(self) -> WikiLayout:
        return self.layout

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self._search_index_current = self._refresh_search_index(
            force=not self._search_index_current
        )
        return self.search_index.search(query, limit)

    def read(self, page: str) -> WikiReadResult:
        logical_path = validate_logical_page(page).as_posix()
        with wiki_read_lock(self.layout, timeout=_LOCK_TIMEOUT_SECONDS):
            revision = _read_global_revision(self.layout)
            unresolved = self.layout.root / logical_path
            target = resolve_page_path(self.layout.root, logical_path)
            if unresolved.is_symlink() or not target.is_file():
                raise FileNotFoundError(logical_path)
            text = target.read_text(encoding="utf-8")
            parsed = parse_page(text, logical_path)
        return WikiReadResult(page=parsed, content=text, global_revision=revision)

    def prepare(
        self,
        candidate: WikiCandidate,
        context: WikiContext,
    ) -> PreparedWikiChange | DiscardedCandidate:
        """Prepare a complete mutation and release all locks before permission."""
        snapshot = self._snapshot()
        if not self._workspace_sources_resolve(candidate):
            return DiscardedCandidate(reason="ungrounded", summary=candidate.summary)
        decision = evaluate_candidate(candidate, context, snapshot.pages)
        if not decision.accepted:
            assert decision.reason is not None
            return DiscardedCandidate(reason=decision.reason, summary=candidate.summary)

        duplicate_pages = duplicate_candidate_paths(candidate, snapshot.pages)
        changes = _materialize_changes(
            candidate,
            context,
            snapshot.pages,
            excluded_paths=frozenset(duplicate_pages),
        )
        future_pages = {page.logical_path: page for page in snapshot.pages}
        for change in changes:
            future_pages[change.page.logical_path] = change.page
        index_bytes = _render_index(tuple(future_pages.values())).encode("utf-8")
        new_revision = snapshot.global_revision + 1
        log_bytes = _append_log(
            snapshot.log_text,
            candidate,
            context,
            changes,
            new_revision,
        ).encode("utf-8")
        transaction = WikiTransaction.prepare(
            layout=self.layout,
            changes=changes,
            expected_global_revision=snapshot.global_revision,
            index_bytes=index_bytes,
            log_bytes=log_bytes,
        )
        return PreparedWikiChange(
            summary=" ".join(candidate.summary.split())[:240],
            pages=tuple(change.page.logical_path for change in changes),
            source_ids=tuple(sorted(_audit_source_id(source) for source in candidate.sources)),
            duplicate_pages=duplicate_pages,
            conflict_pages=tuple(
                path for path in context.conflicting_pages if path in future_pages
            ),
            transaction=transaction,
        )

    def commit(self, prepared: PreparedWikiChange) -> CommitResult:
        """Commit an already-decided proposal; WikiTransaction owns the writer lock."""
        revision = prepared.transaction.commit()
        search_current = self._refresh_search_index(force=True)
        self._search_index_current = search_current
        return CommitResult(
            global_revision=revision,
            pages=prepared.pages,
            search_index_current=search_current,
        )

    def ingest(
        self,
        source: CurrentSource,
        instructions: WikiCandidate,
        context: WikiContext,
    ) -> PreparedWikiChange | DiscardedCandidate:
        """Validate a current-session source, then prepare its structured candidate.

        The caller/model supplies the structured candidate as ``instructions``;
        this manager never performs a hidden model call or stores raw source text.
        """
        expected_source = self._source_ref_for_ingest(source, context)
        if isinstance(expected_source, DiscardedCandidate):
            return expected_source
        expected_key = expected_source.model_dump_json(exclude_none=True)
        supplied = {item.model_dump_json(exclude_none=True) for item in instructions.sources}
        page_source_sets = (
            {item.model_dump_json(exclude_none=True) for item in change.page.sources}
            for change in instructions.pages
        )
        if expected_key not in supplied or any(
            expected_key not in sources for sources in page_source_sets
        ):
            return DiscardedCandidate(reason="ungrounded", summary=instructions.summary)
        ingest_context = context.model_copy(update={"operation": "ingest"})
        return self.prepare(instructions, ingest_context)

    def lint(self, scope: str | None) -> LintReport:
        if scope is not None and scope not in CATEGORY_DIRS:
            raise ValueError("Wiki lint scope must be a declared category")
        snapshot = self._snapshot()
        return lint_snapshot(snapshot.raw_pages, scope=scope)

    def _source_ref_for_ingest(
        self,
        source: CurrentSource,
        context: WikiContext,
    ) -> SourceRef | DiscardedCandidate:
        if source.kind == "inline":
            assert source.content is not None
            if contains_sensitive_text(source.content):
                return DiscardedCandidate(reason="sensitive", summary="Ingest source rejected")
            return SourceRef(
                kind="conversation",
                session_id=context.session_id,
                content_hash=content_hash(source.content.encode("utf-8")),
            )
        assert source.workspace_id is not None and source.relative_path is not None
        portable = SourceRef(
            kind="workspace-file",
            workspace_id=source.workspace_id,
            path=source.relative_path,
            content_hash="sha256:" + "0" * 64,
        )
        resolved = self.registry.resolve(portable)
        if resolved is None:
            return DiscardedCandidate(reason="ungrounded", summary="Ingest source rejected")
        data = resolved.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return DiscardedCandidate(reason="sensitive", summary="Ingest source rejected")
        if contains_sensitive_text(text):
            return DiscardedCandidate(reason="sensitive", summary="Ingest source rejected")
        return portable.model_copy(update={"content_hash": content_hash(data)})

    def _workspace_sources_resolve(self, candidate: WikiCandidate) -> bool:
        sources = (
            *candidate.sources,
            *(source for change in candidate.pages for source in change.page.sources),
        )
        checked: set[str] = set()
        for source in sources:
            if source.kind != "workspace-file":
                continue
            key = source.model_dump_json(exclude_none=True)
            if key in checked:
                continue
            checked.add(key)
            resolved = self.registry.resolve(source)
            if resolved is None:
                return False
            try:
                if content_hash(resolved.read_bytes()) != source.content_hash:
                    return False
            except OSError:
                return False
        return True

    def _snapshot(self) -> _AuthoritySnapshot:
        with wiki_read_lock(self.layout, timeout=_LOCK_TIMEOUT_SECONDS) as recovery:
            revision = _read_global_revision(self.layout)
            raw_pages = _read_raw_pages(self.layout)
            pages = _parse_valid_pages(raw_pages)
            log_text = self.layout.log.read_text(encoding="utf-8")
            needs_reindex = recovery.needs_reindex
        return _AuthoritySnapshot(
            global_revision=revision,
            raw_pages=raw_pages,
            pages=pages,
            log_text=log_text,
            needs_reindex=needs_reindex,
        )

    def _refresh_search_index(self, *, force: bool) -> bool:
        """Sync one full revision snapshot, then conditionally acknowledge that revision."""
        snapshot = self._snapshot()
        if not force and not snapshot.needs_reindex:
            return True
        try:
            # Intentionally outside wiki_read_lock: SQLite callbacks and diagnostics
            # must be able to call manager.read without re-entering a shared lock.
            self.search_index.sync(snapshot.pages)
        except sqlite3.Error:
            return False
        acknowledgement = acknowledge_reindex(
            self.layout,
            rebuilt_revision=snapshot.global_revision,
            timeout=_LOCK_TIMEOUT_SECONDS,
        )
        return acknowledgement.acknowledged


def _read_global_revision(layout: WikiLayout) -> int:
    raw = layout.revision.read_text(encoding="ascii").strip()
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError("Wiki revision must be a non-negative integer")
    return int(raw)


def _read_raw_pages(layout: WikiLayout) -> dict[str, str]:
    pages: dict[str, str] = {}
    for category in CATEGORY_DIRS:
        directory = layout.root / category
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or not directory.resolve(strict=True).is_relative_to(layout.root.resolve(strict=True))
        ):
            continue
        for target in sorted(directory.glob("*.md"), key=lambda path: path.name):
            logical_path = f"{category}/{target.name}"
            validate_logical_page(logical_path)
            if target.is_symlink() or not target.is_file():
                continue
            try:
                pages[logical_path] = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Preserve only a sentinel in memory so lint deterministically
                # reports malformed UTF-8; authoritative bytes stay untouched.
                pages[logical_path] = ""
    return pages


def _parse_valid_pages(raw_pages: Mapping[str, str]) -> tuple[WikiPage, ...]:
    pages: list[WikiPage] = []
    for logical_path, text in sorted(raw_pages.items()):
        try:
            pages.append(parse_page(text, logical_path))
        except (ValueError, UnicodeError):
            continue
    return tuple(pages)


def _materialize_changes(
    candidate: WikiCandidate,
    context: WikiContext,
    existing_pages: tuple[WikiPage, ...],
    *,
    excluded_paths: frozenset[str] = frozenset(),
) -> tuple[PageChange, ...]:
    existing_by_path = {page.logical_path: page for page in existing_pages}
    conflicts = set(context.conflicting_pages)
    changes: list[PageChange] = []
    for proposed in candidate.pages:
        if proposed.page.logical_path in excluded_paths:
            continue
        current = existing_by_path.get(proposed.page.logical_path)
        if current is None:
            if proposed.expected_revision is not None:
                raise ValueError(f"page is missing: {proposed.page.logical_path}")
            page = proposed.page.model_copy(update={"revision": 1})
            changes.append(PageChange(page=page, expected_revision=None))
            continue
        if proposed.expected_revision != current.revision:
            raise ValueError(f"page revision changed: {proposed.page.logical_path}")
        sources = _merge_sources(current.sources, proposed.page.sources)
        body = (
            _merge_conflict(current.body, proposed.page.body)
            if proposed.page.logical_path in conflicts
            else proposed.page.body
        )
        updated = max(proposed.page.updated, datetime.now().astimezone(), current.updated)
        page = proposed.page.model_copy(
            update={
                "created": current.created,
                "updated": updated,
                "sources": sources,
                "revision": current.revision + 1,
                "body": body,
            }
        )
        changes.append(PageChange(page=page, expected_revision=current.revision))
    return tuple(changes)


def _merge_sources(existing: list[SourceRef], proposed: list[SourceRef]) -> list[SourceRef]:
    result: list[SourceRef] = []
    seen: set[str] = set()
    for source in (*existing, *proposed):
        key = source.model_dump_json(exclude_none=True)
        if key not in seen:
            seen.add(key)
            result.append(source)
    return result


def _merge_conflict(existing: str, proposed: str) -> str:
    return (
        f"{existing.rstrip()}\n\n"
        "## Conflict\n\n"
        "### Additional sourced position\n\n"
        f"{proposed.strip()}\n"
    )


def _render_index(pages: tuple[WikiPage, ...]) -> str:
    by_category: dict[str, list[WikiPage]] = {category: [] for category in CATEGORY_DIRS}
    for page in pages:
        by_category[page.logical_path.split("/", 1)[0]].append(page)
    lines = ["# Wiki Index", ""]
    for category in CATEGORY_DIRS:
        lines.extend((f"## {_CATEGORY_LABELS[category]}", ""))
        for page in sorted(by_category[category], key=lambda item: item.logical_path):
            link = page.logical_path.removesuffix(".md")
            lines.append(f"- [[{link}]] — {_one_line_summary(page.body)}")
        if by_category[category]:
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _one_line_summary(body: str) -> str:
    for line in body.splitlines():
        normalized = line.strip().lstrip("-* ").strip()
        if normalized and not normalized.startswith("#"):
            return normalized[:240]
    return "No summary available."


def _append_log(
    previous: str,
    candidate: WikiCandidate,
    context: WikiContext,
    changes: tuple[PageChange, ...],
    revision: int,
) -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    pages = ",".join(change.page.logical_path for change in changes)
    sources = ",".join(sorted(_audit_source_id(source) for source in candidate.sources))
    summary = " ".join(candidate.summary.split()).replace("%", "%25").replace("|", "%7C")
    entry = (
        f"{timestamp} | operation={context.operation} | revision={revision} | "
        f"pages={pages} | sources={sources} | summary={summary}\n"
    )
    return previous.rstrip() + "\n\n" + entry


def _audit_source_id(source: SourceRef) -> str:
    if source.kind == "conversation":
        return f"conversation:{source.session_id}"
    if source.kind == "workspace-file":
        return f"workspace:{source.workspace_id}"
    return f"web:{source.content_hash}"
