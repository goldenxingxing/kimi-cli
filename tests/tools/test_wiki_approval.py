"""Approval and execution policy for managed global Wiki writes."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from uuid import UUID

import pytest

from kimi_cli.approval_runtime import ApprovalRuntime
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval, ApprovalState
from kimi_cli.soul.toolset import current_tool_call
from kimi_cli.tools.display import WikiApprovalBlock
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.tools.wiki import (
    Params,
    Wiki,
    WikiToolContext,
    extend_wiki_turn_context,
    reset_wiki_turn_context,
    set_wiki_turn_context,
)
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wiki.models import PageChange, SourceRef, WikiCandidate, WikiPage
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.value_gate import WikiContext
from kimi_cli.wire.types import ToolCall

_SESSION_ID = UUID("923e4567-e89b-12d3-a456-426614174000")
_NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)


def _source(marker: str = "a") -> SourceRef:
    return SourceRef(
        kind="conversation",
        session_id=_SESSION_ID,
        content_hash="sha256:" + marker * 64,
    )


def _candidate(
    *,
    value: Literal["high", "medium", "low"] = "high",
    summary: str = "Atomic Wiki recovery",
    marker: str = "a",
    source: SourceRef | None = None,
    paths: tuple[str, ...] = (
        "concepts/atomic-wiki-recovery.md",
        "entities/sqlite.md",
    ),
) -> WikiCandidate:
    source = source or _source(marker)
    pages = [
        PageChange(
            page=WikiPage(
                logical_path=path,
                title=path.rsplit("/", 1)[1].removesuffix(".md").replace("-", " ").title(),
                created=_NOW,
                updated=_NOW,
                tags=["wiki"],
                sources=[source],
                revision=1,
                body=f"Durable sourced conclusion for {path}.\n",
            ),
            expected_revision=None,
        )
        for path in paths
    ]
    return WikiCandidate(
        summary=summary,
        pages=pages,
        sources=[source],
        value=value,
    )


@pytest.fixture
def manager(tmp_path: Path):
    instance = WikiManager(tmp_path / "wiki", wal=False)
    yield instance
    instance.close()


def _runtime(
    manager: WikiManager,
    *,
    yolo: bool = False,
    afk: bool = False,
    trusted_turn: bool = True,
):
    session = SimpleNamespace(
        id="named-session",
        state=SimpleNamespace(
            approval=SimpleNamespace(auto_approve_actions=set()),
        ),
    )
    state: ApprovalState

    def persist_approval() -> None:
        session.state.approval.auto_approve_actions = set(state.auto_approve_actions)

    approval_runtime = ApprovalRuntime()
    state = ApprovalState(
        yolo=yolo,
        afk=afk,
        on_change=persist_approval,
    )
    approval = Approval(
        state=state,
        runtime=approval_runtime,
    )
    runtime = SimpleNamespace(
        wiki=manager,
        approval=approval,
        session=session,
        workspace_id=None,
        wiki_tool_context=WikiToolContext(
            provenance_session_id=_SESSION_ID,
            conversation_hashes=(
                frozenset(
                    {
                        "sha256:" + "a" * 64,
                        "sha256:" + "b" * 64,
                        "sha256:" + "c" * 64,
                    }
                )
                if trusted_turn
                else frozenset()
            ),
            allowed_workspace_ids=frozenset(),
            candidate_high_value=trusted_turn,
            stable=trusted_turn,
            user_confirmed=trusted_turn,
            reliable_source=False,
        ),
    )
    return cast("Runtime", runtime), approval_runtime


async def _next_pending(runtime: ApprovalRuntime):
    async with asyncio.timeout(2):
        while not runtime.list_pending():
            await asyncio.sleep(0)
    return runtime.list_pending()[0]


@contextmanager
def _tool_call_context():
    token = current_tool_call.set(
        ToolCall(
            id="test",
            function=ToolCall.FunctionBody(name="Wiki", arguments=None),
        )
    )
    try:
        yield
    finally:
        current_tool_call.reset(token)


async def _start_write(tool: Wiki, candidate: WikiCandidate):
    with _tool_call_context():
        return asyncio.create_task(
            tool(Params(operation="remember", candidate=candidate)),
        )


async def test_trusted_user_turn_supplies_ephemeral_write_evidence(
    manager,
) -> None:
    runtime, approval_runtime = _runtime(manager, trusted_turn=False)
    user_text = "Please remember this durable conclusion."
    token = set_wiki_turn_context(
        runtime,
        user_text,
        trusted_user_input=True,
    )
    try:
        active = Wiki.current_context(runtime)
        assert active is not None
        assert active.candidate_high_value
        assert active.stable
        assert active.user_confirmed
        assert active.conversation_hashes

        pending = await _start_write(
            Wiki(runtime),
            _candidate(
                source=SourceRef(
                    kind="conversation",
                    session_id=_SESSION_ID,
                    content_hash=content_hash(user_text.encode("utf-8")),
                ),
                paths=("concepts/trusted-turn.md",),
            ),
        )
        request = await _next_pending(approval_runtime)
        approval_runtime.resolve(request.id, "approve")
        assert not (await pending).is_error
    finally:
        reset_wiki_turn_context(token)

    assert Wiki.current_context(runtime) is runtime.wiki_tool_context


def test_synthetic_turn_cannot_enable_wiki_write_evidence(manager) -> None:
    runtime, _approval_runtime = _runtime(manager, trusted_turn=False)

    token = set_wiki_turn_context(
        runtime,
        "internal background bookkeeping",
        trusted_user_input=False,
    )
    try:
        assert runtime.wiki_tool_context is not None
        assert Wiki.current_context(runtime) is runtime.wiki_tool_context
        assert not runtime.wiki_tool_context.candidate_high_value
    finally:
        reset_wiki_turn_context(token)


def test_user_steer_extends_only_the_active_turn_hashes(manager) -> None:
    runtime, _approval_runtime = _runtime(manager, trusted_turn=False)
    token = set_wiki_turn_context(runtime, "initial", trusted_user_input=True)
    try:
        steer = "Remember this follow-up too."
        extend_wiki_turn_context(steer)
        active = Wiki.current_context(runtime)
        assert active is not None
        assert content_hash(steer.encode("utf-8")) in active.conversation_hashes
    finally:
        reset_wiki_turn_context(token)

    extend_wiki_turn_context("outside turn")
    assert Wiki.current_context(runtime) is runtime.wiki_tool_context


async def test_normal_write_asks_before_lock_and_approve_once_commits(
    manager,
) -> None:
    runtime, approval_runtime = _runtime(manager)
    pending = await _start_write(Wiki(runtime), _candidate())

    request = await _next_pending(approval_runtime)
    assert request.action == "wiki.write"
    assert request.description == "Record: Atomic Wiki recovery\nChanges: 2 pages"
    assert request.sender == "Wiki"
    assert request.display == [
        WikiApprovalBlock(
            summary="Atomic Wiki recovery",
            pages=["concepts/atomic-wiki-recovery.md", "entities/sqlite.md"],
            workspace_id=None,
            session_id=str(_SESSION_ID),
            details=[
                f"Sources: conversation:{_SESSION_ID}@sha256:{'a' * 64}",
                "Paths are normalized relative to the managed Wiki.",
            ],
        )
    ]
    # Preparing and waiting for the user must not retain the cross-process writer lock.
    with manager.lock.exclusive(timeout=0.1):
        assert manager.layout.revision.read_text(encoding="ascii") == "0\n"

    approval_runtime.resolve(request.id, "approve")
    result = await pending

    assert not result.is_error
    assert manager.layout.revision.read_text(encoding="ascii") == "1\n"
    assert (manager.layout.root / "concepts" / "atomic-wiki-recovery.md").is_file()


async def test_model_initiated_normal_proposal_with_workspace_evidence_still_asks(
    manager, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "evidence.md"
    evidence.write_text("A stable workspace conclusion.\n", encoding="utf-8")
    workspace_id = manager.registry.register(workspace)
    source = SourceRef(
        kind="workspace-file",
        workspace_id=workspace_id,
        path="evidence.md",
        content_hash=content_hash(evidence.read_bytes()),
    )
    runtime, approval_runtime = _runtime(manager)
    runtime.workspace_id = workspace_id
    runtime.wiki_tool_context = WikiToolContext(
        provenance_session_id=_SESSION_ID,
        conversation_hashes=frozenset(),
        allowed_workspace_ids=frozenset({workspace_id}),
        candidate_high_value=True,
        stable=True,
        user_confirmed=False,
        reliable_source=False,
    )

    pending = await _start_write(
        Wiki(runtime),
        _candidate(
            source=source,
            paths=("concepts/workspace-grounded.md",),
        ),
    )
    request = await _next_pending(approval_runtime)

    assert request.action == "wiki.write"
    assert cast(WikiApprovalBlock, request.display[0]).workspace_id == str(workspace_id)
    approval_runtime.resolve(request.id, "approve")
    assert not (await pending).is_error


async def test_collapsed_approval_metadata_bounds_large_page_sets(manager) -> None:
    runtime, approval_runtime = _runtime(manager)
    paths = tuple(f"concepts/bounded-{number}.md" for number in range(25))
    pending = await _start_write(Wiki(runtime), _candidate(paths=paths))

    request = await _next_pending(approval_runtime)
    block = cast(WikiApprovalBlock, request.display[0])
    assert len(block.pages) == 20
    assert block.details[-1] == "Additional pages omitted: 5."

    approval_runtime.resolve(request.id, "reject")
    assert (await pending).is_error


async def test_approve_for_session_is_scoped_to_wiki_write_and_skips_next_popup(
    manager,
) -> None:
    runtime, approval_runtime = _runtime(manager)
    tool = Wiki(runtime)
    first = await _start_write(
        tool,
        _candidate(paths=("concepts/first-approved.md",)),
    )
    request = await _next_pending(approval_runtime)

    approval_runtime.resolve(request.id, "approve_for_session")
    assert not (await first).is_error
    assert runtime.session.state.approval.auto_approve_actions == {"wiki.write"}

    with _tool_call_context():
        second = await tool(
            Params(
                operation="remember",
                candidate=_candidate(
                    marker="b",
                    paths=("concepts/session-approved.md",),
                ),
            )
        )
    assert not second.is_error
    assert approval_runtime.list_pending() == []
    assert manager.layout.revision.read_text(encoding="ascii") == "2\n"


async def test_decline_discards_candidate_without_queue_or_write(
    manager,
) -> None:
    runtime, approval_runtime = _runtime(manager)
    pending = await _start_write(Wiki(runtime), _candidate())
    request = await _next_pending(approval_runtime)

    approval_runtime.resolve(request.id, "reject")
    result = await pending

    assert isinstance(result, ToolRejectedError)
    assert manager.layout.revision.read_text(encoding="ascii") == "0\n"
    assert approval_runtime.list_pending() == []
    assert not (manager.layout.root / "concepts" / "atomic-wiki-recovery.md").exists()
    assert list((manager.layout.metadata / "journal").iterdir()) == []


async def test_cancelled_approval_commits_nothing(manager) -> None:
    runtime, approval_runtime = _runtime(manager)
    pending = await _start_write(Wiki(runtime), _candidate())
    await _next_pending(approval_runtime)

    assert approval_runtime.cancel_by_source("foreground_turn", "test") == 1
    result = await pending

    assert isinstance(result, ToolRejectedError)
    assert manager.layout.revision.read_text(encoding="ascii") == "0\n"


async def test_approval_delivery_failure_commits_nothing(manager, monkeypatch) -> None:
    runtime, _approval_runtime = _runtime(manager)

    async def fail_delivery(*_args, **_kwargs):
        raise ConnectionError("wire disconnected at /Users/private")

    monkeypatch.setattr(runtime.approval, "request", fail_delivery)
    with _tool_call_context():
        result = await Wiki(runtime)(
            Params(operation="remember", candidate=_candidate()),
        )

    assert result.is_error
    assert "/Users/private" not in result.message
    assert manager.layout.revision.read_text(encoding="ascii") == "0\n"


async def test_independent_change_rebases_only_after_approval(
    manager,
) -> None:
    runtime, approval_runtime = _runtime(manager)
    pending = await _start_write(
        Wiki(runtime),
        _candidate(paths=("concepts/approved-after-rebase.md",)),
    )
    request = await _next_pending(approval_runtime)

    independent = _candidate(
        marker="b",
        paths=("concepts/concurrent-independent.md",),
    )
    manager.commit(
        manager.prepare(
            independent,
            WikiContext(
                session_id=_SESSION_ID,
                cross_turn_utility=True,
                stable=True,
                user_confirmed=True,
            ),
        )
    )
    assert manager.layout.revision.read_text(encoding="ascii") == "1\n"

    approval_runtime.resolve(request.id, "approve")
    result = await pending

    assert not result.is_error
    assert manager.layout.revision.read_text(encoding="ascii") == "2\n"
    assert (manager.layout.root / "concepts" / "approved-after-rebase.md").is_file()
    assert (manager.layout.root / "concepts" / "concurrent-independent.md").is_file()


async def test_afk_uses_approval_path_instead_of_yolo_shortcut(manager, monkeypatch) -> None:
    runtime, _approval_runtime = _runtime(manager, afk=True)
    original_request = runtime.approval.request
    calls = 0

    async def counted_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original_request(*args, **kwargs)

    monkeypatch.setattr(runtime.approval, "request", counted_request)
    with _tool_call_context():
        result = await Wiki(runtime)(
            Params(
                operation="remember",
                candidate=_candidate(paths=("concepts/afk-approved.md",)),
            )
        )

    assert not result.is_error
    assert calls == 1
    assert manager.layout.revision.read_text(encoding="ascii") == "1\n"


async def test_yolo_high_value_commits_without_popup(manager) -> None:
    runtime, approval_runtime = _runtime(manager, yolo=True)

    with _tool_call_context():
        result = await Wiki(runtime)(
            Params(operation="remember", candidate=_candidate()),
        )

    assert not result.is_error
    assert approval_runtime.list_pending() == []
    assert manager.layout.revision.read_text(encoding="ascii") == "1\n"
    assert "operation=remember" in manager.layout.log.read_text(encoding="utf-8")


async def test_yolo_still_discards_low_value_before_write(
    manager,
) -> None:
    runtime, approval_runtime = _runtime(manager, yolo=True)

    with _tool_call_context():
        result = await Wiki(runtime)(
            Params(operation="remember", candidate=_candidate(value="low")),
        )

    assert result.is_error
    assert approval_runtime.list_pending() == []
    assert manager.layout.revision.read_text(encoding="ascii") == "0\n"
