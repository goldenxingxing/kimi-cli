"""Lifecycle, isolation, and shutdown tests for the Wiki trigger runtime."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.wiki import Params, Wiki, WikiToolContext
from kimi_cli.wiki import telemetry as wiki_telemetry
from kimi_cli.wiki.evidence import WikiEvidenceReporter
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wiki.models import PageChange, WikiCandidate, WikiPage
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import (
    EvidenceObservation,
    WikiTriggerRejected,
    WikiTurnCoordinator,
)

_NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _attach_wiki(runtime: Runtime, root: Path) -> WikiManager:
    manager = WikiManager(root, wal=False)
    workspace = Path(str(runtime.session.work_dir)).resolve()
    workspace_id = manager.registry.register(workspace)
    coordinator = WikiTurnCoordinator(
        provenance_session_id=uuid4(),
        workspace_id=workspace_id,
    )
    runtime.wiki = manager
    runtime.workspace_id = workspace_id
    runtime.wiki_coordinator = coordinator
    runtime.wiki_evidence_reporter = WikiEvidenceReporter(coordinator, runtime)
    runtime.wiki_tool_context = WikiToolContext(
        provenance_session_id=coordinator.provenance_session_id,
        conversation_hashes=frozenset(),
        allowed_workspace_ids=frozenset({workspace_id}),
        candidate_high_value=True,
        stable=True,
        user_confirmed=True,
        reliable_source=True,
    )
    return manager


@pytest.fixture
def wiki_runtime(runtime: Runtime, tmp_path: Path):
    manager = _attach_wiki(runtime, tmp_path / "wiki")
    try:
        yield runtime
    finally:
        manager.close()


async def _admitted_write(runtime: Runtime, *, name: str = "decision.md"):
    """Open a grounded checkpoint and return params that would commit it."""
    coordinator = runtime.wiki_coordinator
    path = Path(str(runtime.session.work_dir)) / name
    path.write_text("durable decision", encoding="utf-8")
    await coordinator.begin_turn("where is the rule", "where is the rule")
    source = runtime.wiki.registry.relative_source(runtime.workspace_id, path)
    evidence = await coordinator.record_evidence(
        EvidenceObservation(
            root_turn_id=coordinator.active_turn_id,
            workspace_id=runtime.workspace_id,
            producer_role="root",
            producer_id=None,
            run_generation=None,
            tool_call_id=f"call-{name}",
            source_class="workspace-file",
            request_hash=content_hash(b"read"),
            result_hash=source.content_hash,
            logical_paths=(source.path,),
            source_refs=(source,),
            reliable=True,
            stable_snapshot=True,
            triggering=True,
        )
    )
    assert evidence is not None
    checkpoint = await coordinator.create_checkpoint(
        "root_evidence",
        evidence_ids=(evidence.evidence_id,),
        summary_hash=content_hash(b"the rule lives in a file"),
    )
    page = WikiPage(
        logical_path="concepts/release-rule.md",
        title="Release rule",
        created=_NOW,
        updated=_NOW,
        tags=["release"],
        sources=[source],
        revision=1,
        body="Signed tags only for every release.\n",
    )
    candidate = WikiCandidate(
        summary="Record the signed-tag release rule",
        pages=[PageChange(page=page, expected_revision=None)],
        sources=[source],
        value="high",
    )
    return checkpoint, Params(
        operation="remember",
        checkpoint_id=checkpoint.checkpoint_id,
        candidate=candidate,
    )


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_root_sessions_cannot_consume_each_others_checkpoint(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    left_manager = _attach_wiki(runtime, tmp_path / "wiki")
    try:
        checkpoint, params = await _admitted_write(runtime)
        left = runtime.wiki_coordinator
        # A second root runtime over the same shared Wiki root.
        right = WikiTurnCoordinator(
            provenance_session_id=uuid4(),
            workspace_id=runtime.workspace_id,
        )
        await right.begin_turn("other session", "other session")
        runtime.wiki_coordinator = right

        result = await Wiki(runtime)(params)

        assert result.is_error
        runtime.wiki_coordinator = left
        assert len((await left.pending_batch()).checkpoints) == 1
        assert left.unconsumed_grant_count == 0
        assert checkpoint.state == "pending"
    finally:
        left_manager.close()


@pytest.mark.asyncio
async def test_evidence_cannot_cross_coordinators(wiki_runtime) -> None:
    checkpoint, _ = await _admitted_write(wiki_runtime)
    stranger = WikiTurnCoordinator(
        provenance_session_id=uuid4(),
        workspace_id=wiki_runtime.workspace_id,
    )
    await stranger.begin_turn("other", "other")

    with pytest.raises(WikiTriggerRejected):
        await stranger.create_checkpoint(
            "root_evidence",
            evidence_ids=(checkpoint.evidence_ids[0],),
        )


# ---------------------------------------------------------------------------
# Cancellation fails closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelling_the_turn_leaves_no_grant_and_no_write(wiki_runtime) -> None:
    checkpoint, params = await _admitted_write(wiki_runtime)
    coordinator = wiki_runtime.wiki_coordinator
    before = wiki_runtime.wiki.layout.revision.read_text(encoding="utf-8")

    await coordinator.cancel_turn(checkpoint.root_turn_id)
    result = await Wiki(wiki_runtime)(params)

    assert result.is_error
    assert coordinator.unconsumed_grant_count == 0
    assert wiki_runtime.wiki.layout.revision.read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_cancelled_mid_commit_leaves_no_outstanding_authority(
    wiki_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, params = await _admitted_write(wiki_runtime)
    coordinator = wiki_runtime.wiki_coordinator

    def _cancel(*_args: Any, **_kwargs: Any):
        raise asyncio.CancelledError

    monkeypatch.setattr(wiki_runtime.wiki, "commit", _cancel)

    with pytest.raises(asyncio.CancelledError):
        await Wiki(wiki_runtime)(params)

    assert coordinator.unconsumed_grant_count == 0


@pytest.mark.asyncio
async def test_a_failing_prepare_leaves_no_outstanding_authority(
    wiki_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, params = await _admitted_write(wiki_runtime)
    coordinator = wiki_runtime.wiki_coordinator

    def _boom(*_args: Any, **_kwargs: Any):
        raise RuntimeError("prepare exploded")

    monkeypatch.setattr(wiki_runtime.wiki, "prepare", _boom)

    result = await Wiki(wiki_runtime)(params)

    assert result.is_error
    assert coordinator.unconsumed_grant_count == 0


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_is_idempotent_and_creates_no_checkpoint_or_write(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _attach_wiki(runtime, tmp_path / "wiki")
    checkpoint, _ = await _admitted_write(runtime)
    coordinator = runtime.wiki_coordinator
    revision_path = runtime.wiki.layout.revision
    before = revision_path.read_text(encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(wiki_telemetry, "track", lambda event, **_: events.append(event))

    await runtime.close()
    await runtime.close()

    assert revision_path.read_text(encoding="utf-8") == before
    assert coordinator.unconsumed_grant_count == 0
    # Shutdown only invalidates; it never opens new work.
    assert "wiki_checkpoint_created" not in events
    assert "wiki_committed" not in events
    assert events.count("wiki_checkpoint_resolved") == 1
    assert checkpoint.cause == "root_evidence"


@pytest.mark.asyncio
async def test_close_still_closes_the_manager_when_the_coordinator_raises(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _attach_wiki(runtime, tmp_path / "wiki")
    manager = runtime.wiki
    closed: list[bool] = []
    monkeypatch.setattr(manager, "close", lambda: closed.append(True))

    async def _boom() -> None:
        raise RuntimeError("coordinator shutdown failed")

    monkeypatch.setattr(runtime.wiki_coordinator, "close", _boom)

    await runtime.close()

    assert closed == [True]


@pytest.mark.asyncio
async def test_a_subagent_close_never_touches_root_resources(
    wiki_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subagent = wiki_runtime.copy_for_subagent(agent_id="worker", subagent_type="coder")
    closed: list[str] = []
    monkeypatch.setattr(wiki_runtime.wiki, "close", lambda: closed.append("manager"))

    await subagent.close()

    assert closed == []
    assert wiki_runtime.wiki is not None


@pytest.mark.asyncio
async def test_a_closed_coordinator_refuses_every_transition(wiki_runtime) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    await coordinator.begin_turn("a prompt", "a prompt")

    await coordinator.close()

    with pytest.raises(WikiTriggerRejected):
        await coordinator.begin_turn("another", "another")
    assert await coordinator.resolvable_checkpoint("anything") is None


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unresolved_checkpoints_apply_backpressure_without_eviction(
    wiki_runtime,
) -> None:
    from kimi_cli.wiki.triggers import WikiCheckpointBackpressure

    coordinator = wiki_runtime.wiki_coordinator
    await coordinator.begin_turn("many", "many")
    for index in range(16):
        await coordinator.create_checkpoint(
            "root_evidence", summary_hash=content_hash(f"conclusion {index}".encode())
        )

    with pytest.raises(WikiCheckpointBackpressure):
        await coordinator.create_checkpoint(
            "root_evidence", summary_hash=content_hash(b"one too many")
        )

    # Nothing was silently evicted to make room.
    assert coordinator.unresolved_count == 16


@pytest.mark.asyncio
async def test_a_rendered_batch_stays_within_its_budgets(wiki_runtime) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    await coordinator.begin_turn("many", "many")
    for index in range(8):
        await coordinator.create_checkpoint(
            "root_evidence", summary_hash=content_hash(f"结论 {index}".encode())
        )

    batch = await coordinator.pending_batch()

    assert len(batch.checkpoints) <= 4
    assert len(batch.rendered.encode("utf-8")) <= 6 * 1024
