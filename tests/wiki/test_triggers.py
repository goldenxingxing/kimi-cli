"""Tests for trusted Wiki turn, evidence, and checkpoint state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import pytest

from kimi_cli.wiki.models import SourceRef
from kimi_cli.wiki.triggers import (
    EvidenceObservation,
    WikiCheckpointBackpressure,
    WikiTriggerRejected,
    WikiTurnCoordinator,
    canonical_digest,
)


@pytest.fixture
def provenance_session_id() -> UUID:
    return uuid4()


@pytest.fixture
def workspace_id() -> UUID:
    return uuid4()


@pytest.fixture
def workspace_source(workspace_id: UUID) -> SourceRef:
    return SourceRef(
        kind="workspace-file",
        workspace_id=workspace_id,
        path="docs/decision.md",
        content_hash="sha256:" + "4" * 64,
    )


@pytest.fixture
def coordinator(provenance_session_id: UUID, workspace_id: UUID) -> WikiTurnCoordinator:
    return WikiTurnCoordinator(
        provenance_session_id=provenance_session_id,
        workspace_id=workspace_id,
    )


def make_observation(
    root_turn_id: str,
    workspace_source: SourceRef,
    *,
    tool_call_id: str = "call-1",
) -> EvidenceObservation:
    return EvidenceObservation(
        root_turn_id=root_turn_id,
        producer_role="root",
        producer_id=None,
        run_generation=None,
        tool_call_id=tool_call_id,
        source_class="workspace-file",
        request_hash="sha256:" + "1" * 64,
        result_hash="sha256:" + "2" * 64,
        logical_paths=("docs/decision.md",),
        source_refs=(workspace_source,),
        reliable=True,
        stable_snapshot=True,
        triggering=True,
    )


async def test_checkpoint_identity_is_turn_session_workspace_and_evidence_bound(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
    provenance_session_id: UUID,
    workspace_id: UUID,
) -> None:
    turn = await coordinator.begin_turn("remember this", "remember this")
    evidence = await coordinator.record_evidence(
        make_observation(turn.root_turn_id, workspace_source)
    )
    assert evidence is not None
    checkpoint = await coordinator.create_checkpoint(
        "root_evidence",
        evidence_ids=(evidence.evidence_id,),
        summary_hash="sha256:" + "3" * 64,
    )

    assert checkpoint.root_turn_id == turn.root_turn_id
    assert checkpoint.dedupe_key.startswith("sha256:")
    assert evidence.session_provenance_id == provenance_session_id
    assert evidence.workspace_id == workspace_id
    assert (
        await coordinator.create_checkpoint(
            "root_evidence",
            evidence_ids=(evidence.evidence_id,),
            summary_hash="sha256:" + "3" * 64,
        )
        == checkpoint
    )
    with pytest.raises(FrozenInstanceError):
        checkpoint.state = "cancelled"  # type: ignore[misc]


async def test_coordinator_rejects_cross_runtime_evidence_and_cancellation_hides_pending(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
) -> None:
    other = WikiTurnCoordinator(provenance_session_id=uuid4(), workspace_id=uuid4())
    turn = await coordinator.begin_turn("x", "x")
    evidence = await coordinator.record_evidence(
        make_observation(turn.root_turn_id, workspace_source)
    )
    assert evidence is not None

    with pytest.raises(WikiTriggerRejected, match="runtime"):
        await other.import_evidence(evidence)

    await coordinator.create_checkpoint("root_evidence", evidence_ids=(evidence.evidence_id,))
    await coordinator.cancel_turn(turn.root_turn_id)
    assert (await coordinator.pending_batch()).checkpoints == ()


async def test_pending_batch_prioritizes_explicit_intent_and_enforces_all_budgets(
    coordinator: WikiTurnCoordinator,
) -> None:
    for index in range(16):
        await coordinator.begin_turn(f"turn {index}", f"turn {index}")
        cause = "explicit_user_durable" if index == 15 else "root_evidence"
        await coordinator.create_checkpoint(cause)

    batch = await coordinator.pending_batch()

    assert len(batch.checkpoints) == 4
    assert batch.checkpoints[0].cause == "explicit_user_durable"
    assert len(batch.rendered.encode("utf-8")) <= 6 * 1024
    assert coordinator.unresolved_count == 16


async def test_qualifying_checkpoint_over_capacity_is_rejected_without_eviction(
    coordinator: WikiTurnCoordinator,
) -> None:
    for index in range(16):
        await coordinator.begin_turn(f"turn {index}", f"turn {index}")
        await coordinator.create_checkpoint("root_evidence")

    overflow = await coordinator.begin_turn("overflow", "overflow")
    with pytest.raises(WikiCheckpointBackpressure):
        await coordinator.create_checkpoint("root_evidence")

    assert coordinator.unresolved_count == 16
    await coordinator.cancel_turn(overflow.root_turn_id)


def test_canonical_digest_is_length_prefixed_and_ordered() -> None:
    assert canonical_digest(("ab", "c")) != canonical_digest(("a", "bc"))
    assert canonical_digest(("first", "second")) != canonical_digest(("second", "first"))
    assert canonical_digest(("same",)) == canonical_digest(("same",))
