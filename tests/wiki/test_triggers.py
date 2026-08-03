"""Tests for trusted Wiki turn, evidence, and checkpoint state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import cast
from uuid import UUID, uuid4

import pytest

from kimi_cli.wiki.models import SourceRef
from kimi_cli.wiki.triggers import (
    CheckpointCause,
    CheckpointDiscardReason,
    EvidenceClass,
    EvidenceObservation,
    ProducerRole,
    WikiCheckpointBackpressure,
    WikiTriggerRejected,
    WikiTurnCoordinator,
    _source_key,
    canonical_digest,
    canonical_evidence_source_digest,
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
    logical_paths: tuple[str, ...] = ("docs/decision.md",),
    source_class: EvidenceClass = "workspace-file",
    source_refs: tuple[SourceRef, ...] | None = None,
) -> EvidenceObservation:
    return EvidenceObservation(
        root_turn_id=root_turn_id,
        producer_role="root",
        producer_id=None,
        run_generation=None,
        tool_call_id=tool_call_id,
        source_class=source_class,
        request_hash="sha256:" + "1" * 64,
        result_hash="sha256:" + "2" * 64,
        logical_paths=logical_paths,
        source_refs=source_refs if source_refs is not None else (workspace_source,),
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
    assert coordinator.active_turn_id is None
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


async def test_evidence_source_digest_separates_class_paths_and_source_groups(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
) -> None:
    turn = await coordinator.begin_turn("digest identity", "digest identity")
    shared = {"root_turn_id": turn.root_turn_id, "workspace_source": workspace_source}
    baseline = await coordinator.record_evidence(
        make_observation(**shared, tool_call_id="call-baseline")
    )
    other_class = await coordinator.record_evidence(
        make_observation(**shared, tool_call_id="call-class", source_class="workspace-search")
    )
    other_path = await coordinator.record_evidence(
        make_observation(**shared, tool_call_id="call-path", logical_paths=("docs/other.md",))
    )
    group_shift = await coordinator.record_evidence(
        make_observation(
            **shared,
            tool_call_id="call-group",
            logical_paths=(_source_key(workspace_source),),
            source_refs=(),
        )
    )
    empty_paths = await coordinator.record_evidence(
        make_observation(**shared, tool_call_id="call-empty", logical_paths=())
    )
    assert baseline is not None
    assert other_class is not None
    assert other_path is not None
    assert group_shift is not None
    assert empty_paths is not None

    digests = {
        canonical_evidence_source_digest(evidence)
        for evidence in (baseline, other_class, other_path, group_shift, empty_paths)
    }

    assert len(digests) == 5
    assert canonical_evidence_source_digest(baseline) == canonical_evidence_source_digest(
        replace(baseline, source_class=cast(EvidenceClass, "workspace_file"))
    )


async def test_record_evidence_snapshots_mutable_source_refs_at_every_boundary(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
) -> None:
    turn = await coordinator.begin_turn("snapshot", "snapshot")
    evidence = await coordinator.record_evidence(
        make_observation(turn.root_turn_id, workspace_source)
    )
    assert evidence is not None

    workspace_source.path = "docs/caller-mutated.md"
    evidence.source_refs[0].path = "docs/output-mutated.md"
    duplicate = await coordinator.record_evidence(
        make_observation(
            turn.root_turn_id,
            SourceRef(
                kind="workspace-file",
                workspace_id=coordinator.workspace_id,
                path="docs/decision.md",
                content_hash="sha256:" + "4" * 64,
            ),
        )
    )

    assert duplicate is not None
    assert duplicate.source_refs[0].path == "docs/decision.md"


async def test_record_evidence_rejects_cross_session_conversation_source(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
) -> None:
    turn = await coordinator.begin_turn("session", "session")
    foreign = SourceRef(
        kind="conversation",
        session_id=uuid4(),
        content_hash="sha256:" + "5" * 64,
    )

    with pytest.raises(WikiTriggerRejected, match="session"):
        await coordinator.record_evidence(
            make_observation(
                turn.root_turn_id,
                workspace_source,
                logical_paths=(),
                source_class="shell-result",
                source_refs=(foreign,),
            )
        )


async def test_record_evidence_rejects_mismatched_workspace_source_kind_and_producer_role(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
) -> None:
    turn = await coordinator.begin_turn("bindings", "bindings")
    foreign_workspace = SourceRef(
        kind="workspace-file",
        workspace_id=uuid4(),
        path="docs/decision.md",
        content_hash="sha256:" + "6" * 64,
    )
    web_source = SourceRef(
        kind="web",
        url="https://example.test/decision",
        content_hash="sha256:" + "7" * 64,
    )

    with pytest.raises(WikiTriggerRejected, match="workspace"):
        await coordinator.record_evidence(make_observation(turn.root_turn_id, foreign_workspace))
    with pytest.raises(WikiTriggerRejected, match="source kind"):
        await coordinator.record_evidence(
            make_observation(turn.root_turn_id, workspace_source, source_refs=(web_source,))
        )
    with pytest.raises(WikiTriggerRejected, match="subagent identity"):
        await coordinator.record_evidence(
            replace(
                make_observation(turn.root_turn_id, workspace_source),
                producer_id="untrusted-subagent",
                run_generation=0,
            )
        )


@pytest.mark.parametrize("logical_path", ("/private/decision.md", "config/.env"))
async def test_record_evidence_rejects_absolute_or_sensitive_logical_paths(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
    logical_path: str,
) -> None:
    turn = await coordinator.begin_turn("paths", "paths")

    with pytest.raises(WikiTriggerRejected, match="logical path"):
        await coordinator.record_evidence(
            make_observation(
                turn.root_turn_id,
                workspace_source,
                logical_paths=(logical_path,),
            )
        )


async def test_discard_and_close_clear_active_turn_and_pending_checkpoints(
    coordinator: WikiTurnCoordinator,
) -> None:
    turn = await coordinator.begin_turn("discard", "discard")
    checkpoint = await coordinator.create_checkpoint("root_evidence")
    assert coordinator.active_turn_id == turn.root_turn_id

    await coordinator.discard(checkpoint.checkpoint_id, "not_useful")
    assert (await coordinator.pending_batch()).checkpoints == ()

    await coordinator.close()
    assert coordinator.active_turn_id is None
    with pytest.raises(WikiTriggerRejected, match="closed"):
        await coordinator.begin_turn("closed", "closed")


@pytest.mark.parametrize("role", ("worker", "", "ROOT"))
async def test_record_evidence_rejects_unknown_producer_roles_at_runtime(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
    role: str,
) -> None:
    turn = await coordinator.begin_turn("roles", "roles")

    with pytest.raises(WikiTriggerRejected, match="producer role"):
        await coordinator.record_evidence(
            replace(
                make_observation(turn.root_turn_id, workspace_source),
                producer_role=cast(ProducerRole, role),
            )
        )


@pytest.mark.parametrize("run_generation", (True, False, 1.0, "1", -1))
async def test_record_evidence_rejects_non_integer_or_negative_run_generations(
    coordinator: WikiTurnCoordinator,
    workspace_source: SourceRef,
    run_generation: object,
) -> None:
    turn = await coordinator.begin_turn("generations", "generations")
    observation = replace(
        make_observation(turn.root_turn_id, workspace_source),
        producer_role="subagent",
        producer_id="subagent-1",
        run_generation=cast(int | None, run_generation),
    )

    with pytest.raises(WikiTriggerRejected, match="run generation"):
        await coordinator.record_evidence(observation)


async def test_checkpoint_rejects_unknown_cause_at_runtime(
    coordinator: WikiTurnCoordinator,
) -> None:
    await coordinator.begin_turn("cause", "cause")

    with pytest.raises(WikiTriggerRejected, match="checkpoint cause"):
        await coordinator.create_checkpoint(cast(CheckpointCause, "background"))

    assert (await coordinator.pending_batch()).checkpoints == ()


async def test_discard_rejects_unknown_reason_before_telemetry() -> None:
    events: list[str] = []
    coordinator = WikiTurnCoordinator(
        provenance_session_id=uuid4(),
        telemetry_track=lambda event, **_properties: events.append(event),
    )
    await coordinator.begin_turn("discard", "discard")
    checkpoint = await coordinator.create_checkpoint("root_evidence")
    event_count_before = len(events)

    with pytest.raises(WikiTriggerRejected, match="discard reason"):
        await coordinator.discard(
            checkpoint.checkpoint_id,
            cast(CheckpointDiscardReason, "model-supplied"),
        )

    assert len(events) == event_count_before
    assert (await coordinator.pending_batch()).checkpoints == (checkpoint,)
