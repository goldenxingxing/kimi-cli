"""Tests for privacy-safe, exactly-once Wiki trigger observability."""

from __future__ import annotations

import json
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
from kimi_cli.wiki.retrieval import retrieve_for_turn
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.telemetry import allowed_fields, track_wiki_event
from kimi_cli.wiki.triggers import EvidenceObservation, WikiTurnCoordinator

_NOW = datetime(2026, 8, 3, tzinfo=UTC)


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    def _track(event: str, **properties: Any) -> None:
        events.append((event, properties))

    monkeypatch.setattr(wiki_telemetry, "track", _track)
    return events


@pytest.fixture
def observed_runtime(runtime: Runtime, tmp_path: Path):
    manager = WikiManager(tmp_path / "wiki", wal=False)
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
    try:
        yield runtime
    finally:
        manager.close()


def _names(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [name for name, _ in events]


async def _durable_turn(runtime: Runtime):
    """Read a real file, open a checkpoint on it, and return the write params."""
    coordinator = runtime.wiki_coordinator
    workspace = Path(str(runtime.session.work_dir))
    path = workspace / "decision.md"
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
            tool_call_id="call-read",
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
        summary_hash=content_hash(b"the rule lives in decision.md"),
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
# The allowlist itself
# ---------------------------------------------------------------------------


def test_unknown_events_are_never_emitted(captured_events) -> None:
    track_wiki_event("not_a_wiki_event", anything="leaked")  # type: ignore[arg-type]

    assert captured_events == []


def test_fields_outside_the_allowlist_are_dropped(captured_events) -> None:
    track_wiki_event(
        "wiki_checkpoint_created",
        cause="root_evidence",
        checkpoint_id="abc",
        evidence_count=2,
        raw_prompt="the user's actual words",
        page_body="the whole page",
        absolute_path="/Users/qunwei/secret/notes.md",
    )

    assert len(captured_events) == 1
    name, fields = captured_events[0]
    assert name == "wiki_checkpoint_created"
    assert set(fields) <= allowed_fields("wiki_checkpoint_created")
    assert "raw_prompt" not in fields
    assert "page_body" not in fields
    assert "absolute_path" not in fields


def test_a_failing_telemetry_sink_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(event: str, **properties: Any) -> None:
        raise RuntimeError("sink is down")

    monkeypatch.setattr(wiki_telemetry, "track", _boom)

    track_wiki_event("wiki_retrieval_miss", reason="empty")


# ---------------------------------------------------------------------------
# Event order and exactly-once terminal outcomes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_write_emits_a_deterministic_event_sequence(
    observed_runtime,
    captured_events,
) -> None:
    _, params = await _durable_turn(observed_runtime)
    captured_events.clear()

    result = await Wiki(observed_runtime)(params)

    assert not result.is_error
    assert _names(captured_events) == [
        "wiki_committed",
        "wiki_checkpoint_resolved",
    ]
    assert captured_events[-1][1]["outcome"] == "persist"


@pytest.mark.asyncio
async def test_evidence_and_checkpoint_events_fire_at_their_transitions(
    observed_runtime,
    captured_events,
) -> None:
    await _durable_turn(observed_runtime)

    names = _names(captured_events)
    assert names.count("wiki_evidence_recorded") == 1
    assert names.count("wiki_checkpoint_created") == 1
    # Observing evidence must not, by itself, claim a checkpoint was created.
    assert names.index("wiki_evidence_recorded") < names.index("wiki_checkpoint_created")
    evidence_fields = dict(captured_events[names.index("wiki_evidence_recorded")][1])
    assert evidence_fields["producer_role"] == "root"
    assert evidence_fields["evidence_class"] == "workspace-file"
    assert evidence_fields["source_count"] == 1


@pytest.mark.asyncio
async def test_a_discarded_checkpoint_resolves_exactly_once(
    observed_runtime,
    captured_events,
) -> None:
    checkpoint, _ = await _durable_turn(observed_runtime)
    captured_events.clear()
    tool = Wiki(observed_runtime)
    discard = Params(
        operation="discard",
        checkpoint_id=checkpoint.checkpoint_id,
        discard_reason="low_value",
    )

    await tool(discard)
    await tool(discard)

    resolved = [fields for name, fields in captured_events if name == "wiki_checkpoint_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["outcome"] == "discard"
    assert resolved[0]["checkpoint_id"] == checkpoint.checkpoint_id


@pytest.mark.asyncio
async def test_cancelling_a_turn_resolves_its_checkpoints_as_cancelled(
    observed_runtime,
    captured_events,
) -> None:
    checkpoint, _ = await _durable_turn(observed_runtime)
    captured_events.clear()

    await observed_runtime.wiki_coordinator.cancel_turn(checkpoint.root_turn_id)

    resolved = [fields for name, fields in captured_events if name == "wiki_checkpoint_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_abandoning_an_unresolved_checkpoint_reports_unresolved(
    observed_runtime,
    captured_events,
) -> None:
    coordinator = observed_runtime.wiki_coordinator
    checkpoint, _ = await _durable_turn(observed_runtime)
    await coordinator.mark_delivered([checkpoint.checkpoint_id])
    await coordinator.mark_delivered([checkpoint.checkpoint_id])
    captured_events.clear()

    await coordinator.abandon_unresolved()

    resolved = [fields for name, fields in captured_events if name == "wiki_checkpoint_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["outcome"] == "unresolved"


@pytest.mark.asyncio
async def test_a_rejected_candidate_reports_discarded_with_its_reason(
    observed_runtime,
    captured_events,
) -> None:
    checkpoint, params = await _durable_turn(observed_runtime)
    # Strip the grounding so the value gate rejects it after admission.
    unusable = params.candidate.model_copy(update={"value": "low"})
    captured_events.clear()

    result = await Wiki(observed_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=unusable,
        )
    )

    assert result.is_error
    discarded = [fields for name, fields in captured_events if name == "wiki_candidate_discarded"]
    assert len(discarded) == 1
    assert discarded[0]["reason"] == "low_value"
    assert discarded[0]["checkpoint_id"] == checkpoint.checkpoint_id


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_reports_a_miss_on_an_empty_wiki(
    observed_runtime,
    captured_events,
) -> None:
    coordinator = observed_runtime.wiki_coordinator
    await coordinator.begin_turn("durable architecture", "durable architecture")
    captured_events.clear()

    result = await retrieve_for_turn(
        observed_runtime.wiki, coordinator, "durable architecture notes"
    )

    assert result is None
    assert _names(captured_events) == ["wiki_retrieval_miss"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [({"synthetic": True}, "synthetic"), ({"slash_command": True}, "slash_command")],
)
async def test_retrieval_reports_why_it_was_skipped(
    observed_runtime,
    captured_events,
    kwargs: dict[str, bool],
    reason: str,
) -> None:
    coordinator = observed_runtime.wiki_coordinator
    await coordinator.begin_turn("a prompt", "a prompt")
    captured_events.clear()

    await retrieve_for_turn(observed_runtime.wiki, coordinator, "a prompt", **kwargs)

    assert _names(captured_events) == ["wiki_retrieval_skipped"]
    assert captured_events[0][1]["reason"] == reason


@pytest.mark.asyncio
async def test_a_sensitive_prompt_is_skipped_without_recording_the_prompt(
    observed_runtime,
    captured_events,
) -> None:
    coordinator = observed_runtime.wiki_coordinator
    await coordinator.begin_turn("secret", "secret")
    captured_events.clear()

    await retrieve_for_turn(
        observed_runtime.wiki,
        coordinator,
        "api_key=sk-abcdefghijklmnopqrstuvwx please search",
    )

    assert _names(captured_events) == ["wiki_retrieval_skipped"]
    assert "sk-abcdefghijklmnopqrstuvwx" not in json.dumps(captured_events)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_wiki_event_ever_carries_a_sensitive_payload(
    observed_runtime,
    captured_events,
) -> None:
    checkpoint, params = await _durable_turn(observed_runtime)
    tool = Wiki(observed_runtime)
    await tool(params)
    await retrieve_for_turn(observed_runtime.wiki, observed_runtime.wiki_coordinator, "release")

    serialized = json.dumps(captured_events, default=str)
    for forbidden in (
        str(observed_runtime.session.work_dir),
        "/Users/",
        "C:\\\\Users",
        "api_key",
        "Bearer ",
        "durable decision",
        "Signed tags only",
        "where is the rule",
        "decision.md",
        checkpoint.summary_hash or "sha256:absent",
    ):
        assert forbidden not in serialized, forbidden


@pytest.mark.asyncio
async def test_telemetry_failure_never_changes_the_commit_outcome(
    observed_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, params = await _durable_turn(observed_runtime)

    def _boom(event: str, **properties: Any) -> None:
        raise RuntimeError("sink is down")

    monkeypatch.setattr(wiki_telemetry, "track", _boom)

    result = await Wiki(observed_runtime)(params)

    assert not result.is_error
    assert observed_runtime.wiki.layout.revision.read_text(encoding="utf-8").strip() != "0"


@pytest.mark.asyncio
async def test_telemetry_failure_never_blocks_retrieval(
    observed_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = observed_runtime.wiki_coordinator
    await coordinator.begin_turn("a prompt", "a prompt")

    def _boom(event: str, **properties: Any) -> None:
        raise RuntimeError("sink is down")

    monkeypatch.setattr(wiki_telemetry, "track", _boom)

    assert await retrieve_for_turn(observed_runtime.wiki, coordinator, "a prompt") is None
