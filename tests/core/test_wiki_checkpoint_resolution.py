"""Tests for deterministic root resolution of open Wiki checkpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

import kimi_cli.soul.kimisoul as kimisoul_module
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul, StepOutcome
from kimi_cli.tools.wiki import Params, Wiki
from kimi_cli.wiki.intent import detect_durable_intent
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wiki.models import PageChange, SourceRef, WikiCandidate, WikiPage
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import (
    OPENKIMO_WIKI_CHECKPOINT_START,
    WikiTurnCoordinator,
)


@pytest.fixture
def wiki_runtime(runtime: Runtime, tmp_path: Path):
    manager = WikiManager(tmp_path / "wiki", wal=False)
    workspace = Path(str(runtime.session.work_dir)).resolve()
    workspace_id = manager.registry.register(workspace)
    runtime.wiki = manager
    runtime.workspace_id = workspace_id
    runtime.wiki_coordinator = WikiTurnCoordinator(
        provenance_session_id=uuid4(),
        workspace_id=workspace_id,
    )
    try:
        yield runtime
    finally:
        manager.close()


def _soul(runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KimiSoul:
    monkeypatch.setattr(kimisoul_module, "wire_send", lambda _message: None)
    soul = KimiSoul(
        Agent(name="root", system_prompt="system", toolset=EmptyToolset(), runtime=runtime),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )
    monkeypatch.setattr(soul, "_checkpoint", AsyncMock())
    monkeypatch.setattr(
        soul,
        "_step",
        AsyncMock(
            return_value=StepOutcome(
                stop_reason="no_tool_calls",
                assistant_message=Message(role="assistant", content="Done."),
            )
        ),
    )
    return soul


def _release_candidate(runtime: Runtime) -> WikiCandidate:
    source = SourceRef(
        kind="conversation",
        session_id=runtime.wiki_coordinator.provenance_session_id,
        content_hash=content_hash(b"Signed tags only for every release."),
    )
    now = datetime(2026, 8, 3, tzinfo=UTC)
    return WikiCandidate(
        summary="Record the signed-tag release rule",
        pages=[
            PageChange(
                page=WikiPage(
                    logical_path="concepts/release-rule.md",
                    title="Release",
                    created=now,
                    updated=now,
                    tags=["release"],
                    sources=[source],
                    revision=1,
                    body="Signed tags only for every release.\n",
                ),
                expected_revision=None,
            )
        ],
        sources=[source],
        value="high",
    )


async def _open_durable_checkpoint(runtime: Runtime, text: str = "Remember this release rule"):
    coordinator = runtime.wiki_coordinator
    assert coordinator is not None
    await coordinator.begin_turn(text, text)
    intent = detect_durable_intent(text)
    assert intent is not None
    checkpoint = await coordinator.record_durable_intent(intent)
    assert checkpoint is not None
    return checkpoint


# ---------------------------------------------------------------------------
# Creating the explicit durable checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_durable_intent_opens_exactly_one_explicit_checkpoint(wiki_runtime) -> None:
    coordinator = wiki_runtime.wiki_coordinator

    checkpoint = await _open_durable_checkpoint(wiki_runtime)

    assert checkpoint.cause == "explicit_user_durable"
    assert checkpoint.intent_family == "remember"
    assert checkpoint.evidence_ids == ()
    assert len((await coordinator.pending_batch()).checkpoints) == 1


@pytest.mark.asyncio
async def test_repeating_the_same_intent_in_one_turn_adds_no_second_checkpoint(
    wiki_runtime,
) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    text = "Remember this release rule"
    await coordinator.begin_turn(text, text)
    intent = detect_durable_intent(text)
    assert intent is not None

    first = await coordinator.record_durable_intent(intent)
    second = await coordinator.record_durable_intent(intent)

    assert first is not None and second is not None
    assert first.checkpoint_id == second.checkpoint_id
    assert len((await coordinator.pending_batch()).checkpoints) == 1


@pytest.mark.asyncio
async def test_intent_from_another_turn_cannot_open_a_checkpoint(wiki_runtime) -> None:
    """A synthetic prompt or replayed reminder carries hashes of the wrong text."""
    coordinator = wiki_runtime.wiki_coordinator
    await coordinator.begin_turn("what is the release rule", "what is the release rule")
    foreign = detect_durable_intent("Remember this unrelated rule")
    assert foreign is not None

    assert await coordinator.record_durable_intent(foreign) is None
    assert (await coordinator.pending_batch()).checkpoints == ()


@pytest.mark.asyncio
async def test_durable_intent_without_an_active_turn_is_rejected(wiki_runtime) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    intent = detect_durable_intent("Remember this release rule")
    assert intent is not None

    assert await coordinator.record_durable_intent(intent) is None


@pytest.mark.asyncio
async def test_explicit_intent_is_delivered_before_other_causes(wiki_runtime) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    text = "Remember this release rule"
    turn = await coordinator.begin_turn(text, text)
    earlier = await coordinator.create_checkpoint(
        "root_evidence", summary_hash=None, evidence_ids=()
    )
    intent = detect_durable_intent(text)
    assert intent is not None
    durable = await coordinator.record_durable_intent(intent)

    batch = await coordinator.pending_batch()

    assert durable is not None
    assert batch.checkpoints[0].checkpoint_id == durable.checkpoint_id
    assert earlier.root_turn_id == turn.root_turn_id


# ---------------------------------------------------------------------------
# The bounded resolution loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_checkpoint_is_delivered_once_then_reminded_once_then_abandoned(
    wiki_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    checkpoint = await _open_durable_checkpoint(wiki_runtime)
    soul = _soul(wiki_runtime, tmp_path, monkeypatch)

    outcome = await soul._agent_loop()

    # One delivery, one reminder, then the turn finishes without a third ask.
    assert outcome.stop_reason == "no_tool_calls"
    assert soul._step.await_count == 3
    injected = [
        message.extract_text("\n") for message in soul.context.history if message.role == "user"
    ]
    blocks = [text for text in injected if OPENKIMO_WIKI_CHECKPOINT_START in text]
    assert len(blocks) == 2
    assert checkpoint.checkpoint_id in blocks[0]
    assert "still unresolved" in blocks[1]
    assert (await coordinator.pending_batch()).checkpoints == ()


@pytest.mark.asyncio
async def test_resolving_the_checkpoint_stops_the_loop_after_one_delivery(
    wiki_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    checkpoint = await _open_durable_checkpoint(wiki_runtime)
    soul = _soul(wiki_runtime, tmp_path, monkeypatch)
    resolved = False

    async def _step_then_discard() -> StepOutcome:
        nonlocal resolved
        if not resolved and (await coordinator.undelivered_pending()) == ():
            await Wiki(wiki_runtime)(
                Params(
                    operation="discard",
                    checkpoint_id=checkpoint.checkpoint_id,
                    discard_reason="low_value",
                )
            )
            resolved = True
        return StepOutcome(
            stop_reason="no_tool_calls",
            assistant_message=Message(role="assistant", content="Done."),
        )

    monkeypatch.setattr(soul, "_step", AsyncMock(side_effect=_step_then_discard))

    outcome = await soul._agent_loop()

    assert outcome.stop_reason == "no_tool_calls"
    # Deliver, model resolves, finish — no reminder needed.
    assert soul._step.await_count == 2
    assert resolved
    assert (await coordinator.pending_batch()).checkpoints == ()


@pytest.mark.asyncio
async def test_no_open_checkpoint_finishes_the_turn_immediately(
    wiki_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    await coordinator.begin_turn("just a question", "just a question")
    soul = _soul(wiki_runtime, tmp_path, monkeypatch)

    outcome = await soul._agent_loop()

    assert outcome.stop_reason == "no_tool_calls"
    assert soul._step.await_count == 1
    assert all(
        OPENKIMO_WIKI_CHECKPOINT_START not in message.extract_text("\n")
        for message in soul.context.history
    )


@pytest.mark.asyncio
async def test_runtime_without_a_coordinator_never_delivers_a_block(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runtime.wiki_coordinator is None
    soul = _soul(runtime, tmp_path, monkeypatch)

    outcome = await soul._agent_loop()

    assert outcome.stop_reason == "no_tool_calls"
    assert soul._step.await_count == 1


@pytest.mark.asyncio
async def test_a_failing_coordinator_never_blocks_the_turn(
    wiki_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _open_durable_checkpoint(wiki_runtime)
    soul = _soul(wiki_runtime, tmp_path, monkeypatch)
    monkeypatch.setattr(
        wiki_runtime.wiki_coordinator,
        "undelivered_pending",
        AsyncMock(side_effect=RuntimeError("coordinator down")),
    )

    outcome = await soul._agent_loop()

    assert outcome.stop_reason == "no_tool_calls"
    assert soul._step.await_count == 1


# ---------------------------------------------------------------------------
# Resolution through the Wiki tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_closes_the_checkpoint_and_writes_nothing(wiki_runtime) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    checkpoint = await _open_durable_checkpoint(wiki_runtime)
    revision_before = wiki_runtime.wiki.layout.revision.read_text(encoding="utf-8")

    result = await Wiki(wiki_runtime)(
        Params(
            operation="discard",
            checkpoint_id=checkpoint.checkpoint_id,
            discard_reason="not_reusable",
        )
    )

    assert not result.is_error
    assert (await coordinator.pending_batch()).checkpoints == ()
    assert wiki_runtime.wiki.layout.revision.read_text(encoding="utf-8") == revision_before


@pytest.mark.asyncio
async def test_discard_is_single_use(wiki_runtime) -> None:
    checkpoint = await _open_durable_checkpoint(wiki_runtime)
    tool = Wiki(wiki_runtime)
    params = Params(
        operation="discard",
        checkpoint_id=checkpoint.checkpoint_id,
        discard_reason="duplicate",
    )

    first = await tool(params)
    second = await tool(params)

    assert not first.is_error
    assert second.is_error
    assert "not open for resolution" in second.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        Params(operation="discard", discard_reason="low_value"),
        Params(operation="discard", checkpoint_id="attacker-chosen", discard_reason="low_value"),
    ],
)
async def test_discard_rejects_missing_or_invented_checkpoint_ids(wiki_runtime, params) -> None:
    await _open_durable_checkpoint(wiki_runtime)

    result = await Wiki(wiki_runtime)(params)

    assert result.is_error
    assert (await wiki_runtime.wiki_coordinator.pending_batch()).checkpoints != ()


@pytest.mark.asyncio
async def test_discard_requires_a_reason(wiki_runtime) -> None:
    checkpoint = await _open_durable_checkpoint(wiki_runtime)

    result = await Wiki(wiki_runtime)(
        Params(operation="discard", checkpoint_id=checkpoint.checkpoint_id)
    )

    assert result.is_error
    assert "discard_reason" in result.message


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["remember", "ingest"])
async def test_writes_require_the_checkpoint_they_resolve(wiki_runtime, operation: str) -> None:
    await _open_durable_checkpoint(wiki_runtime)
    candidate = _release_candidate(wiki_runtime)

    result = await Wiki(wiki_runtime)(
        Params(operation=operation, candidate=candidate)  # type: ignore[arg-type]
    )

    assert result.is_error
    assert "checkpoint_id" in result.message


@pytest.mark.asyncio
async def test_checkpoint_from_a_finished_turn_no_longer_resolves(wiki_runtime) -> None:
    coordinator = wiki_runtime.wiki_coordinator
    checkpoint = await _open_durable_checkpoint(wiki_runtime)
    await coordinator.begin_turn("a new prompt", "a new prompt")

    result = await Wiki(wiki_runtime)(
        Params(
            operation="discard",
            checkpoint_id=checkpoint.checkpoint_id,
            discard_reason="low_value",
        )
    )

    assert result.is_error


@pytest.mark.asyncio
async def test_subagent_cannot_discard_a_root_checkpoint(wiki_runtime) -> None:
    checkpoint = await _open_durable_checkpoint(wiki_runtime)
    subagent = wiki_runtime.copy_for_subagent(agent_id="worker", subagent_type="coder")

    result = await Wiki(subagent)(
        Params(
            operation="discard",
            checkpoint_id=checkpoint.checkpoint_id,
            discard_reason="low_value",
        )
    )

    assert result.is_error
    assert "root agent only" in result.message
    assert (await wiki_runtime.wiki_coordinator.pending_batch()).checkpoints != ()


@pytest.mark.asyncio
async def test_evidence_checkpoints_cost_no_extra_model_call(
    wiki_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only explicit intent is worth interrupting a turn for.

    A root_evidence checkpoint arises on nearly every turn that reads a file.
    Delivering one costs a completion, and a second if it is ignored, so
    charging that to every conversation is a bad trade — they are retired with
    the turn instead.
    """
    coordinator = wiki_runtime.wiki_coordinator
    await coordinator.begin_turn("look something up", "look something up")
    await coordinator.create_checkpoint(
        "root_evidence", summary_hash=content_hash(b"a reusable conclusion")
    )
    soul = _soul(wiki_runtime, tmp_path, monkeypatch)

    outcome = await soul._agent_loop()

    assert outcome.stop_reason == "no_tool_calls"
    assert soul._step.await_count == 1, "an evidence checkpoint must not add a completion"
    assert all(
        OPENKIMO_WIKI_CHECKPOINT_START not in message.extract_text("\n")
        for message in soul.context.history
    )


@pytest.mark.asyncio
async def test_explicit_intent_still_interrupts_the_turn(
    wiki_runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the user actually asked to keep is still worth the round trip."""
    await _open_durable_checkpoint(wiki_runtime)
    soul = _soul(wiki_runtime, tmp_path, monkeypatch)

    await soul._agent_loop()

    assert soul._step.await_count == 3  # deliver, remind, finish
