"""Tests for bridging subagent evidence into root Wiki checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest
from kosong.tooling import ToolOk

from kimi_cli.background.models import (
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    WikiEvidenceManifest,
)
from kimi_cli.notifications import build_notification_message
from kimi_cli.notifications.models import (
    NotificationDelivery,
    NotificationEvent,
    NotificationView,
)
from kimi_cli.soul.agent import BuiltinSystemPromptArgs
from kimi_cli.subagents.models import AgentLaunchSpec
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.wiki import Wiki
from kimi_cli.wiki.evidence import WikiEvidenceReporter
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import PersistedEvidenceRef, WikiTurnCoordinator


@pytest.fixture
def root_runtime(runtime, tmp_path: Path):
    """A root runtime with a live Wiki manager and turn coordinator."""
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
    runtime.builtin_args = BuiltinSystemPromptArgs(
        **{
            **{
                field: getattr(runtime.builtin_args, field)
                for field in runtime.builtin_args.__dataclass_fields__
            },
            "KIMI_WIKI_CONTEXT": "<OPENKIMO_GLOBAL_WIKI_START>index<OPENKIMO_GLOBAL_WIKI_END>",
        }
    )
    try:
        yield runtime
    finally:
        manager.close()


def _subagent_reporter(root_runtime, *, agent_id: str = "a1", run_generation: int = 1):
    subagent = root_runtime.copy_for_subagent(
        agent_id=agent_id,
        subagent_type="coder",
        run_generation=run_generation,
    )
    assert subagent.wiki_evidence_reporter is not None
    return subagent


async def _observe_workspace_file(subagent, name: str = "decision.md") -> None:
    workspace = Path(str(subagent.session.work_dir))
    path = workspace / name
    path.write_text(f"durable {name}", encoding="utf-8")
    reporter = subagent.wiki_evidence_reporter
    assert reporter is not None
    await reporter.observe(
        ReadFile(subagent),
        {"path": name},
        ToolOk(output=f"durable {name}"),
        tool_call_id=f"call-{name}",
    )


def _write_agent_task(runtime, task_id: str, agent_id: str, *, status: TaskStatus) -> None:
    store = runtime.background_tasks.store
    store.create_task(
        TaskSpec(
            id=task_id,
            kind="agent",
            session_id=runtime.session.id,
            description="background agent task",
            tool_call_id="tool-agent",
            kind_payload={"agent_id": agent_id, "subagent_type": "coder", "prompt": "work"},
        )
    )
    state = TaskRuntime(status=status, updated_at=time.time())
    state.finished_at = time.time()
    store.write_runtime(task_id, state)


# ---------------------------------------------------------------------------
# Subagent runtime isolation
# ---------------------------------------------------------------------------


def test_subagent_runtime_has_no_wiki_authority(root_runtime) -> None:
    subagent = _subagent_reporter(root_runtime)

    assert subagent.role == "subagent"
    assert subagent.wiki is None
    assert subagent.wiki_tool_context is None
    assert subagent.wiki_coordinator is None
    assert subagent.wiki_evidence_reporter is not None
    # The root keeps its own authority untouched.
    assert root_runtime.wiki is not None
    assert root_runtime.wiki_coordinator is not None


def test_subagent_runtime_copy_without_root_coordinator_has_no_reporter(runtime) -> None:
    assert runtime.wiki_coordinator is None

    subagent = runtime.copy_for_subagent(agent_id="a9", subagent_type="coder")

    assert subagent.wiki_evidence_reporter is None


@pytest.mark.asyncio
async def test_wiki_tool_rejects_every_non_root_caller(root_runtime) -> None:
    subagent = _subagent_reporter(root_runtime)
    tool = Wiki(subagent)

    result = await tool(Wiki.params(operation="search", query="anything"))

    assert result.is_error
    assert "root agent only" in result.message


def test_subagent_system_prompt_args_blank_the_wiki_context(root_runtime) -> None:
    from kimi_cli.agentspec import ResolvedAgentSpec
    from kimi_cli.soul.agent import _apply_spec_to_builtin_args

    subagent = _subagent_reporter(root_runtime)
    spec = ResolvedAgentSpec(
        name="coder",
        system_prompt_path=Path("coder.md"),
        system_prompt_args={},
        model=None,
        when_to_use="",
        tools=[],
        allowed_tools=None,
        exclude_tools=[],
        subagents={},
    )

    root_args = _apply_spec_to_builtin_args(root_runtime, spec)
    subagent_args = _apply_spec_to_builtin_args(subagent, spec)

    assert "OPENKIMO_GLOBAL_WIKI_START" in root_args.KIMI_WIKI_CONTEXT
    assert subagent_args.KIMI_WIKI_CONTEXT == ""


# ---------------------------------------------------------------------------
# Run generation
# ---------------------------------------------------------------------------


def test_begin_run_advances_the_run_generation_monotonically(runtime) -> None:
    store = runtime.subagent_store
    store.create_instance(
        agent_id="a7",
        description="worker",
        launch_spec=AgentLaunchSpec(
            agent_id="a7",
            subagent_type="coder",
            model_override=None,
            effective_model=None,
        ),
    )

    assert store.require_instance("a7").run_generation == 0
    assert store.begin_run("a7").run_generation == 1
    assert store.begin_run("a7").run_generation == 2
    # Unrelated updates must not reset it.
    assert store.update_instance("a7", status="idle").run_generation == 2
    assert store.require_instance("a7").run_generation == 2


# ---------------------------------------------------------------------------
# Sealing subagent evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_evidence_is_sealed_without_touching_the_root_turn(root_runtime) -> None:
    coordinator = root_runtime.wiki_coordinator
    await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime)

    await _observe_workspace_file(subagent)

    sealed = subagent.wiki_evidence_reporter.seal_subagent_run()
    assert len(sealed) == 1
    assert sealed[0].source_class == "workspace-file"
    assert sealed[0].logical_paths == ("decision.md",)
    assert sealed[0].reliable and sealed[0].stable_snapshot
    # Nothing entered the root turn on its own.
    assert (await coordinator.pending_batch()).checkpoints == ()


@pytest.mark.asyncio
async def test_subagent_can_seal_evidence_with_no_active_root_turn(root_runtime) -> None:
    """A background run outlives the turn that started it."""
    subagent = _subagent_reporter(root_runtime)

    await _observe_workspace_file(subagent)

    assert root_runtime.wiki_coordinator.active_turn_id is None
    assert len(subagent.wiki_evidence_reporter.seal_subagent_run()) == 1


@pytest.mark.asyncio
async def test_root_reporter_seals_nothing_for_a_subagent_run(root_runtime) -> None:
    subagent = _subagent_reporter(root_runtime)
    assert root_runtime.wiki_evidence_reporter.seal_subagent_run() == ()
    assert subagent.wiki_evidence_reporter.seal_subagent_run() == ()


# ---------------------------------------------------------------------------
# Admitting a subagent result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_subagent_result_creates_one_bound_checkpoint(root_runtime) -> None:
    coordinator = root_runtime.wiki_coordinator
    turn = await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime, agent_id="a3", run_generation=4)
    await _observe_workspace_file(subagent)
    sealed = subagent.wiki_evidence_reporter.seal_subagent_run()
    summary_hash = content_hash(b"The retry budget lives in config.toml")

    checkpoint = await coordinator.accept_subagent_result(
        agent_id="a3",
        run_generation=4,
        summary_hash=summary_hash,
        evidence=sealed,
        receiving_root_turn_id=turn.root_turn_id,
    )

    assert checkpoint is not None
    assert checkpoint.cause == "subagent_result"
    assert checkpoint.producer_id == "a3"
    assert checkpoint.run_generation == 4
    assert checkpoint.root_turn_id == turn.root_turn_id
    assert len(checkpoint.evidence_ids) == 1
    batch = await coordinator.pending_batch()
    assert batch.checkpoints == (checkpoint,)
    assert "OPENKIMO_WIKI_CHECKPOINT_START" in batch.rendered


@pytest.mark.asyncio
async def test_replayed_subagent_result_reuses_its_checkpoint(root_runtime) -> None:
    coordinator = root_runtime.wiki_coordinator
    turn = await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime)
    await _observe_workspace_file(subagent)
    sealed = subagent.wiki_evidence_reporter.seal_subagent_run()
    call = {
        "agent_id": "a1",
        "run_generation": 1,
        "summary_hash": content_hash(b"same conclusion"),
        "evidence": sealed,
        "receiving_root_turn_id": turn.root_turn_id,
    }

    first = await coordinator.accept_subagent_result(**call)
    second = await coordinator.accept_subagent_result(**call)

    assert first is not None
    assert second is not None
    assert first.checkpoint_id == second.checkpoint_id
    assert len((await coordinator.pending_batch()).checkpoints) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"agent_id": ""},
        {"run_generation": -1},
        {"summary_hash": "not-a-hash"},
        {"evidence": ()},
        {"receiving_root_turn_id": "a-stale-turn"},
    ],
)
async def test_unverifiable_subagent_result_is_rejected_without_a_checkpoint(
    root_runtime,
    overrides: dict[str, object],
) -> None:
    coordinator = root_runtime.wiki_coordinator
    turn = await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime)
    await _observe_workspace_file(subagent)
    call: dict[str, object] = {
        "agent_id": "a1",
        "run_generation": 1,
        "summary_hash": content_hash(b"conclusion"),
        "evidence": subagent.wiki_evidence_reporter.seal_subagent_run(),
        "receiving_root_turn_id": turn.root_turn_id,
    }
    call.update(overrides)

    assert await coordinator.accept_subagent_result(**call) is None
    assert (await coordinator.pending_batch()).checkpoints == ()


@pytest.mark.asyncio
async def test_subagent_result_after_the_turn_ended_is_rejected(root_runtime) -> None:
    coordinator = root_runtime.wiki_coordinator
    turn = await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime)
    await _observe_workspace_file(subagent)
    sealed = subagent.wiki_evidence_reporter.seal_subagent_run()
    await coordinator.cancel_turn(turn.root_turn_id)

    checkpoint = await coordinator.accept_subagent_result(
        agent_id="a1",
        run_generation=1,
        summary_hash=content_hash(b"conclusion"),
        evidence=sealed,
        receiving_root_turn_id=turn.root_turn_id,
    )

    assert checkpoint is None


@pytest.mark.asyncio
async def test_forged_evidence_reference_cannot_widen_a_subagent_result(root_runtime) -> None:
    coordinator = root_runtime.wiki_coordinator
    turn = await coordinator.begin_turn("root work", "root work")
    forged = PersistedEvidenceRef(
        evidence_id="forged",
        source_class="workspace-file",
        request_hash=content_hash(b"req"),
        result_hash=content_hash(b"res"),
        logical_paths=("../outside.md",),
        source_refs=(),
        reliable=True,
        stable_snapshot=True,
    )

    checkpoint = await coordinator.accept_subagent_result(
        agent_id="a1",
        run_generation=1,
        summary_hash=content_hash(b"conclusion"),
        evidence=(forged,),
        receiving_root_turn_id=turn.root_turn_id,
    )

    assert checkpoint is None
    assert (await coordinator.pending_batch()).checkpoints == ()


# ---------------------------------------------------------------------------
# Background manifest and once-only delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_manifest_round_trips_hash_only_evidence(root_runtime) -> None:
    subagent = _subagent_reporter(root_runtime, agent_id="a5", run_generation=3)
    await _observe_workspace_file(subagent)
    store = root_runtime.background_tasks.store
    _write_agent_task(root_runtime, "t-round-trip", "a5", status="completed")
    manifest = WikiEvidenceManifest(
        agent_id="a5",
        task_id="t-round-trip",
        run_generation=3,
        summary_hash=content_hash(b"conclusion"),
        evidence=subagent.wiki_evidence_reporter.seal_subagent_run(),
    )

    store.write_wiki_evidence_manifest("t-round-trip", manifest)
    loaded = store.read_wiki_evidence_manifest("t-round-trip")

    assert loaded == manifest
    assert loaded.status == "sealed"
    raw = store.wiki_evidence_path("t-round-trip").read_text(encoding="utf-8")
    assert "durable decision.md" not in raw
    assert "conclusion" not in raw


@pytest.mark.asyncio
async def test_notification_and_repeated_task_output_deliver_the_block_once(
    root_runtime,
    task_output_tool,
) -> None:
    coordinator = root_runtime.wiki_coordinator
    await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime, agent_id="a5", run_generation=3)
    await _observe_workspace_file(subagent)
    _write_agent_task(root_runtime, "t-deliver", "a5", status="completed")
    root_runtime.background_tasks.store.write_wiki_evidence_manifest(
        "t-deliver",
        WikiEvidenceManifest(
            agent_id="a5",
            task_id="t-deliver",
            run_generation=3,
            summary_hash=content_hash(b"conclusion"),
            evidence=subagent.wiki_evidence_reporter.seal_subagent_run(),
        ),
    )
    view = NotificationView(
        event=NotificationEvent(
            id="n1",
            category="task",
            type="task.completed",
            source_kind="background_task",
            source_id="t-deliver",
            title="Task finished",
            body="done",
            severity="info",
            targets=["wire"],
        ),
        delivery=NotificationDelivery(),
    )

    block = await root_runtime.background_tasks.checkpoint_block_for_task("t-deliver")
    notification = build_notification_message(view, root_runtime, checkpoint_block=block)
    first = await task_output_tool(task_output_tool.params(task_id="t-deliver"))
    second = await task_output_tool(task_output_tool.params(task_id="t-deliver"))

    combined = notification.extract_text("\n") + first.output + second.output
    assert combined.count("<OPENKIMO_WIKI_CHECKPOINT_START>") == 1
    assert block in notification.extract_text("\n")
    assert len((await coordinator.pending_batch()).checkpoints) == 1


@pytest.mark.asyncio
async def test_task_output_alone_delivers_the_block_once(root_runtime, task_output_tool) -> None:
    coordinator = root_runtime.wiki_coordinator
    await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime, agent_id="a6", run_generation=2)
    await _observe_workspace_file(subagent)
    _write_agent_task(root_runtime, "t-output", "a6", status="completed")
    root_runtime.background_tasks.store.write_wiki_evidence_manifest(
        "t-output",
        WikiEvidenceManifest(
            agent_id="a6",
            task_id="t-output",
            run_generation=2,
            summary_hash=content_hash(b"conclusion"),
            evidence=subagent.wiki_evidence_reporter.seal_subagent_run(),
        ),
    )

    first = await task_output_tool(task_output_tool.params(task_id="t-output"))
    second = await task_output_tool(task_output_tool.params(task_id="t-output"))

    assert "<OPENKIMO_WIKI_CHECKPOINT_START>" in first.output
    assert "<OPENKIMO_WIKI_CHECKPOINT_START>" not in second.output
    # The ordinary consumer bookkeeping still happens on both calls.
    consumer = root_runtime.background_tasks.store.read_consumer("t-output")
    assert consumer.last_viewed_at is not None
    assert len(consumer.delivered_wiki_checkpoint_keys) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "killed", "lost", "running"])
async def test_non_completed_task_never_delivers_a_checkpoint(
    root_runtime,
    status: TaskStatus,
) -> None:
    coordinator = root_runtime.wiki_coordinator
    await coordinator.begin_turn("root work", "root work")
    subagent = _subagent_reporter(root_runtime, agent_id="a8", run_generation=1)
    await _observe_workspace_file(subagent)
    store = root_runtime.background_tasks.store
    _write_agent_task(root_runtime, "t-bad", "a8", status=status)
    store.write_wiki_evidence_manifest(
        "t-bad",
        WikiEvidenceManifest(
            agent_id="a8",
            task_id="t-bad",
            run_generation=1,
            summary_hash=content_hash(b"conclusion"),
            evidence=subagent.wiki_evidence_reporter.seal_subagent_run(),
        ),
    )

    assert await root_runtime.background_tasks.checkpoint_block_for_task("t-bad") == ""
    assert (await coordinator.pending_batch()).checkpoints == ()
    # A terminal non-completed task must not leave a readable manifest behind.
    if status != "running":
        assert store.read_wiki_evidence_manifest("t-bad") is None


@pytest.mark.asyncio
async def test_restart_style_manifest_binds_to_the_new_receiving_turn(root_runtime) -> None:
    """A manifest survives a restart, but the checkpoint belongs to the turn that reads it."""
    coordinator = root_runtime.wiki_coordinator
    subagent = _subagent_reporter(root_runtime, agent_id="a4", run_generation=1)
    await _observe_workspace_file(subagent)
    _write_agent_task(root_runtime, "t-restart", "a4", status="completed")
    root_runtime.background_tasks.store.write_wiki_evidence_manifest(
        "t-restart",
        WikiEvidenceManifest(
            agent_id="a4",
            task_id="t-restart",
            run_generation=1,
            summary_hash=content_hash(b"conclusion"),
            evidence=subagent.wiki_evidence_reporter.seal_subagent_run(),
        ),
    )

    # No turn is open yet: the sealed manifest cannot admit itself.
    assert await root_runtime.background_tasks.checkpoint_block_for_task("t-restart") == ""

    later = await coordinator.begin_turn("later prompt", "later prompt")
    block = await root_runtime.background_tasks.checkpoint_block_for_task("t-restart")

    assert "<OPENKIMO_WIKI_CHECKPOINT_START>" in block
    checkpoints = (await coordinator.pending_batch()).checkpoints
    assert len(checkpoints) == 1
    assert checkpoints[0].root_turn_id == later.root_turn_id


@pytest.mark.asyncio
async def test_checkpoint_delivery_is_inert_without_a_root_coordinator(runtime) -> None:
    _write_agent_task(runtime, "t-none", "a2", status="completed")

    assert await runtime.background_tasks.checkpoint_block_for_task("t-none") == ""
