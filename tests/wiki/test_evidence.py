"""Tests for typed, runtime-owned Wiki evidence capture."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from kosong.tooling import BriefDisplayBlock, CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel

from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import Params as ReadParams
from kimi_cli.tools.shell import Shell
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import WikiTurnCoordinator


class _FakeParams(BaseModel):
    path: str


class _FakeNamedRead(CallableTool2[_FakeParams]):
    name = "ReadFile"
    description = "Not the built-in ReadFile implementation."
    params = _FakeParams

    async def __call__(self, params: _FakeParams) -> ToolReturnValue:
        return ToolOk(output=params.path)


@pytest.fixture
def evidence_runtime(runtime, tmp_path: Path):
    from kimi_cli.wiki.evidence import WikiEvidenceReporter

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
    try:
        yield runtime
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_read_file_records_exact_workspace_source(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "docs" / "decision.md"
    path.parent.mkdir(parents=True)
    data = "稳定结论\n".encode()
    path.write_bytes(data)
    turn = await evidence_runtime.wiki_coordinator.begin_turn("read it", "read it")
    result = await read_file_tool(ReadParams(path=str(path)))

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        read_file_tool,
        {"path": str(path)},
        result,
        tool_call_id="read-1",
    )

    assert evidence is not None
    assert evidence.root_turn_id == turn.root_turn_id
    assert evidence.source_class == "workspace-file"
    assert evidence.logical_paths == ("docs/decision.md",)
    assert evidence.source_refs[0].content_hash == content_hash(data)
    assert evidence.result_hash == content_hash(data)
    assert evidence.reliable and evidence.stable_snapshot and evidence.triggering
    assert str(workspace) not in repr(evidence)


@pytest.mark.asyncio
async def test_model_name_or_success_flag_cannot_forge_evidence(evidence_runtime) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("fake", "fake")
    fake_tool = _FakeNamedRead()

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        fake_tool,
        {"path": "docs/decision.md", "success": True},
        ToolOk(output="pretend durable content"),
        tool_call_id="fake",
    )

    assert evidence is None


@pytest.mark.asyncio
async def test_late_tool_result_cannot_cross_root_turn_generation(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "late.md"
    path.write_text("old turn result", encoding="utf-8")
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    first = await coordinator.begin_turn("first", "first")
    reporter.start_root_turn(first.root_turn_id)
    reporter.begin_tool_call("late-read")
    reporter.finish_root_turn(first.root_turn_id)
    second = await coordinator.begin_turn("second", "second")
    reporter.start_root_turn(second.root_turn_id)

    evidence = await reporter.observe(
        read_file_tool,
        {"path": str(path)},
        ToolOk(output="old turn output"),
        tool_call_id="late-read",
    )

    assert evidence is None
    assert (await coordinator.pending_batch()).checkpoints == ()


@pytest.mark.asyncio
async def test_failed_empty_and_transient_shell_results_create_no_triggering_evidence(
    evidence_runtime,
    shell_tool: Shell,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("inspect", "inspect")
    reporter = evidence_runtime.wiki_evidence_reporter

    assert (
        await reporter.observe(
            shell_tool,
            {"command": "printf hidden"},
            ToolError(message="failed", brief="failed"),
            tool_call_id="shell-error",
        )
        is None
    )
    assert (
        await reporter.observe(
            shell_tool,
            {"command": "printf hidden"},
            ToolOk(output=""),
            tool_call_id="shell-empty",
        )
        is None
    )
    transient = await reporter.observe(
        shell_tool,
        {"command": "date"},
        ToolOk(output="Mon Aug 3 12:00:00 CST 2026\n"),
        tool_call_id="shell-date",
    )
    assert transient is not None
    assert transient.source_class == "shell-result"
    assert not transient.reliable
    assert not transient.stable_snapshot
    assert not transient.triggering
    assert transient.logical_paths == ()


@pytest.mark.asyncio
async def test_shell_hashes_normalize_equivalent_invocations_and_line_endings(
    evidence_runtime,
    shell_tool: Shell,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("normalize", "normalize")
    reporter = evidence_runtime.wiki_evidence_reporter

    first = await reporter.observe(
        shell_tool,
        {"command": "printf   durable"},
        ToolOk(output="durable\r\nresult\r\n"),
        tool_call_id="shell-normalize-1",
    )
    second = await reporter.observe(
        shell_tool,
        {"command": "printf durable"},
        ToolOk(output="durable\nresult\n"),
        tool_call_id="shell-normalize-2",
    )

    assert first is not None and second is not None
    assert first.request_hash == second.request_hash
    assert first.result_hash == second.result_hash


@pytest.mark.asyncio
async def test_background_shell_start_is_discovery_not_triggering_evidence(
    evidence_runtime,
    shell_tool: Shell,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("start build", "start build")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        shell_tool,
        {"command": "make build", "run_in_background": True},
        ToolOk(output="task_id: bg-1\nstatus: running\nautomatic_notification: true"),
        tool_call_id="shell-background",
    )

    assert evidence is not None
    assert not evidence.triggering


@pytest.mark.asyncio
async def test_successful_empty_mutation_rehashes_file_after_tool_completion(
    evidence_runtime,
    write_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "docs" / "written.md"
    path.parent.mkdir(parents=True)
    data = b"durable result\n"
    path.write_bytes(data)
    await evidence_runtime.wiki_coordinator.begin_turn("write it", "write it")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        write_file_tool,
        {"path": str(path), "content": "model supplied content is not trusted"},
        ToolReturnValue(is_error=False, output="", message="written", display=[]),
        tool_call_id="write-1",
    )

    assert evidence is not None
    assert evidence.source_class == "workspace-mutation"
    assert evidence.source_refs[0].content_hash == content_hash(data)
    assert evidence.result_hash == content_hash(data)


@pytest.mark.asyncio
async def test_workspace_source_rejects_symlink_sensitive_and_outside_targets(
    evidence_runtime,
    read_file_tool,
    tmp_path: Path,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    ordinary = workspace / "ordinary.md"
    ordinary.write_text("ordinary", encoding="utf-8")
    symlink = workspace / "alias.md"
    symlink.symlink_to(ordinary)
    sensitive = workspace / ".env"
    sensitive.write_text("API_KEY=secret", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    await evidence_runtime.wiki_coordinator.begin_turn("guard", "guard")
    reporter = evidence_runtime.wiki_evidence_reporter

    for index, path in enumerate((symlink, sensitive, outside)):
        assert (
            await reporter.observe(
                read_file_tool,
                {"path": str(path)},
                ToolOk(output="claimed success"),
                tool_call_id=f"guard-{index}",
            )
            is None
        )


@pytest.mark.asyncio
async def test_discovery_and_web_document_evidence_use_bounded_safe_metadata(
    evidence_runtime,
    glob_tool: Glob,
    fetch_url_tool: FetchURL,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("discover", "discover")
    reporter = evidence_runtime.wiki_evidence_reporter

    discovery = await reporter.observe(
        glob_tool,
        {"pattern": "**/*.md", "directory": "/private/workspace"},
        ToolOk(output="docs/decision.md\n"),
        tool_call_id="glob-1",
    )
    document = await reporter.observe(
        fetch_url_tool,
        {"url": "HTTPS://Example.COM/reference"},
        ToolOk(output="reference body" * 20_000),
        tool_call_id="fetch-1",
    )
    rejected = await reporter.observe(
        fetch_url_tool,
        {"url": "https://example.com/?api_key=secret"},
        ToolOk(output="secret response"),
        tool_call_id="fetch-secret",
    )

    assert discovery is not None
    assert discovery.source_class == "workspace-search"
    assert not discovery.reliable and not discovery.stable_snapshot
    assert "/private/workspace" not in repr(discovery)
    assert document is not None
    assert document.source_class == "web-document"
    assert str(document.source_refs[0].url) == "https://example.com/reference"
    assert len(document.result_hash) == 71
    assert rejected is None


@pytest.mark.asyncio
async def test_successful_discovery_evidence_is_triggering_and_can_seal_evaluation(
    evidence_runtime,
    glob_tool: Glob,
    grep_tool: Grep,
    search_web_tool: SearchWeb,
) -> None:
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("evaluate sources", "evaluate sources")

    evidence = [
        await reporter.observe(
            glob_tool,
            {"pattern": "*.md"},
            ToolOk(output="decision.md"),
            tool_call_id="discover-glob",
        ),
        await reporter.observe(
            grep_tool,
            {"pattern": "durable", "path": "."},
            ToolOk(output="decision.md:1:durable"),
            tool_call_id="discover-grep",
        ),
        await reporter.observe(
            search_web_tool,
            {"query": "durable design"},
            ToolOk(output="Title: Durable design"),
            tool_call_id="discover-web",
        ),
    ]

    assert all(item is not None for item in evidence)
    assert all(item.triggering for item in evidence if item is not None)
    assert all(not item.reliable for item in evidence if item is not None)
    assert all(not item.stable_snapshot for item in evidence if item is not None)
    checkpoint = await reporter.seal_root_completion("Evaluate these sources before reuse.")
    assert checkpoint is not None
    assert checkpoint.evidence_ids == tuple(item.evidence_id for item in evidence if item)

    assert (
        await reporter.observe(
            glob_tool,
            {"pattern": "empty"},
            ToolOk(output=""),
            tool_call_id="discover-empty",
        )
        is None
    )
    assert (
        await reporter.observe(
            grep_tool,
            {"pattern": "failed", "path": "."},
            ToolError(message="failed", brief="failed"),
            tool_call_id="discover-failed",
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "date +%s"},
        {"command": "command pwd -P"},
        {"command": "command -- pwd --physical"},
        {"command": "git -C /tmp/project status --short"},
        {"command": "env TZ=UTC date --utc +%FT%TZ"},
        {"command": "env -- TZ=UTC date +%s"},
        {"command": "TZ=UTC /usr/bin/date +%s"},
    ],
)
async def test_tokenized_shell_status_command_families_are_non_triggering(
    evidence_runtime,
    shell_tool: Shell,
    arguments: dict[str, str],
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("status", "status")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        shell_tool,
        arguments,
        ToolOk(output="current snapshot"),
        tool_call_id="shell-status",
    )

    assert evidence is not None
    assert not evidence.triggering


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "git -C /tmp/project log -1 --format=%H",
        "python -c 'print(\"date +%s\")'",
        "date +%s > durable.timestamp",
        "git status && git diff --cached",
    ],
)
async def test_tokenized_shell_classifier_preserves_durable_commands(
    evidence_runtime,
    shell_tool: Shell,
    command: str,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("durable", "durable")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        shell_tool,
        {"command": command},
        ToolOk(output="durable result"),
        tool_call_id="shell-durable",
    )

    assert evidence is not None
    assert evidence.triggering


@pytest.mark.asyncio
@pytest.mark.parametrize("brief", ["clock", "current-status", "progress"])
async def test_exact_transient_shell_brief_categories_are_non_triggering(
    evidence_runtime,
    shell_tool: Shell,
    brief: str,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("inspect", "inspect")
    result = ToolReturnValue(
        is_error=False,
        output="snapshot",
        message="",
        display=[BriefDisplayBlock(text=brief)],
    )

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        shell_tool,
        {"command": "custom-inspector"},
        result,
        tool_call_id="shell-brief",
    )

    assert evidence is not None
    assert not evidence.triggering


@pytest.mark.asyncio
async def test_durable_words_containing_transient_terms_are_not_misclassified(
    evidence_runtime,
    shell_tool: Shell,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("policy", "policy")
    result = ToolReturnValue(
        is_error=False,
        output="durable progress retention policy",
        message="",
        display=[BriefDisplayBlock(text="durable current-status policy")],
    )

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        shell_tool,
        {"command": "generate-policy"},
        result,
        tool_call_id="shell-policy",
    )

    assert evidence is not None
    assert evidence.triggering


@pytest.mark.asyncio
async def test_root_completion_requires_triggering_non_transient_evidence_and_deduplicates(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "docs" / "config.md"
    path.parent.mkdir(parents=True)
    path.write_text("portable setting\n", encoding="utf-8")
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("inspect config", "inspect config")
    result = await read_file_tool(ReadParams(path=str(path)))
    await reporter.observe(read_file_tool, {"path": str(path)}, result, tool_call_id="read-config")

    first = await reporter.seal_root_completion("Reuse the portable setting for future runs.")
    second = await reporter.seal_root_completion("Reuse the portable setting for future runs.")

    assert first is not None
    assert second == first
    assert first.cause == "root_evidence"
    assert first.summary_hash == content_hash(b"Reuse the portable setting for future runs.")
    assert (await coordinator.pending_batch()).checkpoints == (first,)


@pytest.mark.asyncio
async def test_short_chinese_reusable_conclusion_creates_checkpoint(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "docs" / "zh.md"
    path.parent.mkdir(parents=True)
    path.write_text("稳定结论\n", encoding="utf-8")
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("读取稳定结论", "读取稳定结论")
    result = await read_file_tool(ReadParams(path=str(path)))
    await reporter.observe(read_file_tool, {"path": str(path)}, result, tool_call_id="read-zh")

    checkpoint = await reporter.seal_root_completion("这是稳定结论")

    assert checkpoint is not None
    assert checkpoint.summary_hash == content_hash("这是稳定结论".encode())


@pytest.mark.asyncio
async def test_equivalent_subagent_checkpoint_atomically_attaches_root_evidence_beyond_batch(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "docs" / "merge.md"
    path.parent.mkdir(parents=True)
    path.write_text("merge evidence", encoding="utf-8")
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("merge", "merge")
    for index in range(5):
        await coordinator.create_checkpoint(
            "root_evidence",
            summary_hash=content_hash(f"unrelated-{index}".encode()),
        )
    conclusion = "Equivalent subagent conclusion"
    original = await coordinator.create_checkpoint(
        "subagent_result",
        summary_hash=content_hash(conclusion.encode()),
        producer_id="worker",
        run_generation=7,
    )
    result = await read_file_tool(ReadParams(path=str(path)))
    root_evidence = await reporter.observe(
        read_file_tool,
        {"path": str(path)},
        result,
        tool_call_id="merge-read",
    )
    assert root_evidence is not None

    merged = await reporter.seal_root_completion(conclusion)

    assert merged is not None
    assert merged.checkpoint_id == original.checkpoint_id
    assert original.evidence_ids == ()
    assert merged.evidence_ids == (root_evidence.evidence_id,)
    replay = await coordinator.create_checkpoint(
        "subagent_result",
        summary_hash=content_hash(conclusion.encode()),
        producer_id="worker",
        run_generation=7,
    )
    assert replay == merged


@pytest.mark.asyncio
async def test_ambiguous_subagent_generations_do_not_merge_wrong_checkpoint(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "docs" / "ambiguous.md"
    path.parent.mkdir(parents=True)
    path.write_text("root evidence", encoding="utf-8")
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("ambiguous", "ambiguous")
    conclusion = "Same summary across agent generations"
    summary_hash = content_hash(conclusion.encode())
    older = await coordinator.create_checkpoint(
        "subagent_result",
        summary_hash=summary_hash,
        producer_id="worker",
        run_generation=1,
    )
    newer = await coordinator.create_checkpoint(
        "subagent_result",
        summary_hash=summary_hash,
        producer_id="worker",
        run_generation=2,
    )
    result = await read_file_tool(ReadParams(path=str(path)))
    await reporter.observe(
        read_file_tool,
        {"path": str(path)},
        result,
        tool_call_id="ambiguous-read",
    )

    sealed = await reporter.seal_root_completion(conclusion)

    assert sealed is not None
    assert sealed.cause == "root_evidence"
    assert sealed.checkpoint_id not in {older.checkpoint_id, newer.checkpoint_id}


@pytest.mark.asyncio
async def test_concurrent_root_evidence_attachments_preserve_both_updates(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    path = workspace / "docs" / "concurrent.md"
    path.parent.mkdir(parents=True)
    path.write_text("concurrent evidence", encoding="utf-8")
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("concurrent", "concurrent")
    summary_hash = content_hash(b"Concurrent subagent conclusion")
    original = await coordinator.create_checkpoint(
        "subagent_result",
        summary_hash=summary_hash,
        producer_id="worker",
        run_generation=4,
    )
    result = await read_file_tool(ReadParams(path=str(path)))
    left = await reporter.observe(
        read_file_tool,
        {"path": str(path)},
        result,
        tool_call_id="concurrent-left",
    )
    right = await reporter.observe(
        read_file_tool,
        {"path": str(path)},
        result,
        tool_call_id="concurrent-right",
    )
    assert left is not None and right is not None

    await asyncio.gather(
        coordinator.attach_root_evidence_to_equivalent_subagent(
            summary_hash=summary_hash,
            evidence_ids=(left.evidence_id,),
        ),
        coordinator.attach_root_evidence_to_equivalent_subagent(
            summary_hash=summary_hash,
            evidence_ids=(right.evidence_id,),
        ),
    )
    final = await coordinator.attach_root_evidence_to_equivalent_subagent(
        summary_hash=summary_hash,
        evidence_ids=(),
    )

    assert final is not None
    assert final.checkpoint_id == original.checkpoint_id
    assert final.evidence_ids == (left.evidence_id, right.evidence_id)
