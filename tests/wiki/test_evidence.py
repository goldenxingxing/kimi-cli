"""Tests for typed, runtime-owned Wiki evidence capture."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel

from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.read import Params as ReadParams
from kimi_cli.tools.shell import Shell
from kimi_cli.tools.web.fetch import FetchURL
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
