"""Tests for typed, runtime-owned Wiki evidence capture."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from kaos.path import KaosPath
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
from kimi_cli.wiki.triggers import EvidenceObservation, WikiEvidence, WikiTurnCoordinator


class _FakeParams(BaseModel):
    path: str


class _FakeNamedRead(CallableTool2[_FakeParams]):
    name = "ReadFile"
    description = "Not the built-in ReadFile implementation."
    params = _FakeParams

    async def __call__(self, params: _FakeParams) -> ToolReturnValue:
        return ToolOk(output=params.path)


async def _record_matching_subagent_evidence(
    coordinator: WikiTurnCoordinator,
    root_evidence: WikiEvidence,
    *,
    producer_id: str,
    run_generation: int,
    tool_call_id: str,
) -> WikiEvidence:
    evidence = await coordinator.record_evidence(
        EvidenceObservation(
            root_turn_id=root_evidence.root_turn_id,
            workspace_id=root_evidence.workspace_id,
            producer_role="subagent",
            producer_id=producer_id,
            run_generation=run_generation,
            tool_call_id=tool_call_id,
            source_class=root_evidence.source_class,
            request_hash=content_hash(b"subagent request"),
            result_hash=root_evidence.result_hash,
            logical_paths=root_evidence.logical_paths,
            source_refs=root_evidence.source_refs,
            reliable=root_evidence.reliable,
            stable_snapshot=root_evidence.stable_snapshot,
            triggering=root_evidence.triggering,
        )
    )
    assert evidence is not None
    return evidence


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
    workspace = Path(str(evidence_runtime.session.work_dir))
    decision = workspace / "docs" / "decision.md"
    decision.parent.mkdir(parents=True)
    decision.write_text("decision", encoding="utf-8")

    discovery = await reporter.observe(
        glob_tool,
        {"pattern": "**/*.md", "directory": str(workspace)},
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
async def test_glob_and_grep_only_record_verified_current_workspace_matches(
    evidence_runtime,
    glob_tool: Glob,
    grep_tool: Grep,
    tmp_path: Path,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    inside = workspace / "docs" / "inside.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("durable inside", encoding="utf-8")
    outside = tmp_path / "outside-discovery"
    outside.mkdir()
    outside_file = outside / "outside.md"
    outside_file.write_text("durable outside", encoding="utf-8")
    additional = tmp_path / "additional-discovery"
    additional.mkdir()
    (additional / "additional.md").write_text("additional", encoding="utf-8")
    skills = tmp_path / "skills-discovery"
    skills.mkdir()
    (skills / "SKILL.md").write_text("skill", encoding="utf-8")
    evidence_runtime.additional_dirs.append(KaosPath.unsafe_from_local_path(additional))
    evidence_runtime.skills_dirs.append(KaosPath.unsafe_from_local_path(skills))
    escape = workspace / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("workspace discovery", "workspace discovery")

    glob_evidence = await reporter.observe(
        glob_tool,
        {"pattern": "*.md", "directory": str(inside.parent)},
        ToolOk(output="inside.md"),
        tool_call_id="inside-glob",
    )
    grep_evidence = await reporter.observe(
        grep_tool,
        {"pattern": "durable", "path": str(workspace), "output_mode": "content"},
        ToolOk(output="docs/inside.md:1:durable inside"),
        tool_call_id="inside-grep",
    )

    assert glob_evidence is not None
    assert glob_evidence.logical_paths == ("docs/inside.md",)
    assert glob_evidence.source_refs[0].path == "docs/inside.md"
    assert grep_evidence is not None
    assert grep_evidence.logical_paths == ("docs/inside.md",)
    assert grep_evidence.source_refs[0].path == "docs/inside.md"

    rejected = [
        await reporter.observe(
            glob_tool,
            {"pattern": "*.md", "directory": str(outside)},
            ToolOk(output="outside.md"),
            tool_call_id="outside-glob",
        ),
        await reporter.observe(
            grep_tool,
            {"pattern": "durable", "path": str(outside), "output_mode": "content"},
            ToolOk(output="outside.md:1:durable outside"),
            tool_call_id="outside-grep",
        ),
        await reporter.observe(
            glob_tool,
            {"pattern": "*.md", "directory": str(additional)},
            ToolOk(output="additional.md"),
            tool_call_id="additional-glob",
        ),
        await reporter.observe(
            glob_tool,
            {"pattern": "*.md", "directory": str(skills)},
            ToolOk(output="SKILL.md"),
            tool_call_id="skills-glob",
        ),
        await reporter.observe(
            glob_tool,
            {"pattern": "*.md", "directory": str(escape)},
            ToolOk(output="outside.md"),
            tool_call_id="symlink-root-glob",
        ),
        await reporter.observe(
            grep_tool,
            {"pattern": "durable", "path": str(workspace), "output_mode": "content"},
            ToolOk(output="escape/outside.md:1:durable outside"),
            tool_call_id="symlink-match-grep",
        ),
        await reporter.observe(
            glob_tool,
            {"pattern": "*.md", "directory": str(workspace)},
            ToolOk(output=str(outside_file)),
            tool_call_id="absolute-match-glob",
        ),
    ]

    assert rejected == [None] * len(rejected)


@pytest.mark.asyncio
async def test_grep_content_paths_keep_separator_shaped_file_names(
    evidence_runtime,
    grep_tool: Grep,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    tricky = workspace / "foo-1-bar.md"
    tricky.write_text("first\ndurable decision\n", encoding="utf-8")
    await evidence_runtime.wiki_coordinator.begin_turn("tricky grep", "tricky grep")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        grep_tool,
        {"pattern": "durable", "path": str(workspace), "output_mode": "content"},
        ToolOk(output="foo-1-bar.md:2:durable decision"),
        tool_call_id="tricky-grep",
    )

    assert evidence is not None
    assert evidence.logical_paths == ("foo-1-bar.md",)
    assert evidence.source_refs[0].path == "foo-1-bar.md"


@pytest.mark.asyncio
async def test_ambiguous_grep_content_path_split_records_no_evidence(
    evidence_runtime,
    grep_tool: Grep,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    (workspace / "foo-1-bar.md").write_text("first\ndurable decision\n", encoding="utf-8")
    (workspace / "foo").write_text("also a real file", encoding="utf-8")
    await evidence_runtime.wiki_coordinator.begin_turn("ambiguous grep", "ambiguous grep")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        grep_tool,
        {"pattern": "durable", "path": str(workspace), "output_mode": "content"},
        ToolOk(output="foo-1-bar.md:2:durable decision"),
        tool_call_id="ambiguous-grep",
    )

    assert evidence is None


@pytest.mark.asyncio
async def test_grep_count_matches_paths_keep_separator_shaped_file_names(
    evidence_runtime,
    grep_tool: Grep,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    (workspace / "report-2-notes.md").write_text("durable decision\n", encoding="utf-8")
    await evidence_runtime.wiki_coordinator.begin_turn("count grep", "count grep")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        grep_tool,
        {"pattern": "durable", "path": str(workspace), "output_mode": "count_matches"},
        ToolOk(output="report-2-notes.md:1"),
        tool_call_id="count-grep",
    )

    assert evidence is not None
    assert evidence.logical_paths == ("report-2-notes.md",)


@pytest.mark.asyncio
async def test_successful_discovery_evidence_is_triggering_and_can_seal_evaluation(
    evidence_runtime,
    glob_tool: Glob,
    grep_tool: Grep,
    search_web_tool: SearchWeb,
) -> None:
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    workspace = Path(str(evidence_runtime.session.work_dir))
    decision = workspace / "decision.md"
    decision.write_text("durable", encoding="utf-8")
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
            {"pattern": "durable", "path": ".", "output_mode": "content"},
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
        "sudo -n env TZ=UTC command date +%s",
        "nice -n 5 timeout 2s command pwd -P",
        "nohup date +%s",
        "nohup -- pwd -P",
        "sudo -n nohup nice -n 5 git -C /tmp status --short",
        "nohup date +%s | nohup uptime",
        "date +%s | command pwd -P",
        "sudo --non-interactive git -C /tmp status --short | nice -n 3 uptime",
        "env -i -- TZ=UTC date +%s | timeout --signal=TERM 2s pwd --physical",
    ],
)
async def test_wrapped_all_status_shell_pipelines_are_non_triggering(
    evidence_runtime,
    shell_tool: Shell,
    command: str,
) -> None:
    await evidence_runtime.wiki_coordinator.begin_turn("wrapped status", "wrapped status")

    evidence = await evidence_runtime.wiki_evidence_reporter.observe(
        shell_tool,
        {"command": command},
        ToolOk(output="current snapshot"),
        tool_call_id="wrapped-status",
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
        "date +%s | tee durable.timestamp",
        "sudo -n timeout 5s ./build.sh | date +%s",
        "date +%s || pwd",
        "date +%s | | pwd",
        "sudo sh -c 'date +%s'",
        "date $(touch marker)",
        "timeout --signal date +%s",
        "nohup ./build.sh",
        "nohup --help",
        "nohup date +%s | ./build.sh",
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
    result = await read_file_tool(ReadParams(path=str(path)))
    root_evidence = await reporter.observe(
        read_file_tool,
        {"path": str(path)},
        result,
        tool_call_id="merge-read",
    )
    assert root_evidence is not None
    subagent_evidence = await _record_matching_subagent_evidence(
        coordinator,
        root_evidence,
        producer_id="worker",
        run_generation=7,
        tool_call_id="merge-subagent-read",
    )
    conclusion = "Equivalent subagent conclusion"
    original = await coordinator.create_checkpoint(
        "subagent_result",
        evidence_ids=(subagent_evidence.evidence_id,),
        summary_hash=content_hash(conclusion.encode()),
        producer_id="worker",
        run_generation=7,
    )

    merged = await reporter.seal_root_completion(conclusion)

    assert merged is not None
    assert merged.checkpoint_id == original.checkpoint_id
    assert original.evidence_ids == (subagent_evidence.evidence_id,)
    assert merged.evidence_ids == (
        subagent_evidence.evidence_id,
        root_evidence.evidence_id,
    )
    replay = await coordinator.create_checkpoint(
        "subagent_result",
        evidence_ids=(subagent_evidence.evidence_id,),
        summary_hash=content_hash(conclusion.encode()),
        producer_id="worker",
        run_generation=7,
    )
    assert replay == merged


@pytest.mark.asyncio
async def test_unique_same_summary_subagent_checkpoint_with_different_source_does_not_merge(
    evidence_runtime,
    read_file_tool,
) -> None:
    workspace = Path(str(evidence_runtime.session.work_dir))
    root_path = workspace / "docs" / "root-source.md"
    subagent_path = workspace / "docs" / "subagent-source.md"
    root_path.parent.mkdir(parents=True)
    root_path.write_text("root source", encoding="utf-8")
    subagent_path.write_text("different subagent source", encoding="utf-8")
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    await coordinator.begin_turn("different source", "different source")

    root_result = await read_file_tool(ReadParams(path=str(root_path)))
    root_evidence = await reporter.observe(
        read_file_tool,
        {"path": str(root_path)},
        root_result,
        tool_call_id="different-root-read",
    )
    assert root_evidence is not None
    subagent_source = evidence_runtime.wiki.registry.relative_source(
        evidence_runtime.workspace_id,
        subagent_path,
    )
    subagent_evidence = await coordinator.record_evidence(
        EvidenceObservation(
            root_turn_id=root_evidence.root_turn_id,
            workspace_id=evidence_runtime.workspace_id,
            producer_role="subagent",
            producer_id="worker",
            run_generation=3,
            tool_call_id="different-subagent-read",
            source_class="workspace-file",
            request_hash=content_hash(b"different subagent request"),
            result_hash=subagent_source.content_hash,
            logical_paths=(subagent_source.path,),
            source_refs=(subagent_source,),
            reliable=True,
            stable_snapshot=True,
            triggering=True,
        )
    )
    assert subagent_evidence is not None
    conclusion = "Same summary but independently sourced"
    subagent_checkpoint = await coordinator.create_checkpoint(
        "subagent_result",
        evidence_ids=(subagent_evidence.evidence_id,),
        summary_hash=content_hash(conclusion.encode()),
        producer_id="worker",
        run_generation=3,
    )

    sealed = await reporter.seal_root_completion(conclusion)

    assert sealed is not None
    assert sealed.cause == "root_evidence"
    assert sealed.checkpoint_id != subagent_checkpoint.checkpoint_id
    assert subagent_checkpoint.evidence_ids == (subagent_evidence.evidence_id,)


@pytest.mark.asyncio
async def test_same_result_hash_across_source_classes_does_not_merge(
    evidence_runtime,
    search_web_tool: SearchWeb,
) -> None:
    coordinator = evidence_runtime.wiki_coordinator
    reporter = evidence_runtime.wiki_evidence_reporter
    turn = await coordinator.begin_turn("cross class", "cross class")
    result = ToolOk(output="identical discovery output")
    root_evidence = await reporter.observe(
        search_web_tool,
        {"query": "identity fields"},
        result,
        tool_call_id="root-web-search",
    )
    assert root_evidence is not None
    subagent_evidence = await coordinator.record_evidence(
        EvidenceObservation(
            root_turn_id=turn.root_turn_id,
            workspace_id=evidence_runtime.workspace_id,
            producer_role="subagent",
            producer_id="worker",
            run_generation=5,
            tool_call_id="subagent-workspace-search",
            source_class="workspace-search",
            request_hash=content_hash(b"workspace search"),
            result_hash=root_evidence.result_hash,
            logical_paths=(),
            source_refs=(),
            reliable=False,
            stable_snapshot=False,
            triggering=True,
        )
    )
    assert subagent_evidence is not None
    conclusion = "Same bytes do not mean the same source class"
    subagent_checkpoint = await coordinator.create_checkpoint(
        "subagent_result",
        evidence_ids=(subagent_evidence.evidence_id,),
        summary_hash=content_hash(conclusion.encode()),
        producer_id="worker",
        run_generation=5,
    )

    sealed = await reporter.seal_root_completion(conclusion)

    assert sealed is not None
    assert sealed.cause == "root_evidence"
    assert sealed.checkpoint_id != subagent_checkpoint.checkpoint_id


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
    subagent_evidence = await _record_matching_subagent_evidence(
        coordinator,
        left,
        producer_id="worker",
        run_generation=4,
        tool_call_id="concurrent-subagent-read",
    )
    summary_hash = content_hash(b"Concurrent subagent conclusion")
    original = await coordinator.create_checkpoint(
        "subagent_result",
        evidence_ids=(subagent_evidence.evidence_id,),
        summary_hash=summary_hash,
        producer_id="worker",
        run_generation=4,
    )

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
    final = next(
        checkpoint
        for checkpoint in (await coordinator.pending_batch()).checkpoints
        if checkpoint.checkpoint_id == original.checkpoint_id
    )

    assert final.checkpoint_id == original.checkpoint_id
    assert final.evidence_ids == (
        subagent_evidence.evidence_id,
        left.evidence_id,
        right.evidence_id,
    )
