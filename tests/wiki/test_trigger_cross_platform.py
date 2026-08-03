"""Cross-platform and budget guarantees for Wiki trigger records."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from uuid import uuid4

import pytest
from kosong.tooling import ToolOk

from kimi_cli.tools.file.read import ReadFile
from kimi_cli.wiki.evidence import WikiEvidenceReporter
from kimi_cli.wiki.intent import MAX_INTENT_BYTES, detect_durable_intent
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wiki.models import SourceRef, validate_relative_source_path
from kimi_cli.wiki.retrieval import (
    RETRIEVAL_MAX_QUERY_BYTES,
    build_retrieval_query,
    truncate_utf8,
)
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import (
    OPENKIMO_WIKI_CHECKPOINT_START,
    EvidenceObservation,
    WikiTurnCoordinator,
)


@pytest.fixture
def platform_runtime(runtime, tmp_path: Path):
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


# ---------------------------------------------------------------------------
# Portable paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evidence_stores_only_registry_relative_posix_paths(platform_runtime) -> None:
    workspace = Path(str(platform_runtime.session.work_dir))
    nested = workspace / "docs" / "adr" / "0001-release.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("durable decision", encoding="utf-8")
    await platform_runtime.wiki_coordinator.begin_turn("read it", "read it")

    evidence = await platform_runtime.wiki_evidence_reporter.observe(
        ReadFile(platform_runtime),
        {"path": "docs/adr/0001-release.md"},
        ToolOk(output="durable decision"),
        tool_call_id="read-nested",
    )

    assert evidence is not None
    assert evidence.logical_paths == ("docs/adr/0001-release.md",)
    assert evidence.source_refs[0].path == "docs/adr/0001-release.md"
    # No separator that only one platform understands, and no absolute prefix.
    assert "\\" not in evidence.source_refs[0].path
    assert not evidence.source_refs[0].path.startswith("/")
    assert str(workspace) not in repr(evidence)


@pytest.mark.parametrize(
    "windows_path",
    [
        r"C:\Users\qunwei\workspace\docs\decision.md",
        r"\\server\share\docs\decision.md",
        r"docs\decision.md",
        r"..\outside\decision.md",
        r"C:docs\decision.md",
    ],
)
def test_windows_shaped_paths_never_become_logical_source_paths(windows_path: str) -> None:
    """Drive letters, UNC roots, and backslash separators are all rejected."""
    with pytest.raises(ValueError):
        validate_relative_source_path(windows_path)


def test_a_pure_windows_path_is_not_silently_accepted_as_relative() -> None:
    pure = PureWindowsPath(r"C:\workspace\docs\decision.md")

    assert pure.is_absolute()
    with pytest.raises(ValueError):
        validate_relative_source_path(str(pure))


@pytest.mark.asyncio
async def test_no_absolute_path_reaches_a_rendered_checkpoint_block(platform_runtime) -> None:
    workspace = Path(str(platform_runtime.session.work_dir))
    path = workspace / "decision.md"
    path.write_text("durable decision", encoding="utf-8")
    coordinator = platform_runtime.wiki_coordinator
    await coordinator.begin_turn("read it", "read it")
    source = platform_runtime.wiki.registry.relative_source(platform_runtime.workspace_id, path)
    evidence = await coordinator.record_evidence(
        EvidenceObservation(
            root_turn_id=coordinator.active_turn_id,
            workspace_id=platform_runtime.workspace_id,
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
    await coordinator.create_checkpoint(
        "root_evidence",
        evidence_ids=(evidence.evidence_id,),
        summary_hash=content_hash(b"conclusion"),
    )

    rendered = (await coordinator.pending_batch()).rendered

    assert OPENKIMO_WIKI_CHECKPOINT_START in rendered
    assert str(workspace) not in rendered
    assert "decision.md" in rendered  # the portable relative path is fine
    assert "durable decision" not in rendered


@pytest.mark.asyncio
async def test_a_remote_workspace_records_no_workspace_file_evidence(
    platform_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote session may read the shared Wiki but cannot claim local files."""
    from kimi_cli.metadata import WorkDirMeta

    workspace = Path(str(platform_runtime.session.work_dir))
    (workspace / "decision.md").write_text("durable decision", encoding="utf-8")
    await platform_runtime.wiki_coordinator.begin_turn("read it", "read it")
    monkeypatch.setattr(
        platform_runtime.session,
        "work_dir_meta",
        WorkDirMeta(path=str(workspace), kaos="ssh"),
    )

    evidence = await platform_runtime.wiki_evidence_reporter.observe(
        ReadFile(platform_runtime),
        {"path": "decision.md"},
        ToolOk(output="durable decision"),
        tool_call_id="remote-read",
    )

    assert evidence is None


# ---------------------------------------------------------------------------
# Byte budgets at multi-byte boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["测试" * 400, "aπ" * 400, "🌏" * 200])
def test_truncation_always_lands_on_a_code_point_boundary(text: str) -> None:
    for budget in (1, 2, 3, 7, 64, 511, 512):
        truncated = truncate_utf8(text, max_bytes=budget)

        assert len(truncated.encode("utf-8")) <= budget
        # Round-tripping proves no code point was cut in half.
        assert truncated.encode("utf-8").decode("utf-8") == truncated


def test_a_chinese_query_respects_the_512_byte_budget() -> None:
    query = build_retrieval_query("发布规则" * 300)

    assert len(query.encode("utf-8")) <= RETRIEVAL_MAX_QUERY_BYTES
    assert query.encode("utf-8").decode("utf-8") == query


def test_intent_detection_bound_holds_at_a_multibyte_boundary() -> None:
    # 3 bytes per character, so the budget cuts mid-character.
    filler = "测" * ((MAX_INTENT_BYTES // 3) + 1)

    assert detect_durable_intent(f"{filler}请记住这个规则") is None
    assert detect_durable_intent("请记住这个规则") is not None


@pytest.mark.asyncio
async def test_a_chinese_conclusion_still_fits_the_checkpoint_block(platform_runtime) -> None:
    coordinator = platform_runtime.wiki_coordinator
    await coordinator.begin_turn("多字节", "多字节")
    for index in range(4):
        await coordinator.create_checkpoint(
            "root_evidence",
            summary_hash=content_hash(f"发布规则结论 {index}".encode()),
        )

    batch = await coordinator.pending_batch()

    assert len(batch.checkpoints) == 4
    assert len(batch.rendered.encode("utf-8")) <= 6 * 1024
    assert batch.rendered.encode("utf-8").decode("utf-8") == batch.rendered


def test_a_conversation_source_carries_no_text_only_a_hash() -> None:
    source = SourceRef(
        kind="conversation",
        session_id=uuid4(),
        content_hash=content_hash("请记住这个发布规则".encode()),
    )

    rendered = source.model_dump_json(exclude_none=True)

    assert "请记住" not in rendered
    assert source.content_hash in rendered
