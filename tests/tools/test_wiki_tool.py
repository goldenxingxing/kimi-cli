"""Contract tests for the controlled global Wiki tool."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.wiki import Params, Wiki
from kimi_cli.wiki.models import CurrentSource, PageChange, SourceRef, WikiCandidate, WikiPage
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.value_gate import WikiContext

_SESSION_ID = UUID("723e4567-e89b-12d3-a456-426614174000")
_NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)


@pytest.fixture
def manager(tmp_path: Path):
    from kimi_cli.wiki.manager import WikiManager

    instance = WikiManager(tmp_path / "wiki", wal=False)
    yield instance
    instance.close()


def _source() -> SourceRef:
    return SourceRef(
        kind="conversation",
        session_id=_SESSION_ID,
        content_hash="sha256:" + "a" * 64,
    )


def _candidate(source: SourceRef | None = None) -> WikiCandidate:
    source = source or _source()
    page = WikiPage(
        logical_path="concepts/controlled-tools.md",
        title="Controlled tools",
        created=_NOW,
        updated=_NOW,
        tags=["wiki"],
        sources=[source],
        revision=1,
        body="Use the dedicated Wiki tool for durable shared knowledge.\n",
    )
    return WikiCandidate(
        summary="Record controlled Wiki tooling guidance",
        pages=[PageChange(page=page, expected_revision=None)],
        sources=[source],
        value="high",
    )


@pytest.fixture
def wiki_tool(manager):
    runtime = SimpleNamespace(
        wiki=manager,
        session=SimpleNamespace(id=str(_SESSION_ID)),
    )
    return Wiki(cast("Runtime", runtime))


async def test_search_read_and_lint_are_read_only(wiki_tool, manager) -> None:
    prepared = manager.prepare(
        _candidate(),
        WikiContext(
            session_id=_SESSION_ID,
            cross_turn_utility=True,
            stable=True,
            user_confirmed=True,
        ),
    )
    manager.commit(prepared)
    before = manager.layout.revision.read_text(encoding="ascii")

    searched = await wiki_tool(Params(operation="search", query="controlled", limit=3))
    read = await wiki_tool(Params(operation="read", page="concepts/controlled-tools.md"))
    linted = await wiki_tool(Params(operation="lint"))

    assert not searched.is_error
    assert json.loads(searched.output)["results"][0]["path"] == "concepts/controlled-tools.md"
    assert not read.is_error
    assert json.loads(read.output)["page"] == "concepts/controlled-tools.md"
    assert not linted.is_error
    assert manager.layout.revision.read_text(encoding="ascii") == before


async def test_remember_only_prepares_a_high_value_candidate(wiki_tool, manager) -> None:
    result = await wiki_tool(Params(operation="remember", candidate=_candidate()))

    assert not result.is_error
    assert json.loads(result.output) == {
        "status": "prepared",
        "summary": "Record controlled Wiki tooling guidance",
        "pages": ["concepts/controlled-tools.md"],
    }
    assert not (manager.layout.root / "concepts" / "controlled-tools.md").exists()


async def test_ingest_accepts_only_current_inline_content_and_prepares_change(
    wiki_tool, manager
) -> None:
    raw = "Current-turn evidence supports controlled global Wiki writes."
    source = CurrentSource(kind="inline", content=raw)
    provenance = SourceRef(
        kind="conversation",
        session_id=_SESSION_ID,
        content_hash=content_hash(raw.encode("utf-8")),
    )

    result = await wiki_tool(
        Params(operation="ingest", source=source, candidate=_candidate(provenance))
    )

    assert not result.is_error
    assert json.loads(result.output)["status"] == "prepared"
    assert not (manager.layout.root / "concepts" / "controlled-tools.md").exists()


@pytest.mark.parametrize("source", ["/etc/passwd", "wiki.zip", "."])
def test_ingest_rejects_directory_archive_and_arbitrary_path_at_model_boundary(source: str) -> None:
    with pytest.raises(ValueError, match="CurrentSource"):
        Params.model_validate({"operation": "ingest", "source": source})


async def test_ingest_rejects_sensitive_current_turn_content(wiki_tool) -> None:
    result = await wiki_tool(
        Params(
            operation="ingest",
            source=CurrentSource(kind="inline", content="api_key=not-safe"),
            candidate=_candidate(),
        )
    )

    assert result.is_error
    assert "discarded" in result.message.lower()


async def test_ingest_rejects_registered_workspace_archive(
    wiki_tool, manager, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive = workspace / "evidence.zip"
    archive.write_text("not a permitted archive source", encoding="utf-8")
    workspace_id = manager.registry.register(workspace)
    source = CurrentSource(
        kind="workspace-file",
        workspace_id=workspace_id,
        relative_path="evidence.zip",
    )
    provenance = SourceRef(
        kind="workspace-file",
        workspace_id=workspace_id,
        path="evidence.zip",
        content_hash=content_hash(archive.read_bytes()),
    )

    result = await wiki_tool(
        Params(operation="ingest", source=source, candidate=_candidate(provenance))
    )

    assert result.is_error
    assert "archive" in result.message.lower()


async def test_tool_fails_closed_when_global_wiki_is_unavailable() -> None:
    tool = Wiki(
        cast("Runtime", SimpleNamespace(wiki=None, session=SimpleNamespace(id=str(_SESSION_ID))))
    )

    result = await tool(Params(operation="search", query="anything"))

    assert result.is_error
    assert "unavailable" in result.message.lower()


async def test_operation_error_never_echoes_machine_path() -> None:
    class FailingManager:
        def search(self, _query: str, _limit: int):
            raise OSError("cannot read /Users/private/wiki")

    tool = Wiki(
        cast(
            "Runtime",
            SimpleNamespace(wiki=FailingManager(), session=SimpleNamespace(id=str(_SESSION_ID))),
        )
    )

    result = await tool(Params(operation="search", query="anything"))

    assert result.is_error
    assert "/Users/private/wiki" not in result.message
