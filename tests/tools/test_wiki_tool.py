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
from kimi_cli.soul.approval import Approval
from kimi_cli.tools.wiki import Params, Wiki, WikiToolContext
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


def _tool_context(
    *,
    conversation_hashes: frozenset[str] | None = None,
    allowed_workspace_ids: frozenset[UUID] = frozenset(),
    candidate_high_value: bool = True,
    stable: bool = True,
    user_confirmed: bool = True,
) -> WikiToolContext:
    return WikiToolContext(
        provenance_session_id=_SESSION_ID,
        conversation_hashes=conversation_hashes or frozenset({_source().content_hash}),
        allowed_workspace_ids=allowed_workspace_ids,
        candidate_high_value=candidate_high_value,
        stable=stable,
        user_confirmed=user_confirmed,
        reliable_source=False,
    )


@pytest.fixture
def wiki_tool(manager):
    runtime = SimpleNamespace(
        wiki=manager,
        approval=Approval(yolo=True),
        session=SimpleNamespace(id="named-shell-session"),
        workspace_id=None,
        wiki_tool_context=_tool_context(),
    )
    return Wiki(cast("Runtime", runtime))


def test_tool_description_exposes_only_portable_provenance(wiki_tool) -> None:
    assert str(_SESSION_ID) in wiki_tool.description
    assert "SHA-256" in wiki_tool.description
    assert "/Users/" not in wiki_tool.description


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


async def test_remember_yolo_commits_a_high_value_candidate(wiki_tool, manager) -> None:
    result = await wiki_tool(Params(operation="remember", candidate=_candidate()))

    assert not result.is_error
    assert json.loads(result.output) == {
        "status": "committed",
        "summary": "Record controlled Wiki tooling guidance",
        "pages": ["concepts/controlled-tools.md"],
        "global_revision": 1,
        "search_index_current": True,
    }
    assert (manager.layout.root / "concepts" / "controlled-tools.md").is_file()


async def test_ingest_accepts_only_current_inline_content_and_commits_in_yolo(
    wiki_tool, manager
) -> None:
    raw = "Current-turn evidence supports controlled global Wiki writes."
    source = CurrentSource(kind="inline", content=raw)
    provenance = SourceRef(
        kind="conversation",
        session_id=_SESSION_ID,
        content_hash=content_hash(raw.encode("utf-8")),
    )

    wiki_tool._runtime.wiki_tool_context = _tool_context(
        conversation_hashes=frozenset({provenance.content_hash})
    )
    result = await wiki_tool(
        Params(operation="ingest", source=source, candidate=_candidate(provenance))
    )

    assert not result.is_error
    assert json.loads(result.output)["status"] == "committed"
    assert (manager.layout.root / "concepts" / "controlled-tools.md").is_file()


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

    wiki_tool._runtime.wiki_tool_context = _tool_context(
        allowed_workspace_ids=frozenset({workspace_id})
    )
    result = await wiki_tool(
        Params(operation="ingest", source=source, candidate=_candidate(provenance))
    )

    assert result.is_error
    assert "archive" in result.message.lower()


async def test_workspace_ingest_requires_trusted_allowed_workspace(
    wiki_tool, manager, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_file = workspace / "evidence.md"
    source_file.write_text("trusted workspace evidence", encoding="utf-8")
    workspace_id = manager.registry.register(workspace)
    source = CurrentSource(
        kind="workspace-file",
        workspace_id=workspace_id,
        relative_path="evidence.md",
    )
    provenance = SourceRef(
        kind="workspace-file",
        workspace_id=workspace_id,
        path="evidence.md",
        content_hash=content_hash(source_file.read_bytes()),
    )

    result = await wiki_tool(
        Params(operation="ingest", source=source, candidate=_candidate(provenance))
    )

    assert result.is_error
    assert "allowed workspace" in result.message.lower()


async def test_candidate_cannot_self_certify_high_value_or_user_confirmation(wiki_tool) -> None:
    wiki_tool._runtime.wiki_tool_context = _tool_context(candidate_high_value=False)

    result = await wiki_tool(Params(operation="remember", candidate=_candidate()))

    assert result.is_error
    assert "trusted" in result.message.lower()


async def test_named_session_uses_trusted_provenance_uuid(wiki_tool) -> None:
    assert wiki_tool._runtime.session.id == "named-shell-session"

    result = await wiki_tool(Params(operation="remember", candidate=_candidate()))

    assert not result.is_error


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


async def test_wiki_domain_conflict_is_a_safe_retryable_tool_error() -> None:
    from kimi_cli.wiki.transaction import WikiConflictError

    class FailingManager:
        def search(self, _query: str, _limit: int):
            raise WikiConflictError("conflict at /Users/private/wiki")

    tool = Wiki(
        cast(
            "Runtime",
            SimpleNamespace(wiki=FailingManager(), session=SimpleNamespace(id="named-session")),
        )
    )

    result = await tool(Params(operation="search", query="anything"))

    assert result.is_error
    assert "retry" in result.message.lower()
    assert "/Users/private/wiki" not in result.message
