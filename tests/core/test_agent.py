from __future__ import annotations

import asyncio
import dataclasses
import threading
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from kaos.local import local_kaos
from kaos.path import KaosPath

import kimi_cli.soul.agent as agent_module
from kimi_cli.agentspec import DEFAULT_AGENT_FILE
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.metadata import WorkDirMeta
from kimi_cli.session import Session
from kimi_cli.session_state import SessionState
from kimi_cli.soul.agent import Runtime, _initialize_global_wiki, _load_system_prompt
from kimi_cli.tools.wiki import WikiToolContext
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wire.file import WireFile


@pytest.fixture
def lightweight_runtime_create(
    monkeypatch: pytest.MonkeyPatch,
    environment,
    tmp_path: Path,
) -> Path:
    wiki_root = tmp_path / "shared-data" / "users" / "default" / "wiki"
    monkeypatch.setenv("OPENKIMO_WIKI_ROOT", str(wiki_root))
    monkeypatch.setattr(agent_module, "list_directory", AsyncMock(return_value=""))
    monkeypatch.setattr(agent_module, "load_agents_md", AsyncMock(return_value=None))
    monkeypatch.setattr(agent_module, "load_knowledge_base", lambda _work_dir: None)
    monkeypatch.setattr(agent_module.Environment, "detect", AsyncMock(return_value=environment))
    monkeypatch.setattr(agent_module, "resolve_skills_roots", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_module, "discover_skills_from_roots", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_module, "index_skills", lambda _skills: {})
    monkeypatch.setattr(agent_module, "format_skills_for_prompt", lambda _skills: None)
    return wiki_root


def _session(work_dir: Path, session_root: Path, *, session_id: str, kaos: str) -> Session:
    work_dir.mkdir(parents=True)
    session_root.mkdir(parents=True)
    return Session(
        id=session_id,
        work_dir=KaosPath.unsafe_from_local_path(work_dir),
        work_dir_meta=WorkDirMeta(path=str(work_dir), kaos=kaos),
        context_file=session_root / "context.jsonl",
        wire_file=WireFile(path=session_root / "wire.jsonl"),
        state=SessionState(),
        title="Test Session",
        updated_at=0.0,
        storage_dir=session_root,
    )


@pytest.mark.asyncio
async def test_unrelated_sessions_share_global_wiki_and_trusted_runtime_context(
    config,
    tmp_path: Path,
    lightweight_runtime_create: Path,
) -> None:
    session_a = _session(
        tmp_path / "workspace-a",
        tmp_path / "sessions" / "a",
        session_id="named-session-a",
        kaos=local_kaos.name,
    )
    session_b = _session(
        tmp_path / "workspace-b",
        tmp_path / "sessions" / "b",
        session_id="named-session-b",
        kaos=local_kaos.name,
    )

    runtime_a = await Runtime.create(
        config, OAuthManager(config), llm=None, session=session_a, yolo=False
    )
    runtime_b = await Runtime.create(
        config, OAuthManager(config), llm=None, session=session_b, yolo=False
    )
    try:
        assert runtime_a.wiki is not None
        assert runtime_b.wiki is not None
        assert runtime_a.wiki.layout.root == lightweight_runtime_create.resolve()
        assert runtime_b.wiki.layout.root == lightweight_runtime_create.resolve()
        assert runtime_a.workspace_id is not None
        assert runtime_b.workspace_id is not None
        assert runtime_a.workspace_id != runtime_b.workspace_id
        assert isinstance(runtime_a.wiki_tool_context, WikiToolContext)
        assert runtime_a.wiki_tool_context.provenance_session_id != UUID(int=0)
        assert runtime_a.wiki_tool_context.allowed_workspace_ids == frozenset(
            {runtime_a.workspace_id}
        )
        assert runtime_a.wiki_tool_context.candidate_high_value is False
        assert runtime_a.wiki_tool_context.stable is False
        assert runtime_a.wiki_tool_context.user_confirmed is False
        assert runtime_a.builtin_args.KIMI_WIKI_CONTEXT.count(".\n") == 3
        assert len(runtime_a.builtin_args.KIMI_WIKI_CONTEXT.encode("utf-8")) <= 8192
    finally:
        assert runtime_a.wiki is not None
        assert runtime_b.wiki is not None
        runtime_a.wiki.close()
        runtime_b.wiki.close()


@pytest.mark.asyncio
async def test_subagent_shares_wiki_workspace_and_trusted_context(
    config,
    tmp_path: Path,
    lightweight_runtime_create: Path,
) -> None:
    del lightweight_runtime_create
    session = _session(
        tmp_path / "workspace",
        tmp_path / "sessions" / "root",
        session_id="named-session",
        kaos=local_kaos.name,
    )
    runtime = await Runtime.create(
        config, OAuthManager(config), llm=None, session=session, yolo=False
    )
    try:
        subagent = runtime.copy_for_subagent(agent_id="worker", subagent_type="coder")

        assert subagent.wiki is runtime.wiki
        assert subagent.workspace_id == runtime.workspace_id
        assert subagent.wiki_tool_context is runtime.wiki_tool_context
    finally:
        assert runtime.wiki is not None
        runtime.wiki.close()


@pytest.mark.asyncio
async def test_remote_workspace_keeps_shared_wiki_without_local_provenance_registration(
    config,
    tmp_path: Path,
    lightweight_runtime_create: Path,
) -> None:
    session = _session(
        tmp_path / "remote-placeholder",
        tmp_path / "sessions" / "remote",
        session_id="remote-session",
        kaos="ssh",
    )

    runtime = await Runtime.create(
        config, OAuthManager(config), llm=None, session=session, yolo=False
    )
    try:
        assert runtime.wiki is not None
        assert runtime.wiki.layout.root == lightweight_runtime_create.resolve()
        assert runtime.workspace_id is None
        assert isinstance(runtime.wiki_tool_context, WikiToolContext)
        assert runtime.wiki_tool_context.allowed_workspace_ids == frozenset()
    finally:
        assert runtime.wiki is not None
        runtime.wiki.close()


@pytest.mark.asyncio
async def test_wiki_failure_does_not_block_runtime(
    config,
    session,
    lightweight_runtime_create: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del lightweight_runtime_create

    def fail_ensure(_self: WikiManager):
        raise OSError("disk")

    monkeypatch.setattr(WikiManager, "ensure", fail_ensure)

    runtime = await Runtime.create(
        config, OAuthManager(config), llm=None, session=session, yolo=False
    )

    assert runtime.wiki is None
    assert runtime.workspace_id is None
    assert runtime.wiki_tool_context is None
    assert runtime.builtin_args.KIMI_WIKI_CONTEXT == ""
    views = runtime.notifications.store.list_views()
    assert len(views) == 1
    event = views[0].event
    assert event.type == "wiki.unavailable"
    assert event.targets == ["wire", "shell"]
    assert "Wiki" in event.title
    assert "当前会话仍可继续" in event.body
    assert "disk" not in event.title + event.body
    assert "/" not in event.title + event.body

    again = await Runtime.create(
        config, OAuthManager(config), llm=None, session=session, yolo=False
    )
    assert again.wiki is None
    assert len(again.notifications.store.list_views()) == 1


@pytest.mark.asyncio
async def test_cancelled_wiki_initialization_closes_partial_manager(
    session,
    lightweight_runtime_create: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del lightweight_runtime_create
    close = Mock()
    monkeypatch.setattr(WikiManager, "ensure", Mock(side_effect=asyncio.CancelledError()))
    monkeypatch.setattr(WikiManager, "close", close)

    with pytest.raises(asyncio.CancelledError):
        await _initialize_global_wiki(session, owner_id=None)

    close.assert_called_once_with()


@pytest.mark.asyncio
async def test_cancelled_wiki_constructor_waits_for_thread_and_closes_manager(
    session,
    lightweight_runtime_create: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del lightweight_runtime_create
    started = threading.Event()
    release = threading.Event()
    close = Mock()

    class SlowWikiManager:
        def __init__(self) -> None:
            started.set()
            release.wait(timeout=5)

        def close(self) -> None:
            close()

    monkeypatch.setattr("kimi_cli.wiki.manager.WikiManager", SlowWikiManager)

    task = asyncio.create_task(_initialize_global_wiki(session, owner_id=None))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    close.assert_called_once_with()


@pytest.mark.asyncio
async def test_only_root_runtime_closes_shared_wiki_once(
    config,
    tmp_path: Path,
    lightweight_runtime_create: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del lightweight_runtime_create
    session = _session(
        tmp_path / "workspace-close",
        tmp_path / "sessions" / "close",
        session_id="close-session",
        kaos=local_kaos.name,
    )
    runtime = await Runtime.create(
        config, OAuthManager(config), llm=None, session=session, yolo=False
    )
    assert runtime.wiki is not None
    close = Mock(wraps=runtime.wiki.search_index.close)
    monkeypatch.setattr(runtime.wiki.search_index, "close", close)
    subagent = runtime.copy_for_subagent(agent_id="worker", subagent_type="coder")

    await subagent.close()
    await asyncio.gather(runtime.close(), runtime.close())

    close.assert_called_once_with()


def test_global_wiki_prompt_section_is_conditional(builtin_args) -> None:
    empty_prompt = _load_system_prompt(
        DEFAULT_AGENT_FILE.parent / "system.md",
        {"ROLE_ADDITIONAL": ""},
        builtin_args,
    )
    with_wiki = dataclasses.replace(
        builtin_args,
        KIMI_WIKI_CONTEXT=(
            "The global Wiki is shared across all workspaces.\n"
            "Use Wiki search/read for durable knowledge.\n"
            "Propose only durable, sourced conclusions for writing.\n\n"
            "# Wiki Index"
        ),
    )
    rendered_prompt = _load_system_prompt(
        DEFAULT_AGENT_FILE.parent / "system.md",
        {"ROLE_ADDITIONAL": ""},
        with_wiki,
    )

    assert "# Global Wiki" not in empty_prompt
    assert "# Global Wiki" in rendered_prompt
    assert "# Wiki Index" in rendered_prompt
