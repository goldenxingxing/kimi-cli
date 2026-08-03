"""Root-loop tests for direct Wiki completion checkpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

import kimi_cli.soul.kimisoul as kimisoul_module
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul, StepOutcome


@pytest.mark.asyncio
async def test_root_no_tool_completion_seals_checkpoint_at_loop_boundary(
    runtime: Runtime,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = AsyncMock()
    monkeypatch.setattr(kimisoul_module, "wire_send", lambda _message: None)
    runtime.wiki_evidence_reporter = reporter
    soul = KimiSoul(
        Agent(
            name="root",
            system_prompt="system",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )
    monkeypatch.setattr(soul, "_checkpoint", AsyncMock())
    monkeypatch.setattr(
        soul,
        "_step",
        AsyncMock(
            return_value=StepOutcome(
                stop_reason="no_tool_calls",
                assistant_message=Message(role="assistant", content="Reusable conclusion"),
            )
        ),
    )

    outcome = await soul._agent_loop()

    assert outcome.stop_reason == "no_tool_calls"
    reporter.seal_root_completion.assert_awaited_once_with("Reusable conclusion")


@pytest.mark.asyncio
async def test_non_completion_stop_does_not_seal_root_checkpoint(
    runtime: Runtime,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = AsyncMock()
    monkeypatch.setattr(kimisoul_module, "wire_send", lambda _message: None)
    runtime.wiki_evidence_reporter = reporter
    soul = KimiSoul(
        Agent(
            name="root",
            system_prompt="system",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )
    monkeypatch.setattr(soul, "_checkpoint", AsyncMock())
    monkeypatch.setattr(
        soul,
        "_step",
        AsyncMock(
            return_value=StepOutcome(
                stop_reason="tool_rejected",
                assistant_message=Message(role="assistant", content="Rejected"),
            )
        ),
    )

    outcome = await soul._agent_loop()

    assert outcome.stop_reason == "tool_rejected"
    reporter.seal_root_completion.assert_not_awaited()
