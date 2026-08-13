"""Which refusals end a turn, and which are just a "no" to one side errand.

Declining ``rm -rf`` means the plan is wrong: carrying on would be carrying on
without the user. Declining a memory write means "do not save that" — the
agent had started it on its own while working on something else, and ending
the turn there abandons the task the user actually asked for, half-done.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kosong.message import Message, TextPart, ToolCall
from kosong.tooling import Tool, ToolResult

from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.approval import Approval, ApprovalResult
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.wire import Wire
from tests.core.test_kimisoul_ralph_loop import RejectTool, _make_llm, _runtime_with_llm


@pytest.fixture
def approval() -> Approval:
    return Approval(yolo=False)


class _RejectingToolset:
    """One tool whose call is always refused, with a configurable scope."""

    def __init__(self, *, stops_turn: bool) -> None:
        self._tool = RejectTool()
        self._stops_turn = stops_turn

    @property
    def tools(self) -> list[Tool]:
        return [self._tool.base]

    def handle(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            return_value=ToolRejectedError(stops_turn=self._stops_turn),
        )


async def _run(soul: KimiSoul, user_input: str) -> None:
    async def _ui_loop_fn(wire: Wire) -> None:
        wire_ui = wire.ui_side(merge=True)
        while True:
            try:
                await wire_ui.receive()
            except QueueShutDown:
                return

    await run_soul(soul, user_input, _ui_loop_fn, asyncio.Event())


def _soul_with(runtime: Runtime, toolset, tmp_path: Path) -> tuple[KimiSoul, Context]:
    llm = _make_llm(
        [
            [
                ToolCall(
                    id="call-1", function=ToolCall.FunctionBody(name="reject_tool", arguments="{}")
                )
            ],
            [TextPart(text="carried on and finished the actual task")],
        ],
        set(),
    )
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=_runtime_with_llm(runtime, llm),
    )
    context = Context(file_backend=tmp_path / "history.jsonl")
    return KimiSoul(agent, context=context), context


def _assistant_texts(context: Context) -> list[str]:
    return [
        part.text
        for message in context.history
        if isinstance(message, Message) and message.role == "assistant"
        for part in message.content
        if isinstance(part, TextPart)
    ]


@pytest.mark.asyncio
async def test_a_turn_ending_rejection_still_ends_the_turn(
    runtime: Runtime, tmp_path: Path
) -> None:
    soul, context = _soul_with(runtime, _RejectingToolset(stops_turn=True), tmp_path)

    await _run(soul, "delete the build directory")

    assert "carried on and finished the actual task" not in _assistant_texts(context)


@pytest.mark.asyncio
async def test_declining_a_side_errand_lets_the_task_finish(
    runtime: Runtime, tmp_path: Path
) -> None:
    """The regression this file exists for."""
    soul, context = _soul_with(runtime, _RejectingToolset(stops_turn=False), tmp_path)

    await _run(soul, "do the actual task")

    assert "carried on and finished the actual task" in _assistant_texts(context)


def test_an_incidental_rejection_tells_the_model_to_continue() -> None:
    """The message matters as much as the flag: the default one orders a stop."""
    error = ApprovalResult(approved=False).rejection_error(stops_turn=False)

    assert error.stops_turn is False
    assert "continue with the task" in error.message.lower()
    assert "stop what you are doing" not in error.message.lower()


def test_rejections_end_the_turn_unless_told_otherwise() -> None:
    """Every other tool keeps the old behaviour; this is opt-in."""
    assert ToolRejectedError().stops_turn is True
    assert ApprovalResult(approved=False).rejection_error().stops_turn is True
