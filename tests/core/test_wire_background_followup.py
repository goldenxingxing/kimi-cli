"""A finished background task should speak up on its own.

The interactive shell has always done this. A client that only runs a turn
when it is sent one had no such loop, so the result sat unread: the foreground
session had ended, and the answer waited for the user to type something
unrelated before it surfaced.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.wire.server import WireServer


def _make_soul(runtime: Runtime, tmp_path: Path) -> KimiSoul:
    agent = Agent(
        name="Follow-up Test Agent",
        system_prompt="Test prompt.",
        toolset=EmptyToolset(),
        runtime=runtime,
    )
    return KimiSoul(agent, context=Context(file_backend=tmp_path / "history.jsonl"))


async def _drive(server: WireServer, runtime: Runtime, *, pending: bool) -> list[str]:
    """Run one pass of the loop and report the turns it started."""
    started: list[str] = []

    async def fake_turn() -> None:
        started.append("turn")

    server._run_followup_turn = fake_turn  # type: ignore[method-assign]
    runtime.notifications.has_pending_for_sink = lambda sink: pending  # type: ignore[method-assign]
    runtime.config.background.auto_followup_coalesce_ms = 0

    task = asyncio.create_task(server._background_followup_loop())
    runtime.background_tasks.completion_event.set()
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return started


@pytest.mark.asyncio
async def test_a_completion_while_idle_starts_a_turn(runtime: Runtime, tmp_path: Path) -> None:
    server = WireServer(_make_soul(runtime, tmp_path))
    server._followup_armed = True

    assert await _drive(server, runtime, pending=True) == ["turn"]


@pytest.mark.asyncio
async def test_nothing_happens_before_the_user_has_spoken(runtime: Runtime, tmp_path: Path) -> None:
    """Reopening a session with old notifications must not start it talking."""
    server = WireServer(_make_soul(runtime, tmp_path))
    server._followup_armed = False

    assert await _drive(server, runtime, pending=True) == []


@pytest.mark.asyncio
async def test_nothing_happens_when_there_is_nothing_to_report(
    runtime: Runtime, tmp_path: Path
) -> None:
    server = WireServer(_make_soul(runtime, tmp_path))
    server._followup_armed = True

    assert await _drive(server, runtime, pending=False) == []


@pytest.mark.asyncio
async def test_a_running_turn_is_left_to_read_it_itself(runtime: Runtime, tmp_path: Path) -> None:
    """A turn in flight already picks the notification up; two would collide."""
    server = WireServer(_make_soul(runtime, tmp_path))
    server._followup_armed = True
    server._cancel_event = asyncio.Event()  # i.e. streaming

    assert await _drive(server, runtime, pending=True) == []


@pytest.mark.asyncio
async def test_it_gives_up_after_repeated_failures(runtime: Runtime, tmp_path: Path) -> None:
    """A broken provider must not be retried on every completion forever."""
    server = WireServer(_make_soul(runtime, tmp_path))
    server._followup_armed = True
    server._followup_failures = server._MAX_FOLLOWUP_FAILURES

    assert await _drive(server, runtime, pending=True) == []


@pytest.mark.asyncio
async def test_a_failing_turn_is_counted_and_does_not_kill_the_loop(
    runtime: Runtime, tmp_path: Path
) -> None:
    server = WireServer(_make_soul(runtime, tmp_path))
    server._followup_armed = True
    runtime.notifications.has_pending_for_sink = lambda sink: True  # type: ignore[method-assign]
    runtime.config.background.auto_followup_coalesce_ms = 0

    async def boom() -> None:
        raise RuntimeError("provider is down")

    server._run_followup_turn = boom  # type: ignore[method-assign]

    task = asyncio.create_task(server._background_followup_loop())
    runtime.background_tasks.completion_event.set()
    await asyncio.sleep(0.05)
    assert server._followup_failures >= 1
    assert not task.done(), "one bad turn must not end the loop"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
