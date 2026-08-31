"""A browser tab that stops reading must not freeze the session.

Every message a session produces used to be written to each attached socket
inline, on the task that drains the worker's stdout. One client that stopped
draining therefore stopped that read loop, the worker's stdout pipe filled, and
the worker blocked mid-turn -- while `wire.jsonl` kept being written, so
reloading the page revealed a turn that had actually finished. These tests pin
the fan-out to per-connection queues so that backpressure stays on the one
connection it belongs to.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from starlette.websockets import WebSocketState

from kimi_cli.web.runner import process as process_mod
from kimi_cli.web.runner.process import SessionProcess


@pytest.fixture(autouse=True)
def _fast_close_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing a wedged socket waits out a real timeout; don't do that here."""
    monkeypatch.setattr(process_mod, "WS_CLOSE_TIMEOUT_S", 0.05)


EVENT = '{"jsonrpc":"2.0","method":"event","params":{"type":"StepBegin","payload":{"n":%d}}}'


class FakeWS:
    """A WebSocket that records what it is sent."""

    def __init__(self) -> None:
        self.client_state = WebSocketState.CONNECTED
        self.sent: list[str] = []
        self.close_code: int | None = None

    async def send_text(self, message: str) -> None:
        self.sent.append(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.client_state = WebSocketState.DISCONNECTED


class StalledWS(FakeWS):
    """A peer that stops draining its socket: `send_text` never returns."""

    async def send_text(self, message: str) -> None:
        await asyncio.Event().wait()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        # Closing a wedged socket writes a close frame, which blocks for the
        # same reason the data did.
        await asyncio.Event().wait()


def _session_with_stdout() -> tuple[SessionProcess, asyncio.StreamReader]:
    sp = SessionProcess(uuid4())
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stderr.feed_eof()
    process = MagicMock()
    process.stdout = stdout
    process.stderr = stderr
    process.returncode = None
    sp._process = process
    return sp, stdout


async def _attach(sp: SessionProcess, ws: FakeWS) -> None:
    await sp.add_websocket_and_begin_replay(ws)  # type: ignore[arg-type]
    await sp.end_replay(ws)  # type: ignore[arg-type]


async def _run_read_loop(sp: SessionProcess, stdout: asyncio.StreamReader, lines: int) -> None:
    read_task = asyncio.create_task(sp._read_loop())
    for i in range(lines):
        stdout.feed_data((EVENT % i).encode() + b"\n")
    await asyncio.sleep(0.2)
    read_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await read_task


@pytest.mark.asyncio
async def test_stalled_client_does_not_block_other_clients() -> None:
    sp, stdout = _session_with_stdout()
    stalled = StalledWS()
    live = FakeWS()
    await _attach(sp, stalled)
    await _attach(sp, live)

    await _run_read_loop(sp, stdout, 5)

    assert len(live.sent) == 5, f"stalled peer starved a healthy client: {len(live.sent)}/5"
    await sp._close_all_websockets()


@pytest.mark.asyncio
async def test_stalled_client_does_not_stop_draining_worker_stdout() -> None:
    """The read loop must keep consuming stdout, or the worker blocks mid-turn."""
    sp, stdout = _session_with_stdout()
    await _attach(sp, StalledWS())

    await _run_read_loop(sp, stdout, 50)

    unread = len(stdout._buffer)  # type: ignore[attr-defined]
    assert unread == 0, f"read loop stopped draining worker stdout ({unread} bytes stuck)"
    await sp._close_all_websockets()


@pytest.mark.asyncio
async def test_hopelessly_behind_client_is_dropped() -> None:
    """A client that can never catch up is detached instead of held forever."""
    sp, stdout = _session_with_stdout()
    stalled = StalledWS()
    await _attach(sp, stalled)

    original = process_mod.WS_SEND_QUEUE_MAX_MESSAGES
    process_mod.WS_SEND_QUEUE_MAX_MESSAGES = 4
    try:
        await _run_read_loop(sp, stdout, 20)
    finally:
        process_mod.WS_SEND_QUEUE_MAX_MESSAGES = original

    assert sp.websocket_count == 0, "an unreachable client stayed attached"


@pytest.mark.asyncio
async def test_messages_broadcast_during_replay_arrive_in_order() -> None:
    """Live messages produced while history replays are delivered afterwards."""
    sp, _ = _session_with_stdout()
    ws = FakeWS()
    await sp.add_websocket_and_begin_replay(ws)  # type: ignore[arg-type]

    await sp._broadcast("live-1")
    await sp._broadcast("live-2")
    assert ws.sent == [], "live messages leaked into the middle of history replay"

    ws.sent.append("history-tail")
    await sp.end_replay(ws)  # type: ignore[arg-type]
    await sp._broadcast("live-3")
    await asyncio.sleep(0.05)

    assert ws.sent == ["history-tail", "live-1", "live-2", "live-3"]
    await sp._close_all_websockets()


@pytest.mark.asyncio
async def test_remove_websocket_stops_its_writer() -> None:
    sp, _ = _session_with_stdout()
    ws = FakeWS()
    await _attach(sp, ws)
    await sp._broadcast("one")
    await asyncio.sleep(0.05)

    await sp.remove_websocket(ws)  # type: ignore[arg-type]
    await sp._broadcast("two")
    await asyncio.sleep(0.05)

    assert ws.sent == ["one"]
    assert sp.websocket_count == 0
