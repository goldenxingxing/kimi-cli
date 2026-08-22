"""The auto-compaction ratio is a global setting that live sessions honour at once."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kimi_cli.wire.jsonrpc import (
    JSONRPCInMessageAdapter,
    JSONRPCSetCompactionRatioMessage,
)


class _FakeStdin:
    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.drained = 0

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        self.drained += 1


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()


def _sent_payloads(stdin: _FakeStdin) -> list[dict[str, Any]]:
    return [json.loads(chunk.decode()) for chunk in stdin.written]


# ---------------------------------------------------------------------------
# The wire message
# ---------------------------------------------------------------------------


def test_the_ratio_message_is_an_inbound_method() -> None:
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "set_compaction_ratio",
            "id": "abc",
            "params": {"ratio": 0.8},
        }
    )

    parsed = JSONRPCInMessageAdapter.validate_json(raw)

    assert isinstance(parsed, JSONRPCSetCompactionRatioMessage)
    assert parsed.params.ratio == 0.8


@pytest.mark.parametrize("ratio", [0.49, 1.0, 1.5, -0.2])
def test_an_out_of_range_ratio_is_refused_at_the_wire(ratio: float) -> None:
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "set_compaction_ratio",
            "id": "abc",
            "params": {"ratio": ratio},
        }
    )

    # Named rather than bare Exception: a blind raises() also passes when the
    # message never reaches validation at all — a renamed method, a typo in the
    # payload — which is the opposite of what this test claims to show.
    with pytest.raises(ValidationError) as refused:
        JSONRPCInMessageAdapter.validate_json(raw)

    assert any("ratio" in str(error["loc"]) for error in refused.value.errors())


# ---------------------------------------------------------------------------
# Reaching a live worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_running_session_is_retuned_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.web.runner.process import SessionProcess

    session = SessionProcess(uuid4())
    process = _FakeProcess()
    monkeypatch.setattr(session, "_process", process, raising=False)
    monkeypatch.setattr(type(session), "is_running", property(lambda _self: True))

    assert await session.apply_compaction_ratio(0.75) is True

    payloads = _sent_payloads(process.stdin)
    assert len(payloads) == 1
    assert payloads[0]["method"] == "set_compaction_ratio"
    assert payloads[0]["params"] == {"ratio": 0.75}
    assert process.stdin.drained == 1
    # Whatever is written must survive a round trip through the real parser.
    assert isinstance(
        JSONRPCInMessageAdapter.validate_json(json.dumps(payloads[0])),
        JSONRPCSetCompactionRatioMessage,
    )


@pytest.mark.asyncio
async def test_a_stopped_session_needs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """It reads the persisted value from config.toml when it next starts."""
    from kimi_cli.web.runner.process import SessionProcess

    session = SessionProcess(uuid4())
    monkeypatch.setattr(type(session), "is_running", property(lambda _self: False))

    assert await session.apply_compaction_ratio(0.75) is False


@pytest.mark.asyncio
async def test_a_broken_pipe_does_not_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.web.runner.process import SessionProcess

    session = SessionProcess(uuid4())

    class _DeadStdin(_FakeStdin):
        def write(self, data: bytes) -> None:
            raise BrokenPipeError("worker went away")

    process = _FakeProcess()
    process.stdin = _DeadStdin()
    monkeypatch.setattr(session, "_process", process, raising=False)
    monkeypatch.setattr(type(session), "is_running", property(lambda _self: True))

    assert await session.apply_compaction_ratio(0.75) is False


@pytest.mark.asyncio
async def test_the_runner_retunes_every_live_session_and_skips_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kimi_cli.web.runner.process import KimiCLIRunner, SessionProcess

    runner = KimiCLIRunner()
    live_a, live_b, stopped = uuid4(), uuid4(), uuid4()
    for session_id, running in ((live_a, True), (live_b, True), (stopped, False)):
        proc = SessionProcess(session_id)
        process = _FakeProcess()
        object.__setattr__(proc, "_process", process)
        proc._running_for_test = running  # type: ignore[attr-defined]
        runner._sessions[session_id] = proc  # type: ignore[attr-defined]

    monkeypatch.setattr(
        SessionProcess,
        "is_running",
        property(lambda self: getattr(self, "_running_for_test", False)),
    )

    applied = await runner.apply_compaction_ratio(0.7)

    assert set(applied) == {live_a, live_b}
    assert stopped not in applied


# ---------------------------------------------------------------------------
# The setting itself
# ---------------------------------------------------------------------------


def test_the_default_ratio_is_exposed_to_the_settings_surface() -> None:
    from kimi_cli.config import Config

    config = Config()

    assert config.loop_control.compaction_trigger_ratio == 0.95


def test_the_ratio_is_read_fresh_on_every_step() -> None:
    """This is what makes an in-place update take effect without a restart.

    `should_auto_compact` is called with the value pulled from the config
    object each step, so mutating that object is enough — no snapshot is taken
    at construction time that would need invalidating.
    """
    from pathlib import Path

    source = Path("src/kimi_cli/soul/kimisoul.py").read_text(encoding="utf-8")

    assert "trigger_ratio=self._loop_control.compaction_trigger_ratio" in source
    assert "self._loop_control = agent.runtime.config.loop_control" in source
