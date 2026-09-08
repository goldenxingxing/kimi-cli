"""A block the converter skips must not leak its deltas into another one.

`server_tool_use` and `web_search_tool_result` blocks are ignored at
`content_block_start`, but their argument fragments arrive afterwards as
ordinary `input_json_delta` events. The delta branch has no way to tell which
block a delta belongs to, so those fragments used to be merged into whichever
tool call was pending — appending a server tool's arguments onto a real tool
call and corrupting both.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from kosong.contrib.chat_provider.anthropic import AnthropicStreamedMessage
from kosong.message import ToolCall, ToolCallPart


class _FakeStream:
    """Stands in for the SDK's AsyncStream context manager."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event


class _BlockStart:
    """RawContentBlockStartEvent, as far as the converter looks at it."""

    def __init__(self, index: int, content_block: Any) -> None:
        self.index = index
        self.content_block = content_block


class _BlockDelta:
    def __init__(self, index: int, delta: Any) -> None:
        self.index = index
        self.delta = delta


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, id: str, name: str) -> None:
        self.id = id
        self.name = name


class _ServerToolUseBlock:
    type = "server_tool_use"


class _JsonDelta:
    type = "input_json_delta"

    def __init__(self, partial_json: str) -> None:
        self.partial_json = partial_json


class _Unmatchable:
    """A type nothing in the fake stream is an instance of."""


@pytest.fixture
def patched_event_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the converter's isinstance checks at the stand-ins above."""
    import kosong.contrib.chat_provider.anthropic as module

    monkeypatch.setattr(module, "MessageStartEvent", _Unmatchable)
    monkeypatch.setattr(module, "RawContentBlockStartEvent", _BlockStart)
    monkeypatch.setattr(module, "RawContentBlockDeltaEvent", _BlockDelta)
    monkeypatch.setattr(module, "MessageDeltaEvent", _Unmatchable)
    monkeypatch.setattr(module, "MessageStopEvent", _Unmatchable)


async def _parts(events: list[Any]) -> list[Any]:
    message = AnthropicStreamedMessage(_FakeStream(events))  # type: ignore[arg-type]
    return [part async for part in message]


@pytest.mark.anyio
async def test_a_server_tool_blocks_deltas_do_not_join_the_real_tool_call(
    patched_event_types: None,
) -> None:
    parts = await _parts(
        [
            _BlockStart(0, _ToolUseBlock(id="call-1", name="Read")),
            _BlockDelta(0, _JsonDelta('{"path":')),
            # The converter skips this block's start; its deltas follow anyway.
            _BlockStart(1, _ServerToolUseBlock()),
            _BlockDelta(1, _JsonDelta('{"query": "something else"}')),
            _BlockDelta(0, _JsonDelta('"/etc/hosts"}')),
        ],
    )

    assert [type(p) for p in parts] == [ToolCall, ToolCallPart, ToolCallPart]
    assert [p.arguments_part for p in parts if isinstance(p, ToolCallPart)] == [
        '{"path":',
        '"/etc/hosts"}',
    ]


@pytest.mark.anyio
async def test_an_ordinary_tool_call_still_streams_its_arguments(
    patched_event_types: None,
) -> None:
    parts = await _parts(
        [
            _BlockStart(0, _ToolUseBlock(id="call-1", name="Read")),
            _BlockDelta(0, _JsonDelta('{"path": "a"}')),
        ],
    )

    assert isinstance(parts[0], ToolCall)
    assert parts[0].id == "call-1"
    assert [p.arguments_part for p in parts[1:]] == ['{"path": "a"}']
