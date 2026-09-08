"""Regressions for a batch of review findings across kosong.

Each of these fails on the code as it stood before the accompanying fix.
"""

import asyncio
from typing import Any, override

import pytest
from pydantic import ValidationError

import kosong
from kosong import step
from kosong.chat_provider.mock import MockChatProvider
from kosong.chat_provider.openai_common import ensure_tool_call_arguments
from kosong.contrib.chat_provider.google_genai import _tool_call_id_to_name  # pyright: ignore[reportPrivateUsage]
from kosong.message import Message, TextPart, ToolCall
from kosong.tooling import CallableTool, ParametersType, ToolResult, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from kosong.utils.jsonschema import deref_json_schema


def test_a_self_referential_schema_does_not_blow_the_stack() -> None:
    """A tree-shaped tool parameter used to raise RecursionError at registration."""
    schema: dict[str, Any] = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            }
        },
        "$ref": "#/$defs/Node",
    }

    resolved = deref_json_schema(schema)

    # The cycle stays a `$ref`, so `$defs` has to stay with it.
    assert resolved["properties"] == {  # type: ignore[index]
        "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
    }
    assert "$defs" in resolved


def test_an_acyclic_schema_is_still_fully_inlined() -> None:
    schema: dict[str, Any] = {
        "$defs": {"U": {"type": "string"}},
        "properties": {"a": {"$ref": "#/$defs/U"}, "b": {"$ref": "#/$defs/U"}},
    }

    assert deref_json_schema(schema) == {
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}}
    }


def test_an_unknown_content_part_raises_a_validation_error() -> None:
    """It used to be a bare KeyError, which `except ValidationError` misses.

    History restore and the print UI both guard with ValidationError and skip
    the part; the KeyError took the whole session down instead.
    """
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "assistant", "content": [{"type": "brand_new_part"}]})


def test_a_tool_call_with_no_arguments_still_serializes_them() -> None:
    """`exclude_none=True` dropped the key, and the backends 400 on that."""
    dumped = ToolCall(
        id="1", function=ToolCall.FunctionBody(name="now", arguments=None)
    ).model_dump(exclude_none=True)
    message = {"role": "assistant", "tool_calls": [dumped]}

    ensure_tool_call_arguments(message)

    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_a_gemini_tool_call_id_keeps_the_underscores_in_the_name() -> None:
    """Splitting from the left turned "get_weather_12345" into "get"."""
    assert _tool_call_id_to_name("get_weather_12345", {}) == "get_weather"
    assert _tool_call_id_to_name("known_1", {"known_1": "exact"}) == "exact"


def test_the_mock_provider_hands_out_fresh_parts_each_call() -> None:
    """`generate` merges parts in place, so shared parts accumulated."""
    provider = MockChatProvider([TextPart(text="Hello"), TextPart(text=" world")])

    async def main() -> list[str]:
        return [
            (await kosong.generate(provider, "system", [], [])).message.extract_text()
            for _ in range(3)
        ]

    assert asyncio.run(main()) == ["Hello world"] * 3


def test_step_cancels_the_tool_tasks_when_a_callback_raises() -> None:
    """Only ChatProviderError and CancelledError used to trigger the cleanup.

    Anything else left the spawned tool tasks running with nobody awaiting
    them — a Bash tool's subprocess included.
    """
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowTool(CallableTool):
        name: str = "slow"
        description: str = "never finishes on its own"
        parameters: ParametersType = {"type": "object", "properties": {}}

        @override
        async def __call__(self) -> ToolReturnValue:
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("should have been cancelled")

    tool_call = ToolCall(id="slow#1", function=ToolCall.FunctionBody(name="slow", arguments="{}"))
    # The trailing text parts are what flush the tool call out of `generate`'s
    # pending slot, so the tool task exists by the time the callback raises.
    provider = MockChatProvider([tool_call, TextPart(text="a"), TextPart(text="b")])

    seen: list[Any] = []

    async def explode(part: Any) -> None:
        seen.append(part)
        if len(seen) >= 3:
            await started.wait()
            raise RuntimeError("the UI blew up mid-stream")

    async def main() -> None:
        with pytest.raises(RuntimeError):
            await step(provider, "system", SimpleToolset([SlowTool()]), [], on_message_part=explode)
        # Inside the loop: `asyncio.run` cancels whatever is left on the way
        # out, so checking after it would pass either way.
        assert cancelled.is_set()

    asyncio.run(main())
