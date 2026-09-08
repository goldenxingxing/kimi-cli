import copy
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Self

from kosong.chat_provider import (
    ChatProvider,
    StreamedMessage,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
)
from kosong.message import Message
from kosong.tooling import Tool

if TYPE_CHECKING:

    def type_check(mock: "MockChatProvider"):
        _: ChatProvider = mock


class MockChatProvider(ChatProvider):
    """
    A mock chat provider.
    """

    name = "mock"

    def __init__(
        self,
        message_parts: list[StreamedMessagePart],
    ):
        """Initialize the mock chat provider with predefined message parts."""
        self._message_parts = message_parts

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> "MockStreamedMessage":
        """Always return the predefined message parts."""
        # A copy per call: `kosong.generate` merges parts *in place*, so
        # handing out the same objects twice meant the second generation saw
        # the first one's merged text — [Text("Hello"), Text(" world")] gave
        # "Hello world", then "Hello world world".
        return MockStreamedMessage([part.model_copy(deep=True) for part in self._message_parts])

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        return copy.copy(self)


class MockStreamedMessage(StreamedMessage):
    """The streamed message of the mock chat provider."""

    def __init__(self, message_parts: list[StreamedMessagePart]):
        self._iter = self._to_stream(message_parts)

    async def aclose(self) -> None:
        """Release the underlying stream. See `kosong.generate`."""
        await self._iter.aclose()

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    async def _to_stream(
        self, message_parts: list[StreamedMessagePart]
    ) -> AsyncIterator[StreamedMessagePart]:
        for part in message_parts:
            yield part

    @property
    def id(self) -> str:
        return "mock"

    @property
    def usage(self) -> TokenUsage | None:
        return None

    @property
    def trace_id(self) -> str | None:
        return None
