"""Tests for the request-body byte guard and 400 "message size exceeds limit" self-heal.

Covers:
- ``estimate_message_bytes``: serialized byte estimation incl. image/think/tool_call parts
- ``should_auto_compact``: byte-based trigger condition extension
- ``strip_image_parts``: immutable image stripping with last-user-message exemption
- ``_is_request_too_large_error`` / ``classify_api_error``: 400 message-size recognition
"""

from __future__ import annotations

from kosong.chat_provider import APIStatusError
from kosong.message import ImageURLPart, Message, ToolCall

from kimi_cli.soul.compaction import estimate_message_bytes, should_auto_compact
from kimi_cli.soul.kimisoul import _is_request_too_large_error, classify_api_error
from kimi_cli.soul.message import STRIPPED_IMAGE_PLACEHOLDER, strip_image_parts
from kimi_cli.wire.types import TextPart, ThinkPart

# --- estimate_message_bytes tests ---


def _image_part(url: str) -> ImageURLPart:
    return ImageURLPart(image_url=ImageURLPart.ImageURL(url=url))


class TestEstimateMessageBytes:
    def test_empty_messages(self):
        assert estimate_message_bytes([]) == 0

    def test_text_part_counts_utf8_bytes(self):
        messages = [Message(role="user", content=[TextPart(text="abcd")])]
        assert estimate_message_bytes(messages) == len(b"user") + 4

    def test_cjk_text_counts_multibyte(self):
        # "你好" is 6 bytes in UTF-8, 2 chars
        messages = [Message(role="user", content=[TextPart(text="你好")])]
        assert estimate_message_bytes(messages) == len(b"user") + 6

    def test_image_part_counts_full_url(self):
        url = "data:image/png;base64," + "A" * 100_000
        messages = [
            Message(role="user", content=[TextPart(text="look"), _image_part(url)]),
        ]
        estimated = estimate_message_bytes(messages)
        assert estimated >= len(url)

    def test_image_dominates_over_text(self):
        """A large base64 image must not be missed by the byte estimate."""
        url = "data:image/jpeg;base64," + "B" * 1_000_000
        messages = [
            Message(role="user", content=[TextPart(text="tiny"), _image_part(url)]),
            Message(role="assistant", content=[TextPart(text="also tiny")]),
        ]
        assert estimate_message_bytes(messages) > 1_000_000

    def test_think_part_counted(self):
        messages = [
            Message(
                role="assistant",
                content=[ThinkPart(think="x" * 100, encrypted="y" * 20)],
            )
        ]
        assert estimate_message_bytes(messages) == len(b"assistant") + 120

    def test_tool_calls_arguments_counted(self):
        messages = [
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCall.FunctionBody(
                            name="ReadFile", arguments='{"path": "' + "p" * 500 + '"}'
                        ),
                    )
                ],
            )
        ]
        estimated = estimate_message_bytes(messages)
        assert estimated > 500

    def test_tool_result_message_metadata_counted(self):
        messages = [
            Message(
                role="tool",
                content=[TextPart(text="output")],
                tool_call_id="call_1",
                name="ReadFile",
            )
        ]
        estimated = estimate_message_bytes(messages)
        assert estimated == (len(b"tool") + len(b"call_1") + len(b"ReadFile") + 6)


# --- should_auto_compact byte-trigger tests ---


class TestShouldAutoCompactBytes:
    def test_bytes_over_threshold_triggers_despite_low_tokens(self):
        """Byte-heavy history (e.g. images) must compact even at ~0 token usage."""
        assert should_auto_compact(
            1_000,
            200_000,
            trigger_ratio=0.95,
            reserved_context_size=50_000,
            request_bytes=1_900_000,
            max_request_bytes=1_800_000,
        )

    def test_bytes_at_threshold_does_not_trigger(self):
        assert not should_auto_compact(
            1_000,
            200_000,
            trigger_ratio=0.95,
            reserved_context_size=50_000,
            request_bytes=1_800_000,
            max_request_bytes=1_800_000,
        )

    def test_bytes_under_threshold_and_low_tokens_no_trigger(self):
        assert not should_auto_compact(
            1_000,
            200_000,
            trigger_ratio=0.95,
            reserved_context_size=50_000,
            request_bytes=500_000,
            max_request_bytes=1_800_000,
        )

    def test_token_trigger_still_works_with_byte_args(self):
        assert should_auto_compact(
            190_000,
            200_000,
            trigger_ratio=0.95,
            reserved_context_size=50_000,
            request_bytes=100,
            max_request_bytes=1_800_000,
        )

    def test_without_byte_args_behaves_as_before(self):
        assert not should_auto_compact(
            1_000, 200_000, trigger_ratio=0.95, reserved_context_size=50_000
        )
        assert should_auto_compact(
            190_000, 200_000, trigger_ratio=0.95, reserved_context_size=50_000
        )


# --- strip_image_parts tests ---


class TestStripImageParts:
    def _messages(self) -> list[Message]:
        return [
            Message(
                role="user",
                content=[TextPart(text="first"), _image_part("data:image/png;base64,OLD")],
            ),
            Message(
                role="assistant",
                content=[TextPart(text="reply"), _image_part("data:image/png;base64,MID")],
            ),
            Message(
                role="user",
                content=[TextPart(text="latest"), _image_part("data:image/png;base64,NEW")],
            ),
        ]

    def test_strips_all_but_last_user_message(self):
        stripped, count = strip_image_parts(self._messages())

        assert count == 2
        # last user message keeps its image
        assert any(isinstance(p, ImageURLPart) for p in stripped[2].content)
        # earlier messages have placeholders instead of images
        for msg in stripped[:2]:
            assert not any(isinstance(p, ImageURLPart) for p in msg.content)
            assert any(
                isinstance(p, TextPart) and p.text == STRIPPED_IMAGE_PLACEHOLDER
                for p in msg.content
            )

    def test_keep_last_user_message_false_strips_everything(self):
        stripped, count = strip_image_parts(self._messages(), keep_last_user_message=False)

        assert count == 3
        for msg in stripped:
            assert not any(isinstance(p, ImageURLPart) for p in msg.content)

    def test_originals_not_mutated(self):
        messages = self._messages()
        original_first_content = messages[0].content

        stripped, _ = strip_image_parts(messages)

        assert messages[0].content is original_first_content
        assert any(isinstance(p, ImageURLPart) for p in messages[0].content)
        assert stripped[0] is not messages[0]

    def test_no_images_returns_same_messages(self):
        messages = [
            Message(role="user", content=[TextPart(text="hello")]),
            Message(role="assistant", content=[TextPart(text="hi")]),
        ]
        stripped, count = strip_image_parts(messages)

        assert count == 0
        assert stripped == messages
        assert all(s is m for s, m in zip(stripped, messages, strict=True))

    def test_text_parts_preserved_around_stripped_images(self):
        messages = [
            Message(
                role="user",
                content=[
                    TextPart(text="before"),
                    _image_part("data:image/png;base64,IMG"),
                    TextPart(text="after"),
                ],
            ),
            Message(role="user", content=[TextPart(text="latest question")]),
        ]
        stripped, count = strip_image_parts(messages)

        assert count == 1
        texts = [p.text for p in stripped[0].content if isinstance(p, TextPart)]
        assert texts == ["before", STRIPPED_IMAGE_PLACEHOLDER, "after"]


# --- 400 "message size exceeds limit" recognition tests ---

_TOO_LARGE_MESSAGE = (
    "{'error': {'message': 'total message size 3000000 exceeds limit 2097152', "
    "'type': 'invalid_request_error'}}"
)


class TestIsRequestTooLargeError:
    def test_400_message_size_exceeds_limit(self):
        err = APIStatusError(400, _TOO_LARGE_MESSAGE)
        assert _is_request_too_large_error(err)

    def test_400_other_message(self):
        err = APIStatusError(400, "invalid request: bad parameter")
        assert not _is_request_too_large_error(err)

    def test_non_400_status(self):
        err = APIStatusError(500, "total message size 3000000 exceeds limit 2097152")
        assert not _is_request_too_large_error(err)

    def test_non_status_error(self):
        assert not _is_request_too_large_error(ValueError("message size exceeds limit"))

    def test_requires_both_keywords(self):
        assert not _is_request_too_large_error(APIStatusError(400, "message size too big"))
        assert not _is_request_too_large_error(APIStatusError(400, "token exceeds limit"))


class TestClassifyApiErrorMessageSize:
    def test_message_size_classified_as_context_overflow(self):
        err = APIStatusError(400, _TOO_LARGE_MESSAGE)
        error_type, status_code = classify_api_error(err)
        assert error_type == "context_overflow"
        assert status_code == 400

    def test_other_400_remains_4xx_client(self):
        err = APIStatusError(400, "invalid api key format")
        error_type, status_code = classify_api_error(err)
        assert error_type == "4xx_client"
        assert status_code == 400
