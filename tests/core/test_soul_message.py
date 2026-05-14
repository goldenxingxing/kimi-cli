from __future__ import annotations

from io import BytesIO

from inline_snapshot import snapshot
from kosong.message import Message
from kosong.tooling import ToolError, ToolOk

from kimi_cli.llm import ModelCapability
from kimi_cli.soul.message import (
    check_message,
    sanitize_image_parts,
    system,
    tool_result_to_message,
)
from kimi_cli.wire.types import (
    AudioURLPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    ToolResult,
    VideoURLPart,
)


def test_system_message_creation():
    """Test that system messages are properly formatted."""
    message = "Test message"
    assert system(message) == snapshot(TextPart(text="<system>Test message</system>"))


def test_tool_ok_with_string_output():
    """Test ToolOk with string output."""
    tool_ok = ToolOk(output="Hello, world!")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)
    message = tool_result_to_message(tool_result)
    assert message == snapshot(
        Message(role="tool", content=[TextPart(text="Hello, world!")], tool_call_id="call_123")
    )


def test_tool_ok_with_message():
    """Test ToolOk with explanatory message."""
    tool_ok = ToolOk(output="Result", message="Operation completed")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)
    message = tool_result_to_message(tool_result)
    assert message == snapshot(
        Message(
            role="tool",
            content=[
                TextPart(text="<system>Operation completed</system>"),
                TextPart(text="Result"),
            ],
            tool_call_id="call_123",
        )
    )


def test_tool_ok_with_content_part():
    """Test ToolOk with ContentPart output."""
    content_part = TextPart(text="Text content")
    tool_ok = ToolOk(output=content_part)
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)
    message = tool_result_to_message(tool_result)
    assert message == snapshot(
        Message(role="tool", content=[TextPart(text="Text content")], tool_call_id="call_123")
    )


def test_tool_ok_with_sequence_of_parts():
    """Test ToolOk with sequence of ContentParts."""
    text_part = TextPart(text="Text content")
    text_part_2 = TextPart(text="Text content 2")
    tool_ok = ToolOk(output=[text_part, text_part_2])
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)
    message = tool_result_to_message(tool_result)
    assert message == snapshot(
        Message(
            role="tool",
            content=[TextPart(text="Text content"), TextPart(text="Text content 2")],
            tool_call_id="call_123",
        )
    )


def test_tool_ok_with_empty_output():
    """Test ToolOk with empty output."""
    tool_ok = ToolOk(output="")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)
    message = tool_result_to_message(tool_result)
    assert message == snapshot(
        Message(
            role="tool",
            content=[TextPart(text="<system>Tool output is empty.</system>")],
            tool_call_id="call_123",
        )
    )


def test_tool_ok_with_message_but_empty_output():
    """Test ToolOk with message but empty output."""
    tool_ok = ToolOk(output="", message="Just a message")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)
    message = tool_result_to_message(tool_result)
    assert message == snapshot(
        Message(
            role="tool",
            content=[TextPart(text="<system>Just a message</system>")],
            tool_call_id="call_123",
        )
    )


def test_tool_error_result():
    """Test ToolResult with ToolError."""
    tool_error = ToolError(message="Error occurred", brief="Brief error", output="Error details")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_error)

    message = tool_result_to_message(tool_result)

    assert isinstance(message, Message)
    assert message.role == "tool"
    assert message.tool_call_id == "call_123"
    assert len(message.content) == 2  # System message + error output
    assert message.content[0] == system("ERROR: Error occurred")
    assert message.content[1] == TextPart(text="Error details")


def test_tool_error_without_output():
    """Test ToolResult with ToolError without output."""
    tool_error = ToolError(message="Error occurred", brief="Brief error")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_error)

    message = tool_result_to_message(tool_result)

    assert isinstance(message, Message)
    assert message.role == "tool"
    assert len(message.content) == 1  # Only system message
    assert message.content[0] == system("ERROR: Error occurred")


def test_tool_ok_with_text_only():
    """Test ToolResult with ToolOk containing only text parts."""
    tool_ok = ToolOk(output="Simple output", message="Done")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    assert isinstance(message, Message)
    assert message.role == "tool"
    assert message.tool_call_id == "call_123"
    # Should have system message from ToolOk + text output
    assert len(message.content) == 2
    assert message.content[0] == system("Done")
    assert message.content[1] == TextPart(text="Simple output")


def test_tool_ok_with_non_text_parts():
    """Test ToolResult with ToolOk containing non-text parts."""
    text_part = TextPart(text="Text content")
    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.jpg"))
    tool_ok = ToolOk(output=[text_part, image_part], message="Mixed content")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)

    # With current implementation, non-text parts are included in the same message
    message = tool_result_to_message(tool_result)

    assert isinstance(message, Message)
    assert message.role == "tool"
    assert message.tool_call_id == "call_123"

    # Should have system message + text part + image part
    assert len(message.content) == 3
    assert message.content[0] == system("Mixed content")
    assert message.content[1] == text_part
    assert message.content[2] == image_part


def test_tool_ok_with_only_non_text_parts():
    """Test ToolResult with ToolOk containing only non-text parts.

    When a tool returns only non-text content (e.g. image from MCP tools),
    a TextPart must be prepended so the LLM API doesn't reject the message
    with "text content is empty" (see #1663).
    """
    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.jpg"))
    tool_ok = ToolOk(output=image_part)
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    assert isinstance(message, Message)
    assert message.role == "tool"
    assert message.tool_call_id == "call_123"
    # Must have a TextPart prepended + original image part
    assert len(message.content) == 2
    assert isinstance(message.content[0], TextPart)
    assert message.content[1] == image_part


def test_tool_ok_with_only_image_list():
    """Test ToolResult with ToolOk containing a list of only image parts."""
    img1 = ImageURLPart(image_url=ImageURLPart.ImageURL(url="data:image/png;base64,abc"))
    img2 = ImageURLPart(image_url=ImageURLPart.ImageURL(url="data:image/png;base64,def"))
    tool_ok = ToolOk(output=[img1, img2])
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    assert isinstance(message.content[0], TextPart)
    assert message.content[1] == img1
    assert message.content[2] == img2


def test_tool_ok_with_only_audio_part():
    """Test ToolResult with ToolOk containing only audio content."""
    audio_part = AudioURLPart(audio_url=AudioURLPart.AudioURL(url="data:audio/mp3;base64,abc"))
    tool_ok = ToolOk(output=audio_part)
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    assert isinstance(message.content[0], TextPart)
    assert message.content[1] == audio_part


def test_tool_ok_with_message_and_only_image():
    """Test ToolResult with message but only image output — message provides TextPart."""
    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/img.jpg"))
    tool_ok = ToolOk(output=image_part, message="Screenshot captured")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    # The message field provides a TextPart via system(), so no extra TextPart needed
    assert isinstance(message.content[0], TextPart)
    assert "Screenshot captured" in message.content[0].text
    assert message.content[1] == image_part


def test_tool_ok_with_only_text_parts():
    """Test ToolResult with ToolOk containing only text parts."""
    tool_ok = ToolOk(output="Just text")
    tool_result = ToolResult(tool_call_id="call_123", return_value=tool_ok)

    message = tool_result_to_message(tool_result)

    assert isinstance(message, Message)
    assert message.role == "tool"
    assert len(message.content) == 1
    assert message.content[0] == TextPart(text="Just text")


def test_check_message_with_image_and_image_capability():
    """Test check_message with ImageURLPart when model has image_in capability."""
    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.jpg"))
    message = Message(role="user", content=[image_part])
    model_capabilities: set[ModelCapability] = {"image_in", "thinking"}

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == set()


def test_check_message_with_image_no_image_capability():
    """Test check_message with ImageURLPart when model lacks image_in capability."""
    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.jpg"))
    message = Message(role="user", content=[image_part])
    model_capabilities: set[ModelCapability] = {"thinking"}

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == {"image_in"}


def test_check_message_with_video_and_video_capability():
    """Test check_message with VideoURLPart when model has video_in capability."""
    video_part = VideoURLPart(video_url=VideoURLPart.VideoURL(url="https://example.com/video.mp4"))
    message = Message(role="user", content=[video_part])
    model_capabilities: set[ModelCapability] = {"video_in"}

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == set()


def test_check_message_with_video_no_video_capability():
    """Test check_message with VideoURLPart when model lacks video_in capability."""
    video_part = VideoURLPart(video_url=VideoURLPart.VideoURL(url="https://example.com/video.mp4"))
    message = Message(role="user", content=[video_part])
    model_capabilities: set[ModelCapability] = {"image_in"}

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == {"video_in"}


def test_check_message_with_think_and_think_capability():
    """Test check_message with ThinkPart when model has thinking capability."""
    think_part = ThinkPart(think="This is a thinking process")
    message = Message(role="assistant", content=[think_part])
    model_capabilities: set[ModelCapability] = {"image_in", "thinking"}

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == set()


def test_check_message_with_think_no_think_capability():
    """Test check_message with ThinkPart when model lacks thinking capability."""
    think_part = ThinkPart(think="This is a thinking process")
    message = Message(role="assistant", content=[think_part])
    model_capabilities: set[ModelCapability] = {"image_in"}

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == {"thinking"}


def test_check_message_with_mixed_parts_partial_capabilities():
    """Test check_message with both ImageURLPart and ThinkPart, model has only one capability."""
    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.jpg"))
    think_part = ThinkPart(think="Thinking...")
    message = Message(role="user", content=[image_part, think_part])
    model_capabilities: set[ModelCapability] = {"image_in"}

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == {"thinking"}


def test_check_message_with_text_only():
    """Test check_message with only TextPart (no special capabilities needed)."""
    text_part = TextPart(text="Just a text message")
    message = Message(role="user", content=[text_part])
    model_capabilities: set[ModelCapability] = set()

    missing_capabilities = check_message(message, model_capabilities)

    assert missing_capabilities == set()


def test_sanitize_image_parts_passes_through_safe_mimes():
    """JPEG/PNG/WebP/GIF image parts must not be touched."""
    import base64

    jpeg_url = f"data:image/jpeg;base64,{base64.b64encode(b'fake').decode()}"
    part = ImageURLPart(image_url=ImageURLPart.ImageURL(url=jpeg_url))
    msg = Message(role="user", content=[TextPart(text="hi"), part])

    out = sanitize_image_parts([msg])

    assert len(out) == 1
    # Same object identity when nothing changed
    assert out[0] is msg


def test_sanitize_image_parts_transcodes_heic_to_jpeg():
    """HEIC data URLs must be transcoded to JPEG before reaching the LLM."""
    import base64

    Image = __import__("PIL.Image", fromlist=["Image"])
    pillow_heif = __import__("pillow_heif")
    pillow_heif.register_heif_opener()

    img = Image.new("RGB", (4, 5), color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="HEIF")
    heic_url = f"data:image/heic;base64,{base64.b64encode(buf.getvalue()).decode()}"
    part = ImageURLPart(image_url=ImageURLPart.ImageURL(url=heic_url, id="x"))
    msg = Message(role="user", content=[part])

    out = sanitize_image_parts([msg])

    assert len(out) == 1
    new_part = out[0].content[0]
    assert isinstance(new_part, ImageURLPart)
    assert new_part.image_url.url.startswith("data:image/jpeg;base64,")
    assert new_part.image_url.id == "x"
    # Original message untouched
    assert msg.content[0].image_url.url.startswith("data:image/heic;base64,")


def test_sanitize_image_parts_replaces_undecodable_with_text():
    """Garbage-bytes HEIC must fall back to a text placeholder, not crash."""
    import base64

    bad_url = f"data:image/heic;base64,{base64.b64encode(b'not really heic').decode()}"
    part = ImageURLPart(image_url=ImageURLPart.ImageURL(url=bad_url))
    msg = Message(role="user", content=[part, TextPart(text="caption")])

    out = sanitize_image_parts([msg])

    assert len(out) == 1
    new_content = out[0].content
    assert isinstance(new_content[0], TextPart)
    assert "unsupported format image/heic" in new_content[0].text
    assert new_content[1] == TextPart(text="caption")


def test_sanitize_image_parts_leaves_remote_urls_alone():
    """Non-data URLs are the provider's problem, not ours."""
    part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/x.heic"))
    msg = Message(role="user", content=[part])

    out = sanitize_image_parts([msg])

    assert out[0] is msg
