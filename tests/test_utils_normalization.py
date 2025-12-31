"""Tests for message normalization and serialization functions in utils.py"""

import unittest
from typing import Any, List
from freeplay.utils import (
    convert_api_message_content_to_sdk_message_content,
    convert_api_message_to_sdk_message,
    convert_sdk_messages_to_api_messages,
    convert_provider_message_to_dict,
)
from freeplay.model import (
    TextBlock,
    MediaReferenceBlock,
    MediaReference,
    ToolResultBlock,
    ToolCallBlock,
    UserMessage,
    AssistantMessage,
    SystemMessage,
)


class TestMessageNormalization(unittest.TestCase):
    """Test suite for message normalization functions"""

    def test_normalize_message_content_string(self) -> None:
        """Test that string content is returned unchanged"""
        content = "Hello, world!"
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertEqual(result, content)

    def test_normalize_message_content_text_block(self) -> None:
        """Test normalizing text blocks"""
        content = [
            {"type": "text", "text": "First message"},
            {"type": "text", "text": "Second message"},
        ]
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], TextBlock)
        self.assertEqual(result[0].text, "First message")  # pyright: ignore
        self.assertIsInstance(result[1], TextBlock)
        self.assertEqual(result[1].text, "Second message")  # pyright: ignore

    def test_normalize_message_content_tool_blocks(self) -> None:
        """Test normalizing tool result and tool call blocks"""
        content = [
            {
                "type": "tool_result",
                "tool_call_id": "call_123",
                "content": "Tool output",
            },
            {
                "type": "tool_call",
                "id": "call_456",
                "name": "get_weather",
                "arguments": {"location": "New York"},
            },
        ]
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        # Check tool result block
        self.assertIsInstance(result[0], ToolResultBlock)
        self.assertEqual(result[0].tool_call_id, "call_123")  # pyright: ignore
        self.assertEqual(result[0].content, "Tool output")  # pyright: ignore

        # Check tool call block
        self.assertIsInstance(result[1], ToolCallBlock)
        self.assertEqual(result[1].id, "call_456")  # pyright: ignore
        self.assertEqual(result[1].name, "get_weather")  # pyright: ignore
        self.assertEqual(result[1].arguments, {"location": "New York"})  # pyright: ignore

    def test_normalize_message_content_downloaded_media(self) -> None:
        """Test normalizing downloaded media references - kind should be preserved"""
        content = [
            {
                "type": "image",
                "media_reference": {
                    "kind": "downloaded_file",
                    "base64Data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                    "contentType": "image/png",
                    "media_type": "image",
                    "filename": "test.png",
                },
            }
        ]
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], MediaReferenceBlock)
        self.assertEqual(
            result[0].media_reference.kind,  # pyright: ignore
            "downloaded_file",  # pyright: ignore
        )  # Note: kind is preserved as downloaded_file
        self.assertEqual(result[0].media_reference.contentType, "image/png")  # pyright: ignore
        self.assertEqual(result[0].media_reference.media_type, "image")  # pyright: ignore
        self.assertEqual(result[0].media_reference.filename, "test.png")  # pyright: ignore

    def test_normalize_message_content_uploaded_media(self) -> None:
        """Test normalizing uploaded media references"""
        content = [
            {
                "type": "audio",
                "media_reference": {
                    "kind": "uploaded_file",
                    "base64Data": "SGVsbG8gV29ybGQ=",
                    "contentType": "audio/mp3",
                    "media_type": "audio",
                    "filename": "audio.mp3",
                },
            }
        ]
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], MediaReferenceBlock)
        self.assertEqual(result[0].media_reference.kind, "uploaded_file")  # pyright: ignore
        self.assertEqual(result[0].media_reference.contentType, "audio/mp3")  # pyright: ignore
        self.assertEqual(result[0].media_reference.media_type, "audio")  # pyright: ignore
        self.assertEqual(result[0].media_reference.base64Data, "SGVsbG8gV29ybGQ=")  # pyright: ignore

    def test_normalize_message_content_mixed_blocks(self) -> None:
        """Test normalizing mixed content blocks"""
        content = [
            {"type": "text", "text": "Here's an image:"},
            {
                "type": "image",
                "media_reference": {
                    "kind": "uploaded_file",
                    "base64Data": "image_data",
                    "contentType": "image/jpeg",
                    "media_type": "image",
                },
            },
            {"type": "text", "text": "What do you see?"},
        ]
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], TextBlock)
        self.assertIsInstance(result[1], MediaReferenceBlock)
        self.assertIsInstance(result[2], TextBlock)

    def test_normalize_message_content_invalid_blocks(self) -> None:
        """Test that invalid blocks are skipped"""
        content = [
            {"type": "text", "text": "Valid text"},
            {"invalid": "block"},  # Missing type
            "not a dict",  # Not a dict
            {"type": "unknown_type", "data": "something"},  # Unknown type
            {"type": "text"},  # Missing required field
        ]
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertIsInstance(result, list)
        # Only the first valid text block should be included
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextBlock)
        self.assertEqual(result[0].text, "Valid text")  # pyright: ignore

    def test_normalize_message_content_empty_list(self) -> None:
        """Test that empty list returns original content"""
        content: List[Any] = []
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertEqual(result, content)

    def test_normalize_message_content_non_list_non_string(self) -> None:
        """Test that non-list, non-string content is returned unchanged"""
        content = {"not": "a list"}
        result = convert_api_message_content_to_sdk_message_content(content)
        self.assertEqual(result, content)

    def test_normalize_message(self) -> None:
        """Test normalizing a complete message"""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image",
                    "media_reference": {
                        "kind": "downloaded_file",
                        "base64Data": "data",
                        "contentType": "image/png",
                        "media_type": "image",
                    },
                },
            ],
        }
        result = convert_api_message_to_sdk_message(message)
        self.assertEqual(result["role"], "user")
        self.assertIsInstance(result["content"], list)
        self.assertEqual(len(result["content"]), 2)
        self.assertIsInstance(result["content"][0], TextBlock)
        self.assertIsInstance(result["content"][1], MediaReferenceBlock)

    def test_normalize_message_non_dict(self) -> None:
        """Test that non-dict messages are returned unchanged"""
        message = "not a dict"
        result = convert_api_message_to_sdk_message(message)
        self.assertEqual(result, message)

    def test_convert_sdk_messages_to_api_messages_text_block(self) -> None:
        """Test serializing messages with TextBlock objects"""
        message = UserMessage(
            content=[
                TextBlock(text="Hello"),
                TextBlock(text="World"),
            ]
        )
        result = convert_sdk_messages_to_api_messages(message)
        self.assertEqual(result["role"], "user")
        self.assertEqual(
            result["content"],
            [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        )

    def test_convert_sdk_messages_to_api_messages_media_block(self) -> None:
        """Test serializing messages with MediaReferenceBlock objects"""
        message = AssistantMessage(
            content=[
                TextBlock(text="Here's an image:"),
                MediaReferenceBlock(
                    media_reference=MediaReference(
                        base64Data="image_data",
                        contentType="image/png",
                        media_type="image",
                        filename="test.png",
                        kind="uploaded_file",
                    )
                ),
            ]
        )
        result = convert_sdk_messages_to_api_messages(message)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(
            result["content"][0], {"type": "text", "text": "Here's an image:"}
        )
        self.assertEqual(
            result["content"][1],
            {
                "type": "image",
                "media_reference": {
                    "base64Data": "image_data",
                    "contentType": "image/png",
                    "media_type": "image",
                    "filename": "test.png",
                    "kind": "uploaded_file",
                },
            },
        )

    def test_convert_sdk_messages_to_api_messages_tool_blocks(self) -> None:
        """Test serializing messages with tool blocks"""
        message = AssistantMessage(
            content=[
                ToolCallBlock(
                    id="call_123",
                    name="get_weather",
                    arguments={"location": "NYC"},
                ),
                ToolResultBlock(
                    tool_call_id="call_123",
                    content="Sunny, 72°F",
                ),
            ]
        )
        result = convert_sdk_messages_to_api_messages(message)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(
            result["content"],
            [
                {
                    "type": "tool_call",
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": {"location": "NYC"},
                },
                {
                    "type": "tool_result",
                    "tool_call_id": "call_123",
                    "content": "Sunny, 72°F",
                },
            ],
        )

    def test_convert_sdk_messages_to_api_messages_string_content(self) -> None:
        """Test serializing messages with string content"""
        message = SystemMessage(content="You are a helpful assistant.")
        result = convert_sdk_messages_to_api_messages(message)
        self.assertEqual(result["role"], "system")
        self.assertEqual(result["content"], "You are a helpful assistant.")

    def test_convert_sdk_messages_to_api_messages_dict_input(self) -> None:
        """Test serializing dict messages with ContentBlock objects"""
        message = {
            "role": "user",
            "content": [
                TextBlock(text="Hello"),
                MediaReferenceBlock(
                    media_reference=MediaReference(
                        base64Data="data",
                        contentType="image/png",
                        media_type="image",
                        kind="uploaded_file",
                    )
                ),
            ],
        }
        result = convert_sdk_messages_to_api_messages(message)
        self.assertEqual(result["role"], "user")
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][1]["type"], "image")

    def test_convert_sdk_messages_to_api_messages_dict_passthrough(self) -> None:
        """Test that dict messages without ContentBlock objects pass through"""
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Already serialized"},
            ],
        }
        result = convert_sdk_messages_to_api_messages(message)
        self.assertEqual(result, message)

    def test_convert_provider_message_converts_downloaded_to_uploaded(self) -> None:
        """Test that convert_provider_message_to_dict converts downloaded_file to uploaded_file"""
        # Test with dict containing media reference
        message_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's an image"},
                {
                    "type": "image",
                    "media_reference": {
                        "kind": "downloaded_file",
                        "base64Data": "test_data",
                        "contentType": "image/jpeg",
                        "media_type": "image",
                        "filename": "photo.jpg",
                    },
                },
            ],
        }
        result = convert_provider_message_to_dict(message_dict)
        self.assertEqual(result["role"], "user")
        self.assertEqual(
            result["content"][1]["media_reference"]["kind"], "uploaded_file"
        )  # Converted!
        self.assertEqual(
            result["content"][1]["media_reference"]["base64Data"], "test_data"
        )

    def test_convert_provider_message_converts_dataclass_media_reference(self) -> None:
        """Test that convert_provider_message_to_dict converts MediaReference dataclass"""
        message = UserMessage(
            content=[
                TextBlock(text="Image"),
                MediaReferenceBlock(
                    media_reference=MediaReference(
                        base64Data="data",
                        contentType="image/png",
                        media_type="image",
                        kind="downloaded_file",
                    )
                ),
            ]
        )
        result = convert_provider_message_to_dict(message)
        # The MediaReference dataclass should be converted to dict with uploaded_file
        self.assertEqual(
            result["content"][1]["media_reference"]["kind"], "uploaded_file"
        )


if __name__ == "__main__":
    unittest.main()
