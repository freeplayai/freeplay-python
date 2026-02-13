import unittest
from typing import Any, List, Dict

from freeplay.resources.adapters import (
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    BedrockConverseAdapter,
    MissingFlavorError,
    TextContent,
    MediaContentUrl,
    MediaContentBase64,
    adaptor_for_flavor,
)


class TestAdapters(unittest.TestCase):
    def test_openai(self) -> None:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "has_media": True,
                "content": [
                    TextContent("Take a look at these images!"),
                    MediaContentUrl(
                        type="image",
                        url="https://localhost/image.png",
                        slot_name="image1",
                    ),
                    MediaContentBase64(
                        type="image",
                        content_type="image/png",
                        data="some-data",
                        slot_name="image2",
                    ),
                ],
            },
        ]

        formatted = OpenAIAdapter().to_llm_syntax(messages)

        self.assertEqual(
            formatted,
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "How can I help you?"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Take a look at these images!"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://localhost/image.png"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,some-data"},
                        },
                    ],
                },
            ],
        )

    def test_openai_audio(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "has_media": True,
                "content": [
                    MediaContentBase64(
                        type="audio",
                        content_type="audio/mpeg",
                        data="some-data",
                        slot_name="audio1",
                    ),
                ],
            }
        ]

        formatted = OpenAIAdapter().to_llm_syntax(messages)

        self.assertEqual(
            formatted,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "some-data", "format": "mp3"},
                        }
                    ],
                }
            ],
        )

    def test_openai_pdf(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "has_media": True,
                "content": [
                    MediaContentBase64(
                        type="file",
                        content_type="application/pdf",
                        data="some-data",
                        slot_name="document1",
                    ),
                ],
            }
        ]

        formatted = OpenAIAdapter().to_llm_syntax(messages)

        self.assertEqual(
            formatted,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "file_data": "data:application/pdf;base64,some-data",
                                "filename": "document1.pdf",
                            },
                        }
                    ],
                }
            ],
        )

    def test_anthropic(self) -> None:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "has_media": True,
                "content": [
                    TextContent("Take a look at these images!"),
                    MediaContentUrl(
                        type="image",
                        url="https://localhost/image.png",
                        slot_name="image1",
                    ),
                    MediaContentBase64(
                        type="image",
                        content_type="image/png",
                        data="some-data",
                        slot_name="image2",
                    ),
                ],
            },
        ]

        formatted = AnthropicAdapter().to_llm_syntax(messages)

        self.assertEqual(
            formatted,
            [
                {"role": "assistant", "content": "How can I help you?"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Take a look at these images!"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://localhost/image.png",
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "data": "some-data",
                                "media_type": "image/png",
                            },
                        },
                    ],
                },
            ],
        )

    def test_anthropic_pdf(self) -> None:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "has_media": True,
                "content": [
                    MediaContentUrl(
                        type="file",
                        url="https://localhost/file.pdf",
                        slot_name="document1",
                    ),
                    MediaContentBase64(
                        type="file",
                        content_type="application/pdf",
                        data="some-data",
                        slot_name="document2",
                    ),
                ],
            }
        ]

        formatted = AnthropicAdapter().to_llm_syntax(messages)

        self.assertEqual(
            formatted,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "url",
                                "url": "https://localhost/file.pdf",
                            },
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "data": "some-data",
                                "media_type": "application/pdf",
                            },
                        },
                    ],
                }
            ],
        )

    def test_gemini(self) -> None:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "has_media": True,
                "content": [
                    TextContent("Take a look at these images!"),
                    MediaContentBase64(
                        type="image",
                        content_type="image/png",
                        data="some-data",
                        slot_name="image1",
                    ),
                ],
            },
        ]

        formatted = GeminiAdapter().to_llm_syntax(messages)

        self.assertEqual(
            formatted,
            [
                {"role": "model", "parts": [{"text": "How can I help you?"}]},
                {
                    "role": "user",
                    "parts": [
                        {"text": "Take a look at these images!"},
                        {
                            "inline_data": {
                                "data": "some-data",
                                "mime_type": "image/png",
                            }
                        },
                    ],
                },
            ],
        )

    def test_bedrock_converse(self) -> None:
        import base64

        # Input has base64-encoded data (as would come from Freeplay)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "has_media": True,
                "content": [
                    TextContent("Take a look at this image and document!"),
                    MediaContentBase64(
                        type="image",
                        content_type="image/png",
                        data=base64.b64encode(b"image-data").decode("utf-8"),
                        slot_name="image1",
                    ),
                    MediaContentBase64(
                        type="file",
                        content_type="application/pdf",
                        data=base64.b64encode(b"pdf-data").decode("utf-8"),
                        slot_name="document1",
                    ),
                ],
            },
        ]

        formatted = BedrockConverseAdapter().to_llm_syntax(messages)

        self.assertEqual(len(formatted), 2)  # System message should be filtered out
        self.assertEqual(formatted[0]["role"], "assistant")
        self.assertEqual(formatted[0]["content"], [{"text": "How can I help you?"}])

        self.assertEqual(formatted[1]["role"], "user")
        self.assertEqual(len(formatted[1]["content"]), 3)

        # Check text content
        self.assertEqual(
            formatted[1]["content"][0],
            {"text": "Take a look at this image and document!"},
        )

        # Check image content - adapter converts to actual bytes for Bedrock
        self.assertEqual(
            formatted[1]["content"][1],
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": b"image-data"},
                }
            },
        )

        # Check document content - adapter converts to actual bytes for Bedrock
        self.assertEqual(
            formatted[1]["content"][2],
            {
                "document": {
                    "format": "pdf",
                    "name": "document1",
                    "source": {"bytes": b"pdf-data"},
                }
            },
        )

    # ------------------------------------------------------------------
    # Gemini parts passthrough (history with function calls / responses)
    # ------------------------------------------------------------------

    def test_gemini_parts_passthrough(self) -> None:
        """Messages with 'parts' key are passed through without re-wrapping."""
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [{"text": "What is the weather?"}],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "Seattle"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temperature": "72°F"},
                        }
                    }
                ],
            },
            {
                "role": "model",
                "parts": [{"text": "It's 72°F in Seattle."}],
            },
        ]

        result = GeminiAdapter().to_llm_syntax(messages)
        assert isinstance(result, list)

        self.assertEqual(len(result), 4)
        # All messages should be preserved as-is
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["parts"], [{"text": "What is the weather?"}])

        self.assertEqual(result[1]["role"], "model")
        self.assertIn("functionCall", result[1]["parts"][0])

        self.assertEqual(result[2]["role"], "user")
        self.assertIn("functionResponse", result[2]["parts"][0])

        self.assertEqual(result[3]["role"], "model")
        self.assertEqual(result[3]["parts"], [{"text": "It's 72°F in Seattle."}])

    def test_gemini_parts_passthrough_translates_assistant_to_model(self) -> None:
        """Parts messages with role 'assistant' are translated to 'model'."""
        messages: List[Dict[str, Any]] = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "test"},
                        }
                    }
                ],
            },
        ]

        result = GeminiAdapter().to_llm_syntax(messages)
        assert isinstance(result, list)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "model")
        self.assertIn("functionCall", result[0]["parts"][0])

    def test_gemini_parts_passthrough_does_not_mutate_original(self) -> None:
        """Parts passthrough deep-copies, so original messages are not mutated."""
        original: Dict[str, Any] = {
            "role": "assistant",
            "parts": [{"text": "original"}],
        }
        messages: List[Dict[str, Any]] = [original]

        result = GeminiAdapter().to_llm_syntax(messages)
        assert isinstance(result, list)

        self.assertEqual(result[0]["role"], "model")
        self.assertEqual(original["role"], "assistant")

    def test_gemini_mixed_content_and_parts(self) -> None:
        """Handles a mix of standard content and pre-formatted parts messages."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Hello"},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "greet", "args": {}}}
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "greet",
                            "response": {"greeting": "Hi!"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Done"},
        ]

        result = GeminiAdapter().to_llm_syntax(messages)
        assert isinstance(result, list)

        self.assertEqual(len(result), 4)  # system is skipped
        self.assertEqual(result[0], {"role": "user", "parts": [{"text": "Hello"}]})
        self.assertEqual(result[1]["role"], "model")
        self.assertIn("functionCall", result[1]["parts"][0])
        self.assertEqual(result[2]["role"], "user")
        self.assertIn("functionResponse", result[2]["parts"][0])
        self.assertEqual(result[3], {"role": "model", "parts": [{"text": "Done"}]})

    def test_gemini_media_in_history_parts(self) -> None:
        """History messages with inlineData parts are passed through correctly."""
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    {"text": "What's in this image?"},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUg==",
                        }
                    },
                ],
            },
            {"role": "model", "parts": [{"text": "I see a cat."}]},
            {"role": "user", "content": "Can you describe it in more detail?"},
        ]

        result = GeminiAdapter().to_llm_syntax(messages)
        assert isinstance(result, list)

        self.assertEqual(len(result), 3)

        # First message: media parts passed through unchanged
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(len(result[0]["parts"]), 2)
        self.assertEqual(result[0]["parts"][0], {"text": "What's in this image?"})
        self.assertIn("inlineData", result[0]["parts"][1])
        self.assertEqual(result[0]["parts"][1]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(
            result[0]["parts"][1]["inlineData"]["data"], "iVBORw0KGgoAAAANSUhEUg=="
        )

        # Second message: text parts passed through
        self.assertEqual(result[1]["role"], "model")
        self.assertEqual(result[1]["parts"], [{"text": "I see a cat."}])

        # Third message: standard content converted to Gemini format
        self.assertEqual(
            result[2],
            {"role": "user", "parts": [{"text": "Can you describe it in more detail?"}]},
        )

    # ------------------------------------------------------------------
    # adaptor_for_flavor() registry tests
    # ------------------------------------------------------------------

    def test_adaptor_for_gemini_chat(self) -> None:
        adapter = adaptor_for_flavor("gemini_chat")
        self.assertIsInstance(adapter, GeminiAdapter)

    def test_adaptor_for_gemini_api_chat(self) -> None:
        adapter = adaptor_for_flavor("gemini_api_chat")
        self.assertIsInstance(adapter, GeminiAdapter)

    def test_adaptor_for_unknown_flavor_raises(self) -> None:
        with self.assertRaises(MissingFlavorError):
            adaptor_for_flavor("nonexistent_flavor")
