import unittest
from typing import Any, List, Dict

from freeplay.resources.adapters import (
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    TextContent,
    MediaContentUrl,
    MediaContentBase64,
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
