import unittest
from typing import Any, List, Dict

from freeplay.resources.adapters import OpenAIAdapter, AnthropicAdapter, GeminiAdapter, TextContent, ImageContentUrl, \
    ImageContentBase64


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
                    ImageContentUrl("https://localhost/image.png"),
                    ImageContentBase64("image/png", "some-data"),
                ],
            }]

        formatted = OpenAIAdapter().to_llm_syntax(messages)

        self.assertEqual(formatted, [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Take a look at these images!"},
                    {"type": "image_url", "image_url": {"url": "https://localhost/image.png"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,some-data"}},
                ]
            }
        ])

    def test_anthropic(self) -> None:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "has_media": True,
                "content": [
                    TextContent("Take a look at these images!"),
                    ImageContentUrl("https://localhost/image.png"),
                    ImageContentBase64("image/png", "some-data"),
                ],
            }]

        formatted = AnthropicAdapter().to_llm_syntax(messages)

        self.assertEqual(formatted, [
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Take a look at these images!"},
                    {"type": "image", "source": {"type": "url", "url": "https://localhost/image.png"}},
                    {"type": "image", "source": {"type": "base64", "data": "some-data", "media_type": "image/png"}},
                ]
            }
        ])

    def test_gemini(self) -> None:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
            {
                "role": "user",
                "has_media": True,
                "content": [
                    TextContent("Take a look at these images!"),
                    ImageContentBase64("image/png", "some-data"),
                ],
            }]

        formatted = GeminiAdapter().to_llm_syntax(messages)

        self.assertEqual(formatted, [
            {"role": "model", "parts": [{"text": "How can I help you?"}]},
            {
                "role": "user",
                "parts": [
                    {"text": "Take a look at these images!"},
                    {"inline_data": {"data": "some-data", "mime_type": "image/png"}},
                ]
            }
        ])
