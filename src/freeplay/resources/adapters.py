import copy
from dataclasses import dataclass
from typing import Protocol, Dict, List, Union, Any

from freeplay.errors import FreeplayConfigurationError


@dataclass
class TextContent:
    text: str


@dataclass
class ImageContentUrl:
    url: str


@dataclass
class ImageContentBase64:
    content_type: str
    data: str


class MissingFlavorError(FreeplayConfigurationError):
    def __init__(self, flavor_name: str):
        super().__init__(
            f'Configured flavor ({flavor_name}) not found in SDK. Please update your SDK version or configure '
            'a different model in the Freeplay UI.'
        )


class LLMAdapter(Protocol):
    def to_llm_syntax(self, messages: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
        pass


class PassthroughAdapter(LLMAdapter):
    def to_llm_syntax(self, messages: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
        # We need a deepcopy here to avoid referential equality with the llm_prompt
        return copy.deepcopy(messages)


class AnthropicAdapter(LLMAdapter):
    def to_llm_syntax(self, messages: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
        anthropic_messages = []

        for message in messages:
            if message['role'] == 'system':
                continue
            if "has_media" in message and message["has_media"]:
                anthropic_messages.append({
                    'role': message['role'],
                    'content': [self.__map_content(content) for content in message['content']]
                })
            else:
                anthropic_messages.append(copy.deepcopy(message))

        return anthropic_messages

    @staticmethod
    def __map_content(content: Union[TextContent, ImageContentBase64, ImageContentUrl]) -> Dict[str, Any]:
        if isinstance(content, TextContent):
            return {
                "type": "text",
                "text": content.text
            }
        elif isinstance(content, ImageContentBase64):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.content_type,
                    "data": content.data,
                }
            }
        elif isinstance(content, ImageContentUrl):
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": content.url,
                }
            }
        else:
            raise ValueError(f"Unexpected content type {type(content)}")


class OpenAIAdapter(LLMAdapter):
    def to_llm_syntax(self, messages: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
        openai_messages = []

        for message in messages:
            if "has_media" in message and message["has_media"]:
                openai_messages.append({
                    'role': message['role'],
                    'content': [self.__map_content(content) for content in message['content']]
                })
            else:
                openai_messages.append(copy.deepcopy(message))

        return openai_messages

    @staticmethod
    def __map_content(content: Union[TextContent, ImageContentBase64, ImageContentUrl]) -> Dict[str, Any]:
        if isinstance(content, TextContent):
            return {
                "type": "text",
                "text": content.text
            }
        elif isinstance(content, ImageContentBase64):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content.content_type};base64,{content.data}"
                }
            }
        elif isinstance(content, ImageContentUrl):
            return {
                "type": "image_url",
                "image_url": {
                    "url": content.url
                }
            }
        else:
            raise ValueError(f"Unexpected content type {type(content)}")


class Llama3Adapter(LLMAdapter):
    def to_llm_syntax(self, messages: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
        if len(messages) < 1:
            raise ValueError("Must have at least one message to format")

        formatted = "<|begin_of_text|>"
        for message in messages:
            formatted += f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>"
        formatted += "<|start_header_id|>assistant<|end_header_id|>"

        return formatted


class GeminiAdapter(LLMAdapter):
    def to_llm_syntax(self, messages: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
        if len(messages) < 1:
            raise ValueError("Must have at least one message to format")

        gemini_messages = []

        for message in messages:
            if message['role'] == 'system':
                continue

            if "has_media" in message and message["has_media"]:
                gemini_messages.append({
                    "role": self.__translate_role(message["role"]),
                    "parts": [self.__map_content(content) for content in message['content']]
                })
            else:
                gemini_messages.append({
                    "role": self.__translate_role(message["role"]),
                    "parts": [{"text": message["content"]}]
                })

        return gemini_messages

    @staticmethod
    def __map_content(content: Union[TextContent, ImageContentBase64, ImageContentUrl]) -> Dict[str, Any]:
        if isinstance(content, TextContent):
            return {"text": content.text}
        elif isinstance(content, ImageContentBase64):
            return {
                "inline_data": {
                    "data": content.data,
                    "mime_type": content.content_type,
                }
            }
        elif isinstance(content, ImageContentUrl):
            raise ValueError("Message contains an image URL, but image URLs are not supported by Gemini")
        else:
            raise ValueError(f"Unexpected content type {type(content)}")

    @staticmethod
    def __translate_role(role: str) -> str:
        if role == "user":
            return "user"
        elif role == "assistant":
            return "model"
        else:
            raise ValueError(f"Gemini formatting found unexpected role {role}")


def adaptor_for_flavor(flavor_name: str) -> LLMAdapter:
    if flavor_name in ["baseten_mistral_chat", "mistral_chat", "perplexity_chat"]:
        return PassthroughAdapter()
    elif flavor_name in ["azure_openai_chat", "openai_chat"]:
        return OpenAIAdapter()
    elif flavor_name == "anthropic_chat":
        return AnthropicAdapter()
    elif flavor_name == "llama_3_chat":
        return Llama3Adapter()
    elif flavor_name == "gemini_chat":
        return GeminiAdapter()
    else:
        raise MissingFlavorError(flavor_name)
