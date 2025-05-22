import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Union

from freeplay.errors import FreeplayConfigurationError
from freeplay.support import MediaType


@dataclass
class TextContent:
    text: str


@dataclass
class MediaContentUrl:
    slot_name: str
    type: MediaType
    url: str


@dataclass
class MediaContentBase64:
    slot_name: str
    type: MediaType
    content_type: str
    data: str


class MissingFlavorError(FreeplayConfigurationError):
    def __init__(self, flavor_name: str):
        super().__init__(
            f'Configured flavor ({flavor_name}) not found in SDK. Please update your SDK version or configure '
            'a different model in the Freeplay UI.'
        )


class LLMAdapter(Protocol):
    # This method must handle BOTH prompt template messages and provider specific messages.
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
    def __map_content(content: Union[TextContent, MediaContentBase64, MediaContentUrl]) -> Dict[str, Any]:
        if isinstance(content, TextContent):
            return {
                "type": "text",
                "text": content.text
            }
        if content.type == "audio" or content.type == "video":
            raise ValueError("Anthropic does not support audio or video content")

        media_type = "image" if content.type == "image" else "document"
        if isinstance(content, MediaContentBase64):
            return {
                "type": media_type,
                "source": {
                    "type": "base64",
                    "media_type": content.content_type,
                    "data": content.data,
                }
            }
        elif isinstance(content, MediaContentUrl):
            return {
                "type": media_type,
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
    def __map_content(content: Union[TextContent, MediaContentBase64, MediaContentUrl]) -> Dict[str, Any]:
        if isinstance(content, TextContent):
            return {
                "type": "text",
                "text": content.text
            }
        elif isinstance(content, MediaContentBase64):
            return OpenAIAdapter.__format_base64_content(content)
        elif isinstance(content, MediaContentUrl):
            if content.type != "image":
                raise ValueError("Message contains a non-image URL, but OpenAI only supports image URLs.")

            return {
                "type": "image_url",
                "image_url": {
                    "url": content.url
                }
            }
        else:
            raise ValueError(f"Unexpected content type {type(content)}")

    @staticmethod
    def __format_base64_content(content: MediaContentBase64) -> Dict[str, Any]:
        if content.type == "audio":
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": content.data,
                    "format": content.content_type.split("/")[-1].replace("mpeg", "mp3")
                }
            }
        elif content.type == "file":
            return {
                "type": "file",
                "file": {
                    "filename": f"{content.slot_name}.{content.content_type.split('/')[-1]}",
                    "file_data": f"data:{content.content_type};base64,{content.data}"
                }
            }
        else:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content.content_type};base64,{content.data}"
                }
            }


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
            elif "content" in message:
                gemini_messages.append({
                    "role": self.__translate_role(message["role"]),
                    "parts": [{"text": message["content"]}]
                })
            else:
                gemini_messages.append(copy.deepcopy(message))

        return gemini_messages

    @staticmethod
    def __map_content(content: Union[TextContent, MediaContentBase64, MediaContentUrl]) -> Dict[str, Any]:
        if isinstance(content, TextContent):
            return {"text": content.text}
        elif isinstance(content, MediaContentBase64):
            return {
                "inline_data": {
                    "data": content.data,
                    "mime_type": content.content_type,
                }
            }
        elif isinstance(content, MediaContentUrl):
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


class BedrockConverseAdapter(LLMAdapter):
    def to_llm_syntax(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converse_messages = []
        for message in messages:
            if message["role"] == "system":
                continue
            if "has_media" in message and message["has_media"]:
                raise ValueError("Bedrock Converse does not support media content yet")

            role = message["role"]
            content = message["content"]
            if role not in ["user", "assistant"]:
                raise ValueError(f"Unexpected role for Bedrock Converse flavor: {role}")
            if isinstance(content, str):
                content = [{"text": content}]
            converse_messages.append({"role": role, "content": content})
        return converse_messages


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
    elif flavor_name == "amazon_bedrock_converse":
        return BedrockConverseAdapter()
    else:
        raise MissingFlavorError(flavor_name)
