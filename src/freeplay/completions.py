from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from openai.types.chat.chat_completion_chunk import ChoiceDeltaFunctionCall

from .llm_parameters import LLMParameters


class ChatMessage(TypedDict):
    role: str
    content: str


class OpenAIFunctionCall(TypedDict):
    name: str
    arguments: str


@dataclass
class CompletionResponse:
    content: str
    is_complete: bool
    openai_function_call: Optional[OpenAIFunctionCall] = None


@dataclass
class ChatCompletionResponse:
    content: str
    is_complete: bool
    message_history: List[ChatMessage]


@dataclass
class PromptTemplateWithMetadata:
    prompt_template_id: str
    prompt_template_version_id: str

    name: str
    content: str
    flavor_name: Optional[str]
    params: Optional[Dict[str, Any]]

    def get_params(self) -> LLMParameters:
        return LLMParameters.empty() if self.params is None else LLMParameters(self.params)


@dataclass
class PromptTemplates:
    templates: List[PromptTemplateWithMetadata]


@dataclass
class CompletionChunk:
    text: str
    is_complete: bool
    openai_function_call: Optional[ChoiceDeltaFunctionCall] = None
