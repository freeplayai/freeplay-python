from dataclasses import dataclass
from typing import List, Union, Any, Dict, Mapping, TypedDict, Literal

InputValue = Union[str, int, bool, float, Dict[str, Any], List[Any]]
InputVariables = Mapping[str, InputValue]
TestRunInput = Mapping[str, InputValue]
FeedbackValue = Union[bool, str, int, float]


@dataclass
class TestRun:
    id: str
    inputs: List[TestRunInput]


class OpenAIFunctionCall(TypedDict):
    name: str
    arguments: str


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolResultBlock:
    # AKA tool_use_id -- the ID of the tool call that this message is responding to.
    tool_call_id: str
    content: Union[str, List[TextBlock]]
    type: Literal["tool_result"] = "tool_result"


@dataclass
class ToolCallBlock:
    id: str
    name: str
    arguments: Any
    type: Literal["tool_call"] = "tool_call"


ContentBlock = Union[TextBlock, ToolResultBlock, ToolCallBlock]


@dataclass
class UserMessage:
    content: Union[str, List[ContentBlock]]
    role: Literal["user"] = "user"


@dataclass
class SystemMessage:
    content: str
    role: Literal["system"] = "system"


@dataclass
class AssistantMessage:
    content: Union[str, List[ContentBlock]]
    role: Literal["assistant"] = "assistant"

# Largely used for history in dataset test cases presently
NormalizedMessage = Union[UserMessage, SystemMessage, AssistantMessage]