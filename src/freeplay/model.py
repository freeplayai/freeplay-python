from dataclasses import dataclass
from typing import List, Union, Any, Dict, Mapping, TypedDict

InputValue = Union[str, int, bool, Dict[str, Any], List[Any]]
InputVariables = Mapping[str, InputValue]
TestRunInput = Mapping[str, InputValue]


@dataclass
class TestRun:
    id: str
    inputs: List[TestRunInput]


class OpenAIFunctionCall(TypedDict):
    name: str
    arguments: str
