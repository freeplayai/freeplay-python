from dataclasses import dataclass
from typing import List, Union, Any, Dict

from pydantic import RootModel

InputValue = Union[str, int, bool, dict[str, Any], list[Any]]
InputVariable = RootModel[Union[Dict[str, "InputVariable"], List["InputVariable"], str, int, bool, float]]
InputVariable.model_rebuild()

InputVariables = Dict[str, InputValue]

PydanticInputVariables = RootModel[Dict[str, InputVariable]]

TestRunInput = Dict[str, InputValue]


@dataclass
class TestRun:
    id: str
    inputs: List[TestRunInput]
