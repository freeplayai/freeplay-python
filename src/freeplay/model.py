from dataclasses import dataclass
from typing import List, Union, Any, Dict, Mapping

from pydantic import RootModel

InputValue = Union[str, int, bool, dict[str, Any], list[Any]]
InputVariable = RootModel[Union[Dict[str, "InputVariable"], List["InputVariable"], str, int, bool, float]]
InputVariable.model_rebuild()

InputVariables = Mapping[str, InputValue]

PydanticInputVariables = RootModel[Dict[str, InputVariable]]

TestRunInput = Mapping[str, InputValue]


@dataclass
class TestRun:
    id: str
    inputs: List[TestRunInput]
