from typing import Any, Dict, List, Union

import pystache  # type: ignore
from pydantic import RootModel, ValidationError

from .errors import FreeplayError

InputVariable = RootModel[Union[Dict[str, "InputVariable"], List["InputVariable"], str, int, bool, float]]
InputVariable.model_rebuild()
InputVariables = Dict[str, Union[str, int, bool, Dict[str, Any], List[Any]]]
PydanticInputVariables = RootModel[Dict[str, InputVariable]]


def bind_template_variables(template: str, variables: Dict[str, str]) -> str:
    # Validate that the variables are of the correct type, and do not include functions or None values.
    try:
        PydanticInputVariables.model_validate(variables)
    except ValidationError as err:
        raise FreeplayError(
            'Variables must be a string, number, bool, or a possibly nested'
            ' list or dict of strings, numbers and booleans.'
        )

    # When rendering mustache, do not escape HTML special characters.
    rendered: str = pystache.Renderer(escape=lambda s: s).render(template, variables)
    return rendered
