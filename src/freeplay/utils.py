from typing import Dict, Union, Optional
import importlib.metadata
import platform

import pystache  # type: ignore
from pydantic import ValidationError

from .errors import FreeplayError, FreeplayConfigurationError
from .model import PydanticInputVariables, InputVariables


def bind_template_variables(template: str, variables: InputVariables) -> str:
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


def check_all_values_string_or_number(metadata: Optional[Dict[str, Union[str, int, float]]]) -> None:
    if metadata:
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float)):
                raise FreeplayConfigurationError(f"Invalid value for key {key}: Value must be a string or number.")


def build_request_header(api_key: str) -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {api_key}',
        'User-Agent': get_user_agent()
    }


def get_user_agent() -> str:
    sdk_name = 'Freeplay'
    sdk_version = importlib.metadata.version('Freeplay')
    language = 'Python'
    language_version = platform.python_version()
    os_name = platform.system()
    os_version = platform.release()

    # Output format
    # Freeplay/0.2.30 (Python/3.11.4; Darwin/23.2.0)
    return f"{sdk_name}/{sdk_version} ({language}/{language_version}; {os_name}/{os_version})"
