import importlib.metadata
import json
import platform
from typing import Dict, Union, Optional, Any

import pystache  # type: ignore

from .errors import FreeplayError, FreeplayConfigurationError
from .model import InputVariables


# Validate that the variables are of the correct type, and do not include functions, dates, classes or None values.
def all_valid(obj: Any) -> bool:
    if isinstance(obj, (int, str, bool, float)):
        return True
    elif isinstance(obj, list):
        return all(all_valid(item) for item in obj)
    elif isinstance(obj, dict):
        return all(isinstance(key, str) and all_valid(value) for key, value in obj.items())
    else:
        return False


class StandardPystache(pystache.Renderer):  # type: ignore

    def __init__(self) -> None:
        super().__init__(escape=lambda s: s)

    def str_coerce(self, val: Any) -> str:
        if isinstance(val, dict) or isinstance(val, list):
            # We hide spacing after punctuation so that the templating is the same across all SDKs.
            return json.dumps(val, separators=(',', ':'))
        return str(val)


def bind_template_variables(template: str, variables: InputVariables) -> str:
    if not all_valid(variables):
        raise FreeplayError(
            'Variables must be a string, number, bool, or a possibly nested'
            ' list or dict of strings, numbers and booleans.'
        )

    # When rendering mustache, do not escape HTML special characters.
    rendered: str = StandardPystache().render(template, variables)
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


# Recursively convert Pydantic models, lists, and dicts to dict compatible format -- used to allow us to accept
# provider message shapes (usually generated types) or the default {'content': ..., 'role': ...} shape.
def convert_provider_message_to_dict(obj: Any) -> Any:
    if hasattr(obj, 'model_dump'):
        # Pydantic v2
        return obj.model_dump(mode='json')
    elif hasattr(obj, 'dict'):
        # Pydantic v1
        return obj.dict(encode_json=True)
    elif isinstance(obj, dict):
        return {k: convert_provider_message_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_provider_message_to_dict(item) for item in obj]
    return obj
