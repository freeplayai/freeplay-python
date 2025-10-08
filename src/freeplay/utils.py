import importlib.metadata
import json
import base64
import platform
from dataclasses import asdict, is_dataclass
from typing import Dict, Union, Optional, Any, List, Tuple

import pystache  # type: ignore

from .errors import FreeplayError, FreeplayConfigurationError
from .model import InputVariables


# Validate that the variables are of the correct type, and do not include functions, dates, classes or None values.
def all_valid(obj: Any) -> bool:
    if isinstance(obj, (int, str, bool, float)):
        return True
    elif isinstance(obj, list):
        items: list[Any] = obj  # pyright: ignore[reportUnknownVariableType]
        return all(all_valid(item) for item in items)
    elif isinstance(obj, dict):
        dict_obj: dict[Any, Any] = obj  # pyright: ignore[reportUnknownVariableType]
        return all(
            isinstance(key, str) and all_valid(value) for key, value in dict_obj.items()
        )
    else:
        return False


class StandardPystache(pystache.Renderer):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__(escape=lambda s: s)  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]

    def str_coerce(self, val: Any) -> str:
        if isinstance(val, dict) or isinstance(val, list):
            # We hide spacing after punctuation so that the templating is the same across all SDKs.
            return json.dumps(val, separators=(",", ":"))
        return str(val)


def bind_template_variables(template: str, variables: InputVariables) -> str:
    if not all_valid(variables):
        raise FreeplayError(
            "Variables must be a string, number, bool, or a possibly nested"
            " list or dict of strings, numbers and booleans."
        )

    # When rendering mustache, do not escape HTML special characters.
    rendered = StandardPystache().render(template, variables)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return str(
        rendered  # pyright: ignore[reportUnknownArgumentType]
    )  # Ensure it's a string


def check_all_values_string_or_number(
    metadata: Optional[Dict[str, Union[str, int, float]]],
) -> None:
    if metadata:
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise FreeplayConfigurationError(
                    f"Invalid value for key {key}: Value must be a string or number."
                )


def build_request_header(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "User-Agent": get_user_agent()}


def get_user_agent() -> str:
    sdk_name = "Freeplay"
    sdk_version = importlib.metadata.version("Freeplay")
    language = "Python"
    language_version = platform.python_version()
    os_name = platform.system()
    os_version = platform.release()

    # Output format
    # Freeplay/0.2.30 (Python/3.11.4; Darwin/23.2.0)
    return f"{sdk_name}/{sdk_version} ({language}/{language_version}; {os_name}/{os_version})"


def bytes_as_str_factory(field_list: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Custom dict factory to convert bytes to base64 strings for dataclasses.
    Used with asdict() to handle Bedrock and other providers that use byte strings."""
    result: Dict[str, Any] = {}
    for key, value in field_list:
        if isinstance(value, bytes):
            result[key] = base64.b64encode(value).decode("utf-8")
        else:
            result[key] = value
    return result


# Recursively convert Pydantic models, lists, and dicts to dict compatible format -- used to allow us to accept
# provider message shapes (usually generated types) or the default {'content': ..., 'role': ...} shape.
def convert_provider_message_to_dict(obj: Any) -> Any:
    """
    Convert provider message objects to dictionaries.
    For Vertex AI objects, automatically converts to camelCase.
    Handles bytes objects by converting them to base64 strings.
    """

    # List of possible raw attribute names in Vertex AI objects
    vertex_raw_attrs = [
        "_raw_content",  # For Content objects
        "_raw_tool",  # For Tool objects
        "_raw_message",  # For message objects
        "_raw_candidate",  # For Candidate objects
        "_raw_response",  # For response objects
        "_raw_function_declaration",  # For FunctionDeclaration
        "_raw_generation_config",  # For GenerationConfig
        "_pb",  # Generic protobuf attribute
    ]

    # Check for Vertex AI objects with raw protobuf attributes
    for attr_name in vertex_raw_attrs:
        if hasattr(obj, attr_name):
            raw_obj = getattr(obj, attr_name)
            if raw_obj is not None:
                try:
                    # Use the metaclass to_dict with camelCase conversion
                    # pyright: ignore[reportUnknownMemberType]
                    return type(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                        raw_obj
                    ).to_dict(  # pyright: ignore[reportUnknownMemberType]
                        raw_obj,
                        preserving_proto_field_name=False,  # camelCase
                        use_integers_for_enums=False,  # Keep as strings (we'll lowercase them)
                        including_default_value_fields=False,  # Exclude defaults
                    )
                except:  # noqa: E722
                    # If we can't convert, continue to the next attribute
                    pass

    # For non-Vertex AI objects, use their standard to_dict methods
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        # Regular to_dict (for Vertex AI wrappers without _raw_* attributes)
        return obj.to_dict()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    elif hasattr(obj, "model_dump"):
        # Pydantic v2
        return obj.model_dump(mode="json")
    elif hasattr(obj, "dict"):
        # Pydantic v1
        return obj.dict(encode_json=True)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    elif isinstance(obj, dict):
        # Handle dictionaries recursively
        dict_obj: Dict[Any, Any] = obj  # pyright: ignore [reportUnknownVariableType]
        return {k: convert_provider_message_to_dict(v) for k, v in dict_obj.items()}
    elif isinstance(obj, list):
        # Handle lists recursively
        list_obj: List[Any] = obj  # pyright: ignore [reportUnknownVariableType]
        return [convert_provider_message_to_dict(item) for item in list_obj]
    elif is_dataclass(obj):
        # Handle dataclasses with bytes_as_str_factory to convert bytes to base64
        return asdict(obj, dict_factory=bytes_as_str_factory)  # pyright: ignore [reportUnknownArgumentType, reportArgumentType]

    # Return as-is for primitive types
    return obj
