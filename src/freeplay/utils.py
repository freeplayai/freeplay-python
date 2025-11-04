import importlib.metadata
import json
import base64
import platform
from dataclasses import asdict, is_dataclass
from typing import Dict, Union, Optional, Any, List, Tuple, cast
from uuid import UUID
import pystache  # type: ignore

from .errors import FreeplayError, FreeplayConfigurationError
from .model import (
    InputVariables,
    MediaReference,
    MediaTypes,
    TextBlock,
    MediaReferenceBlock,
    ContentBlock,
    ToolResultBlock,
    ToolCallBlock,
)


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
    """Custom dict factory to convert bytes to base64 strings and UUIDs to strings for dataclasses.
    Used with asdict() to handle Bedrock and other providers that use byte strings, and UUID objects.
    Also converts downloaded_file media references to uploaded_file for recording."""

    result: Dict[str, Any] = {}
    for key, value in field_list:
        if isinstance(value, bytes):
            result[key] = base64.b64encode(value).decode("utf-8")
        elif isinstance(value, UUID):
            result[key] = str(value)
        elif key == "kind" and value == "downloaded_file":
            # Convert downloaded_file to uploaded_file for recording/adapters
            result[key] = "uploaded_file"
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
        result = {k: convert_provider_message_to_dict(v) for k, v in dict_obj.items()}

        # Convert downloaded_file to uploaded_file in media references
        if result.get("kind") == "downloaded_file":
            result["kind"] = "uploaded_file"

        return result
    elif isinstance(obj, list):
        # Handle lists recursively
        list_obj: List[Any] = obj  # pyright: ignore [reportUnknownVariableType]
        return [convert_provider_message_to_dict(item) for item in list_obj]
    elif is_dataclass(obj):
        # Handle dataclasses with bytes_as_str_factory to convert bytes to base64
        return asdict(obj, dict_factory=bytes_as_str_factory)  # pyright: ignore [reportUnknownArgumentType, reportArgumentType]

    # Return as-is for primitive types
    return obj


def convert_api_message_content_to_sdk_message_content(
    content: Any,
) -> Union[str, List[ContentBlock]]:
    """
    Normalize message content from API format to SDK format.
    Args:
        content: The content field from a message (can be str or list of content blocks)

    Returns:
        Normalized content (str or list of ContentBlock objects)
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        # If it's not a string or list, return as-is
        return content

    normalized_blocks: List[ContentBlock] = []

    for block in content:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(block, dict):
            # If block is not a dict, we can't process it
            continue

        # Cast to Dict[str, Any] since we verified it's a dict
        block = cast(Dict[str, Any], block)
        block_type = block.get("type")

        if block_type == "text":
            text_value = block.get("text")
            if isinstance(text_value, str):
                normalized_blocks.append(TextBlock(text=text_value))
        elif block_type == "tool_result":
            tool_call_id = block.get("tool_call_id")
            content_val = block.get("content")
            if isinstance(tool_call_id, str) and content_val is not None:
                normalized_blocks.append(
                    ToolResultBlock(tool_call_id=tool_call_id, content=content_val)
                )
        elif block_type == "tool_call":
            id_val = block.get("id")
            name_val = block.get("name")
            args_val = block.get("arguments")
            if isinstance(id_val, str) and isinstance(name_val, str):
                normalized_blocks.append(
                    ToolCallBlock(id=id_val, name=name_val, arguments=args_val)
                )
        elif block_type in ["image", "audio", "file"]:
            # Handle media references (both uploaded_file and downloaded_file)
            media_ref = block.get("media_reference")
            if media_ref and isinstance(media_ref, dict):
                media_ref = cast(Dict[str, Any], media_ref)
                base64_data = media_ref.get("base64Data")
                content_type = media_ref.get("contentType")
                media_type = media_ref.get("media_type")
                kind = media_ref.get("kind", "uploaded_file")

                if (
                    isinstance(base64_data, str)
                    and isinstance(content_type, str)
                    and media_type in ["image", "audio", "file"]
                    and kind in ["uploaded_file", "downloaded_file"]
                ):
                    normalized_blocks.append(
                        MediaReferenceBlock(
                            media_reference=MediaReference(
                                base64Data=base64_data,
                                contentType=content_type,
                                media_type=cast(MediaTypes, media_type),
                                filename=media_ref.get("filename"),
                                kind=kind,  # Keep the original kind
                            )
                        )
                    )

    return normalized_blocks if normalized_blocks else content  # pyright: ignore[reportUnknownVariableType]


def convert_api_message_to_sdk_message(message: Any) -> Dict[str, Any]:
    """
    Normalize a message from API format to SDK format.

    Args:
        message: Message dict from the API

    Returns:
        Normalized message dict
    """
    if not isinstance(message, dict):
        return message

    normalized = dict(cast(Dict[str, Any], message))

    if "content" in normalized:
        normalized["content"] = convert_api_message_content_to_sdk_message_content(
            normalized["content"]
        )

    return normalized


def convert_sdk_messages_to_api_messages(
    message: Union[Dict[str, Any], Any],
) -> Dict[str, Any]:
    """
    Serialize a NormalizedMessage (UserMessage, SystemMessage, AssistantMessage) to a dict for API.

    Converts ContentBlock objects to dictionaries suitable for the API.

    Args:
        message: NormalizedMessage object or dict to serialize

    Returns:
        Dictionary representation suitable for API
    """
    # If it's already a dict, check if it has ContentBlock objects in content
    if isinstance(message, dict):
        result = dict(message)  # pyright: ignore[reportUnknownArgumentType]
        if "content" in result and isinstance(result["content"], list):
            # Serialize the content blocks
            content_list: List[Dict[str, Any]] = []
            for block in result["content"]:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, MediaReferenceBlock):
                    content_list.append(
                        {
                            "type": block.media_reference.media_type,
                            "media_reference": {
                                "base64Data": block.media_reference.base64Data,
                                "contentType": block.media_reference.contentType,
                                "media_type": block.media_reference.media_type,
                                "filename": block.media_reference.filename,
                                "kind": block.media_reference.kind,
                            },
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    content_list.append(
                        {
                            "type": "tool_result",
                            "tool_call_id": block.tool_call_id,
                            "content": block.content,
                        }
                    )
                elif isinstance(block, ToolCallBlock):
                    content_list.append(
                        {
                            "type": "tool_call",
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.arguments,
                        }
                    )
                elif is_dataclass(block) and not isinstance(block, type):  # pyright: ignore[reportUnknownArgumentType]
                    # Handle other dataclass instances (not types) using asdict
                    content_list.append(
                        asdict(block, dict_factory=bytes_as_str_factory)
                    )
                elif isinstance(block, dict):
                    # Already a dict
                    content_list.append(block)  # pyright: ignore[reportUnknownArgumentType]
                else:
                    # Unknown type, convert to dict or skip
                    if hasattr(block, "__dict__"):  # pyright: ignore[reportUnknownArgumentType]
                        content_list.append(dict(vars(block)))  # pyright: ignore[reportUnknownArgumentType]
                    else:
                        # Can't convert, pass as dict with raw value
                        content_list.append({"type": "unknown", "value": str(block)})  # pyright: ignore[reportUnknownArgumentType]
            result["content"] = content_list
        return result

    # Handle NormalizedMessage dataclass objects
    if not hasattr(message, "role"):
        # Not a message object, return as-is
        return message

    result: Dict[str, Any] = {"role": message.role}  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    # Handle content
    content = getattr(message, "content", None)
    if isinstance(content, str):
        result["content"] = content
    elif isinstance(content, list):
        # Convert list of ContentBlock objects to list of dicts
        content_list_out: List[Dict[str, Any]] = []
        for block in content:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(block, TextBlock):
                content_list_out.append({"type": "text", "text": block.text})
            elif isinstance(block, MediaReferenceBlock):
                content_list_out.append(
                    {
                        "type": block.media_reference.media_type,
                        "media_reference": {
                            "base64Data": block.media_reference.base64Data,
                            "contentType": block.media_reference.contentType,
                            "media_type": block.media_reference.media_type,
                            "filename": block.media_reference.filename,
                            "kind": block.media_reference.kind,
                        },
                    }
                )
            elif isinstance(block, ToolResultBlock):
                content_list_out.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": block.tool_call_id,
                        "content": block.content,
                    }
                )
            elif isinstance(block, ToolCallBlock):
                content_list_out.append(
                    {
                        "type": "tool_call",
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.arguments,
                    }
                )
            elif is_dataclass(block) and not isinstance(block, type):  # pyright: ignore[reportUnknownArgumentType]
                # Handle other dataclass instances (not types) using asdict
                content_list_out.append(
                    asdict(block, dict_factory=bytes_as_str_factory)
                )
            elif isinstance(block, dict):
                # Already a dict
                content_list_out.append(block)  # pyright: ignore[reportUnknownArgumentType]
            else:
                # Unknown type, convert to dict or skip
                if hasattr(block, "__dict__"):  # pyright: ignore[reportUnknownArgumentType]
                    content_list_out.append(dict(vars(block)))  # pyright: ignore[reportUnknownArgumentType]
                else:
                    # Can't convert, pass as dict with raw value
                    content_list_out.append({"type": "unknown", "value": str(block)})  # pyright: ignore[reportUnknownArgumentType]
        result["content"] = content_list_out
    else:
        # Unknown content type, just pass through
        result["content"] = content

    return result
