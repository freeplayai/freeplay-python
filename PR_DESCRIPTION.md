# Tool Schema Format Support

## Summary

Updates the Python SDK to clarify that tool schemas should be passed in their native provider format (as dictionaries). The SDK accepts tool schemas from OpenAI, Anthropic, and Google GenAI/Vertex AI directly, without requiring conversion to Freeplay-specific wrapper types.

## What Changed

### Tool Schema Handling
- **Simplified Approach**: Tool schemas are passed as raw dictionaries matching the provider's native format
- **Consistency**: Aligns with how messages are handled - provider-native types passed directly to Freeplay
- **Backend Normalization**: Backend automatically handles all formats (OpenAI, Anthropic, GenAI/Vertex)

### Interactive REPL
- Added `make repl` command for development and testing
- Pre-loads Freeplay client and environment variables
- Includes SSL verification disabled for local development

### Documentation & Testing
- Updated testing guides to show raw dictionary approach
- All tool schema formats (OpenAI, Anthropic, GenAI) remain backward compatible

## Key Features

### GenAI Tool Schema Format

GenAI uses a unique structure where a **single tool contains multiple function declarations**, unlike OpenAI/Anthropic where each tool is separate:

```python
from freeplay import RecordPayload, CallInfo

# Tool schema as raw dictionary (GenAI/Vertex format)
tool_schema = [
    {
        "functionDeclarations": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    }
]

# Use in recordings
client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[...],
        tool_schema=tool_schema,
        call_info=CallInfo(provider="vertex", model="gemini-2.0-flash")
    )
)
```

### Using Tool Schemas from Provider SDKs

Tool schemas can be obtained directly from provider SDKs:

```python
# From google-generativeai SDK
import google.generativeai as genai

tools = [
    {
        "functionDeclarations": [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {...}
            }
        ]
    }
]

# Pass directly to Freeplay
client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[...],
        tool_schema=tools,  # Use directly from provider SDK
        call_info=CallInfo(provider="vertex", model="gemini-2.0-flash")
    )
)
```

### Backward Compatibility

- ✅ Existing code using raw dicts continues to work
- ✅ OpenAI and Anthropic tool formats unchanged
- ✅ Backend automatically normalizes all formats

## Testing

### Unit Tests
- Existing tests for tool schema normalization continue to pass
- Integration tests verify backend normalization

### Integration Testing
- Manual testing guide included with test scenarios
- Tested with local Freeplay instance
- Verified tool schema storage and normalization

## Related Work

This approach is consistent across all Freeplay SDKs:
- Python SDK (this PR)
- Node.js SDK (separate PR)
- Java SDK (separate PR)

All SDKs now follow the same principle: accept provider-native types without requiring conversion to SDK-specific wrapper types.

## Version

Updates to **0.5.7** with the following in CHANGELOG:
- Clarified tool schema handling approach
- Interactive REPL

## Checklist

- ✅ Testing guide provided
- ✅ CHANGELOG updated (v0.5.7)
- ✅ Backward compatibility maintained
- ✅ Documentation updated
- ✅ Consistent with message handling approach







