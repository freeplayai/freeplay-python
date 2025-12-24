"""
Example: Recording LLM calls with Google GenAI tool schema format.

This example demonstrates how to use Freeplay's GenaiFunction and GenaiTool
types to record function calling interactions with Google's GenAI API.

Requirements:
    pip install freeplay-client google-genai
"""

import os
from freeplay import Freeplay, GenaiFunction, GenaiTool, RecordPayload, CallInfo

# Initialize Freeplay client
freeplay = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
)

# Define tool schema using GenAI format
# GenAI uses a different structure than OpenAI/Anthropic:
# - A single Tool contains multiple FunctionDeclarations
# - This matches both GenAI API and Vertex AI formats
get_weather_function = GenaiFunction(
    name="get_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature",
            },
        },
        "required": ["location"],
    },
)

get_news_function = GenaiFunction(
    name="get_news",
    description="Get the latest news headlines",
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The news topic to search for",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of articles to return",
                "default": 5,
            },
        },
        "required": ["topic"],
    },
)

# Create a GenaiTool with multiple function declarations
# This is the key difference from OpenAI/Anthropic:
# - OpenAI: List of individual tool objects
# - GenAI: Single tool with multiple function declarations
tool_schema = [
    GenaiTool(
        functionDeclarations=[
            get_weather_function,
            get_news_function,
        ]
    )
]

# Simulate a conversation with function calling
messages = [
    {"role": "user", "content": "What's the weather like in San Francisco?"},
    {
        "role": "model",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"temperature": 68, "condition": "sunny", "humidity": 65}',
        "tool_call_id": "call_1",
    },
    {
        "role": "model",
        "content": "The weather in San Francisco is currently sunny with a temperature of 68°F and 65% humidity. It's a beautiful day!",
    },
]

# Record the conversation with GenAI tool schema
response = freeplay.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=messages,
        inputs={"user_query": "What's the weather like in San Francisco?"},
        tool_schema=tool_schema,  # Accepts GenaiTool format!
        call_info=CallInfo(
            provider="genai",  # or "vertex" - both use same tool format
            model="gemini-2.0-flash",
        ),
    )
)

print(f"✅ Recording created successfully!")
print(f"   Completion ID: {response.completion_id}")
print(f"   Session ID: {response.session_id}")
print(f"\nTool schema recorded with {len(tool_schema)} tools")
print(f"   - {len(tool_schema[0].functionDeclarations)} function declarations")
for func in tool_schema[0].functionDeclarations:
    print(f"     • {func.name}: {func.description}")

