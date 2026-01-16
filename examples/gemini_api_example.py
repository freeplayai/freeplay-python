"""
Example: Using the Gemini API Provider

This example demonstrates how to use Google's Gemini API provider with Freeplay.
The Gemini API provider uses simple API key authentication, compared to Vertex AI
which requires GCP service account authentication.

Both providers support the same Gemini models but differ in:
- Authentication method (API key vs. GCP credentials)
- Endpoint (generativelanguage.googleapis.com vs. aiplatform.googleapis.com)
- Setup complexity (simple vs. GCP infrastructure)

Prerequisites:
- Set FREEPLAY_API_KEY in your environment
- Set FREEPLAY_PROJECT_ID in your environment
- Gemini API configured in Freeplay UI with API key
"""

import os
from freeplay import Freeplay, RecordPayload, CallInfo

# Initialize Freeplay client
client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base="https://app.freeplay.ai/api",
)

project_id = os.environ["FREEPLAY_PROJECT_ID"]

# Example 1: Basic recording with Gemini API
print("Example 1: Basic Gemini API recording")
response = client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        call_info=CallInfo(
            provider="gemini",  # Use Gemini API provider
            model="gemini-2.0-flash",
        ),
    )
)
print(f"✅ Recorded with Gemini API: {response.completion_id}")

# Example 2: Gemini API with tool schema
print("\nExample 2: Gemini API with tool schema")

tool_schema = [
    {
        "functionDeclarations": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g., 'San Francisco'",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]
    }
]

response = client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": "Let me check the weather for you."},
        ],
        tool_schema=tool_schema,
        call_info=CallInfo(
            provider="gemini",  # Use Gemini API provider
            model="gemini-2.5-pro",
        ),
    )
)
print(f"✅ Recorded with tool schema: {response.completion_id}")

# Example 3: Comparison - Vertex AI vs Gemini API
print("\nExample 3: Provider comparison")

# Vertex AI (GCP authentication)
vertex_response = client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            {"role": "user", "content": "Hello from Vertex AI"},
            {"role": "assistant", "content": "Hello! I'm using Vertex AI."},
        ],
        call_info=CallInfo(
            provider="vertex",  # Vertex AI provider
            model="gemini-1.5-pro",
        ),
    )
)
print(f"✅ Vertex AI: {vertex_response.completion_id}")

# Gemini API (simple API key)
gemini_response = client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            {"role": "user", "content": "Hello from Gemini API"},
            {"role": "assistant", "content": "Hello! I'm using Gemini API."},
        ],
        call_info=CallInfo(
            provider="gemini",  # Gemini API provider
            model="gemini-2.0-flash",
        ),
    )
)
print(f"✅ Gemini API: {gemini_response.completion_id}")

print("\n" + "=" * 50)
print("Summary:")
print("- Both providers support Gemini models")
print("- Vertex AI: GCP integration, enterprise features")
print("- Gemini API: Simple API key, quick setup")
print("- Choose based on your infrastructure needs")
print("=" * 50)
