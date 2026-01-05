"""
Generate image with OpenAI DALL-E, then discuss it in a follow-up turn.

This example:
1. Generates an image with DALL-E 3
2. Records Turn 1 (DALL-E image generation) with the image as base64 data URI
3. Adds the generated image to history
4. In Turn 2, uses 'image_generation' prompt with history to ask vision question about the image
5. Records the followup conversation with actual LLM responses and start/stop times
"""

import os
import time
import base64
import requests

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from freeplay import CallInfo, Freeplay, RecordPayload, ResponseInfo
from freeplay.resources.recordings import UsageTokens

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

project_id = os.environ["FREEPLAY_PROJECT_ID"]

print("=" * 80)
print("TURN 1: Generate image with OpenAI DALL-E")
print("=" * 80)

# Create session for multi-turn conversation
session = fp_client.sessions.create()
print(f"Created session ID: {session.session_info.session_id}")

# Turn 1: Generate an image
image_description = "A serene mountain landscape at sunset with a crystal clear lake reflecting the golden sky, surrounded by pine trees"
input_variables_1 = {"description": image_description}

start_1 = time.time()

# Generate image using DALL-E
print(f"Generating image: {image_description}")
image_response = openai_client.images.generate(
    model="dall-e-3",
    prompt=image_description,
    size="1024x1024",
    quality="standard",
    n=1,
)

if not image_response.data or len(image_response.data) == 0:
    raise ValueError("No image data returned from DALL-E API")

generated_image_url = image_response.data[0].url
if not generated_image_url:
    raise ValueError("No image URL returned from DALL-E API")

print(f"Image generated: {generated_image_url[:80]}...")

# Download the image and convert to base64
print("Downloading generated image...")
image_response = requests.get(generated_image_url)
image_response.raise_for_status()
image_base64 = base64.b64encode(image_response.content).decode("utf-8")
image_data_uri = f"data:image/png;base64,{image_base64}"
print(f"Image downloaded ({len(image_response.content)} bytes)")

end_1 = time.time()

# Create assistant message in OpenAI format with embedded base64 image
# This format now works with the updated backend
assistant_message_with_image = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Here's the generated image:"},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ],
}

environment = os.environ.get("FREEPLAY_ENVIRONMENT", "latest")
template_name = os.environ.get("FREEPLAY_PROMPT_TEMPLATE_NAME", "image_generation")
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name=template_name,
    environment=environment,
    variables=input_variables_1,
    history=[],
)

# Build message history using the formatted prompt's llm_prompt
all_messages_1 = [*formatted_prompt.llm_prompt, assistant_message_with_image]

call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start_1, end_1)
response_info = ResponseInfo(is_complete=True)

record_response_1 = fp_client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=all_messages_1,
        session_info=session.session_info,
        inputs=input_variables_1,
        call_info=call_info,
        response_info=response_info,
        prompt_version_info=formatted_prompt.prompt_info,
    )
)

print(f"Recorded Turn 1 completion ID: {record_response_1.completion_id}")

# Turn 2: Ask about the generated image
print("\n" + "=" * 80)
print("TURN 2: Discuss the generated image")
print("=" * 80)

input_variables_2 = {
    "description": "What are the main colors and mood in this landscape?"
}

# Build OpenAI-format history for Turn 2 (what was sent to/from DALL-E conceptually)
history_for_turn2 = [
    assistant_message_with_image,
]

# Get formatted prompt for Turn 2 (vision analysis) with history

formatted_prompt_2 = fp_client.prompts.get_formatted(
    project_id=project_id,
    template_name=template_name,
    environment=environment,
    variables=input_variables_2,
    history=history_for_turn2,  # Pass OpenAI-format history with the generated image
)

# Use the formatted prompt's messages for Turn 2
turn2_messages = formatted_prompt_2.llm_prompt

print(
    f"Turn 2: {len(turn2_messages)} messages (includes generated image from Turn 1 history)"
)

# Call OpenAI for Turn 2
# Filter parameters to only those supported by the provider/model
# Newer versions of the OpenAI SDK are needed here, and will be supported in the future
# These are supported via the Playground/API, but are not yet available in the SDK
parameters = formatted_prompt_2.prompt_info.model_parameters
parameters.pop("verbosity", None)
parameters.pop("reasoning_effort", None)
start_2 = time.time()
completion_2: ChatCompletion = openai_client.chat.completions.create(  # pyright: ignore[reportUnknownVariableType]
    messages=turn2_messages,
    model=formatted_prompt_2.prompt_info.model,
    **parameters,
)
end_2 = time.time()

print(f"Assistant: {completion_2.choices[0].message.content[:100]}...")  # pyright: ignore[reportUnknownMemberType, reportOptionalSubscript]

# Build full message history for recording Turn 2
# Now that backend supports images in assistant messages, we can use the full format
all_messages_2 = formatted_prompt_2.all_messages(completion_2.choices[0].message)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

response_info_2 = ResponseInfo(
    is_complete=completion_2.choices[0].finish_reason == "stop"  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
)

# Create CallInfo with usage tokens from the OpenAI response
call_info_2 = CallInfo.from_prompt_info(
    formatted_prompt_2.prompt_info,
    start_2,
    end_2,
    UsageTokens(completion_2.usage.prompt_tokens, completion_2.usage.completion_tokens),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportOptionalMemberAccess]
)

record_response_2 = fp_client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=all_messages_2,
        session_info=session.session_info,
        inputs=input_variables_2,
        prompt_version_info=formatted_prompt_2.prompt_info,
        call_info=call_info_2,
        tool_schema=formatted_prompt_2.tool_schema,
        response_info=response_info_2,
    )
)

print(f"Recorded Turn 2 completion ID: {record_response_2.completion_id}")

print("\n" + "=" * 80)
print("Multi-turn with generated image complete!")
print(f"Session ID: {session.session_info.session_id}")
print(
    f"View in UI: {os.environ.get('FREEPLAY_API_URL', 'http://localhost:8080')}/projects/{project_id}/sessions"
)
print("=" * 80)
