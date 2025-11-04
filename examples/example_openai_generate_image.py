"""
Example demonstrating image generation using OpenAI's DALL-E 3 API with Freeplay tracking.

This example:
1. Generates an image from a text description using DALL-E 3
2. Downloads the generated image and converts it to base64
3. Records the image generation as a completion in Freeplay using OpenAI message format
"""

import os
import time

from openai import OpenAI

from freeplay import Freeplay, CallInfo, ResponseInfo, RecordPayload


fpclient = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Text description for image generation
image_description = "A serene mountain landscape at sunset with a crystal clear lake reflecting the golden sky, surrounded by pine trees and snow-capped peaks"
input_variables = {"description": image_description}

start = time.time()

# Generate image using DALL-E
print(f"Generating image with description: {image_description}")
image_response = client.images.generate(
    model="dall-e-3",
    prompt=image_description,
    size="1024x1024",
    quality="standard",
    n=1,
    response_format="b64_json",
)

# Get the generated image URL with proper error handling
if not image_response.data or len(image_response.data) == 0:
    raise ValueError("No image data returned from DALL-E API")


image_data_uri = f"data:image/png;base64,{image_response.data[0].b64_json}"
print(f"Image data URI: {image_data_uri[:100]}")
end = time.time()

# Create a completion record for the image generation
session = fpclient.sessions.create()

# Here we just create a message with the image data URI to store it.
assistant_message_with_image = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Here's the generated image."},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ],
}

# Get formatted prompt to use for recording
environment = os.environ.get("FREEPLAY_ENVIRONMENT", "latest")
template_name = os.environ.get("FREEPLAY_PROMPT_TEMPLATE_NAME", "image_generation")
formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name=template_name,
    environment=environment,
    variables=input_variables,
    history=[],
)

# Build message history using the formatted prompt's llm_prompt + assistant response
all_messages = [*formatted_prompt.llm_prompt, assistant_message_with_image]

call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
response_info = ResponseInfo(is_complete=True)

# Record the image generation with OpenAI format messages
record_response = fpclient.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=all_messages,
        session_info=session.session_info,
        inputs=input_variables,
        call_info=call_info,
        response_info=response_info,
        prompt_version_info=formatted_prompt.prompt_info,
    )
)

print("Successfully generated and recorded image")
print(f"Recorded completion ID: {record_response.completion_id}")
