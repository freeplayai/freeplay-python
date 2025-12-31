"""
Create an image via Gemini and record the completion in Freeplay.
"""

import os
import time
import base64
import json
import requests

from freeplay import Freeplay, CallInfo, ResponseInfo, RecordPayload
from freeplay.utils import convert_provider_message_to_dict


fpclient = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)

# Text description for image generation
image_description = "A futuristic cityscape at night with neon lights reflecting on wet streets, cyberpunk style with flying cars and towering skyscrapers"
input_variables = {"description": image_description}

# Optional: Get prompt info for tracking (even though we're not using the formatted messages)
# This allows Freeplay to track which prompt template was used
project_id = os.environ["FREEPLAY_PROJECT_ID"]
environment = os.environ.get("FREEPLAY_ENVIRONMENT", "latest")
template_name = os.environ.get("FREEPLAY_PROMPT_TEMPLATE_NAME", "imagegen")

formatted_prompt = fpclient.prompts.get_formatted(
    project_id=project_id,
    template_name=template_name,
    environment=environment,
    variables=input_variables,
)

start = time.time()

# Try to generate image using Gemini 2.5 Flash Image
print(f"Generating image with description: {image_description}")

model_name = "gemini-2.5-flash-image"  # Use the correct model for image generation
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Prepare the request for image generation
request_data = {
    "contents": [{"parts": [{"text": f"Create a picture of {image_description}"}]}]
}

try:
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_data),
        timeout=60,
    )

    if response.status_code == 429:
        print("Quota exceeded. This is likely due to free tier limits.")
        print("Options:")
        print("1. Wait a few minutes and try again")
        print("2. Upgrade to a paid Gemini API plan")
        print(
            "3. Check your quota usage at: https://ai.google.dev/gemini-api/docs/rate-limits"
        )
        raise ValueError("Gemini API quota exceeded. Please wait or upgrade your plan.")
    elif response.status_code != 200:
        # Print the API response, normally, you wouldn't want to do this in production code
        print(f"API request failed with status {response.status_code}: {response.text}")
        raise ValueError(f"Gemini API request failed: {response.text}")

    response_data = response.json()

    # Check if the response contains an image
    if "candidates" not in response_data or not response_data["candidates"]:
        raise ValueError("No candidates returned from Gemini API")

    candidate = response_data["candidates"][0]
    if "content" not in candidate or "parts" not in candidate["content"]:
        raise ValueError("No content parts in Gemini response")

    # Keep the full assistant message in provider format (preserves inlineData)
    assistant_message = candidate["content"]

    if "role" not in assistant_message:
        assistant_message["role"] = "model"

    # Log what we received for debugging
    parts = assistant_message["parts"]
    has_image = any("inlineData" in part for part in parts)
    has_text = any("text" in part for part in parts)

    if has_image:
        image_part = next(part for part in parts if "inlineData" in part)
        image_data = image_part["inlineData"]["data"]
        image_bytes = base64.b64decode(image_data)
        content_type = image_part["inlineData"].get("mimeType", "image/png")
        print(f"Generated image (size: {len(image_bytes)} bytes.")

    if has_text:
        response_text = "".join(part["text"] for part in parts if "text" in part)
        print(f"Response text: {response_text[:100]}...")

    if not has_image and not has_text:
        raise ValueError("No image or text found in Gemini response")

except requests.RequestException as e:
    raise ValueError(f"Failed to call Gemini API: {e}")

end = time.time()

# Create a completion record for the image generation
session = fpclient.sessions.create()
call_info = CallInfo(
    start_time=start,
    end_time=end,
    provider="gemini",
    model=model_name,
)
response_info = ResponseInfo(is_complete=True)

all_messages = list(formatted_prompt.llm_prompt)
all_messages.append(assistant_message)
all_messages_dict = [convert_provider_message_to_dict(msg) for msg in all_messages]
record_response = fpclient.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=all_messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=call_info,
        response_info=response_info,
    )
)

print("Successfully generated and recorded image with Gemini")
print("Image preserved in assistant message output (not as input media)")
print(f"Recorded completion ID: {record_response.completion_id}")
print()
