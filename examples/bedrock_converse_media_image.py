import base64
import os
import time
from typing import Tuple
import boto3  # type: ignore
import requests  # type: ignore

from freeplay import CallInfo, Freeplay, RecordPayload
from freeplay.model import MediaInputBase64, MediaInputMap


def load_image_from_url(image_url: str) -> Tuple[bytes, str]:
    """Download an image from URL and return bytes with format."""
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content

    # Determine format from URL or content-type
    content_type = response.headers.get("content-type", "")
    if "jpeg" in content_type or "jpg" in content_type:
        image_format = "jpeg"
    elif "png" in content_type:
        image_format = "png"
    elif "gif" in content_type:
        image_format = "gif"
    elif "webp" in content_type:
        image_format = "webp"
    else:
        # Try to infer from URL
        ext = image_url.lower().split(".")[-1].split("?")[0]
        format_map = {
            "jpg": "jpeg",
            "jpeg": "jpeg",
            "png": "png",
            "gif": "gif",
            "webp": "webp",
        }
        image_format = format_map.get(ext, "jpeg")

    return image_bytes, image_format


fp_client = Freeplay(
    api_base=f"{os.environ.get('FREEPLAY_API_URL')}/api",
    freeplay_api_key=os.environ.get("FREEPLAY_API_KEY") or "",
)

aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") or ""
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
if not aws_access_key_id or not aws_secret_access_key:
    raise Exception("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")

converse_client = boto3.client(  # type: ignore
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

project_id: str = os.environ["FREEPLAY_PROJECT_ID"]

# Image URL
image_url = "https://images.pexels.com/photos/30614903/pexels-photo-30614903/free-photo-of-aerial-view-of-bilbao-city-and-guggenheim-museum.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

# Download the image
print(f"Downloading image from: {image_url}")
image_bytes, image_format = load_image_from_url(image_url)
print(f"Downloaded image (format: {image_format}, size: {len(image_bytes)} bytes)")

# Question about the image
question = "What do you see in this image? Describe it in detail."
prompt_vars = {"question": question}

# Convert image to base64 for Freeplay media_inputs
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
content_type = f"image/{image_format}"

media_inputs: MediaInputMap = {
    "city-image": MediaInputBase64(
        type="base64", data=image_base64, content_type=content_type
    )
}

# Get formatted prompt from Freeplay
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=project_id,
    template_name="nova_image_test",
    environment="latest",
    variables=prompt_vars,
    media_inputs=media_inputs,
)

start = time.time()

session = fp_client.sessions.create()

# Call Bedrock API with plain dict messages
response = converse_client.converse(  # type: ignore
    modelId=formatted_prompt.prompt_info.model,
    messages=formatted_prompt.llm_prompt,
    system=[{"text": formatted_prompt.system_content or ""}],
    inferenceConfig=formatted_prompt.prompt_info.model_parameters,
)
output_message = response["output"]["message"]  # type: ignore
response_content = output_message["content"][0]["text"]  # type: ignore
end = time.time()


print(f"Using model: {formatted_prompt.prompt_info.model}")
print(f"Template: {formatted_prompt.prompt_info.template_name}")
print("\n=== Model Response ===")
print(response_content)  # type: ignore
print("\n=== Recording to Freeplay ===")


# Prepare messages for recording - convert provider messages to dicts
record_response = fp_client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=[
            *formatted_prompt.llm_prompt,
            output_message,
        ],
        session_info=session.session_info,
        inputs=prompt_vars,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end),
        media_inputs=media_inputs,
    )
)

print("Successfully recorded to Freeplay")
print(f"Recorded completion ID: {record_response.completion_id}")
