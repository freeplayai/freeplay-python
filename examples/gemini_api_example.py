"""
Generate content with Gemini using the google-genai SDK and Freeplay prompts.

Demonstrates that model_parameters (temperature, thinking_level, max_tokens)
are automatically mapped to Gemini-compatible format by format(), so they can
be spread directly into GenerateContentConfig without manual transformation.

Prerequisites:
    pip install google-genai

Environment variables:
    FREEPLAY_API_KEY          - Freeplay API key
    FREEPLAY_API_URL          - Freeplay API URL
    FREEPLAY_PROJECT_ID       - Freeplay project ID
    GEMINI_API_KEY            - Google Gemini API key
    FREEPLAY_PROMPT_TEMPLATE_NAME - (optional) prompt template name, default "my-gemini-prompt"
    FREEPLAY_ENVIRONMENT      - (optional) environment, default "latest"
"""

import os
import time

from google import genai
from google.genai import types

from freeplay import CallInfo, Freeplay, RecordPayload, ResponseInfo
from freeplay.utils import convert_provider_message_to_dict

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)

project_id = os.environ["FREEPLAY_PROJECT_ID"]
template_name = os.environ.get("FREEPLAY_PROMPT_TEMPLATE_NAME", "my-gemini-prompt")
environment = os.environ.get("FREEPLAY_ENVIRONMENT", "latest")

input_variables = {"location": "Boulder"}
formatted_prompt = (
    fp_client.prompts.get(
        project_id=project_id,
        template_name=template_name,
        environment=environment,
    )
    .bind(input_variables, history=[])
    .format()
)

# model_parameters are automatically mapped to Gemini-compatible names by format():
#   max_tokens      -> max_output_tokens
#   thinking_level  -> thinking_config
#   temperature     -> temperature (unchanged)
# They can be spread directly into GenerateContentConfig.
print(f"Model: {formatted_prompt.prompt_info.model}")
print(f"Mapped parameters: {dict(formatted_prompt.prompt_info.model_parameters)}")

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

start = time.time()

response = gemini_client.models.generate_content(
    model=formatted_prompt.prompt_info.model,
    contents=formatted_prompt.llm_prompt,
    config=types.GenerateContentConfig(
        system_instruction=formatted_prompt.system_content,
        **formatted_prompt.prompt_info.model_parameters,
    ),
)

end = time.time()

print(f"\nResponse: {response.text[:200]}...")

# Build messages for recording -- convert SDK objects to dicts for JSON serialization
assistant_message = convert_provider_message_to_dict(response.candidates[0].content)
all_messages = list(formatted_prompt.llm_prompt)
all_messages.append(assistant_message)

session = fp_client.sessions.create()

fp_client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=all_messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=CallInfo(
            start_time=start,
            end_time=end,
            provider="gemini",
            model=formatted_prompt.prompt_info.model,
        ),
        response_info=ResponseInfo(is_complete=True),
    )
)

print("\nRecorded completion successfully.")
