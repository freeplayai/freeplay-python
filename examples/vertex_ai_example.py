import os

from google import genai
from google.genai import types

from freeplay import Freeplay, RecordPayload, CallInfo
from freeplay.utils import convert_provider_message_to_dict

# Initialize Freeplay and get formatted prompt
fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)

input_variables = {"location": "Boulder"}
formatted_prompt = (
    fp_client.prompts.get(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        template_name="my-openai-prompt",
        environment="latest",
    )
    .bind(input_variables, history=[])
    .format()
)

# Initialize the google-genai client for Vertex AI
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", "fp-d-int-069c"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

# model_parameters are automatically mapped to Gemini-compatible names by format():
#   max_tokens -> max_output_tokens, thinking_level -> thinking_config, etc.
# They can be passed directly as the generation config.
response = client.models.generate_content(
    model=formatted_prompt.prompt_info.model,
    contents=formatted_prompt.llm_prompt,
    config=types.GenerateContentConfig(
        system_instruction=formatted_prompt.system_content,
        tools=formatted_prompt.tool_schema,
        **formatted_prompt.prompt_info.model_parameters,
    ),
)

content = response.candidates[0].content

# Build messages for recording
all_messages = list(formatted_prompt.llm_prompt)
all_messages.append(content)
all_messages_dict = [convert_provider_message_to_dict(msg) for msg in all_messages]

fp_client.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=all_messages_dict,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        tool_schema=formatted_prompt.tool_schema,
        call_info=CallInfo(
            provider="vertex",
            model=formatted_prompt.prompt_info.model,
        ),
    )
)
