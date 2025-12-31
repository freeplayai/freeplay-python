import os
import time
from typing import List

import openai
import pydantic
from openai import OpenAI

from freeplay import Freeplay, RecordPayload, ResponseInfo, CallInfo
from freeplay.resources.recordings import UsageTokens


# Structured output classes
class COTStep(pydantic.BaseModel):
    thinking: str
    result: str


class COTResponse(pydantic.BaseModel):
    response: str
    steps: List[COTStep]


fpclient = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {"question": "why is the sky blue?"}

project_id = os.environ["FREEPLAY_PROJECT_ID"]

formatted_prompt = fpclient.prompts.get_formatted(
    project_id=project_id,
    template_name="my-chat-template",
    environment="latest",
    variables=input_variables,
)

print(f"Tool schema: {formatted_prompt.tool_schema}")
print(f"Output schema: {formatted_prompt.formatted_output_schema}")

start = time.time()

# Build the completion parameters
completion_params = {
    **formatted_prompt.prompt_info.model_parameters,
}

# Add tools if present
if formatted_prompt.tool_schema:
    completion_params["tools"] = formatted_prompt.tool_schema

completion = client.chat.completions.parse(
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    response_format=COTResponse,
)

end = time.time()
print("Completion with pydantic model: %s" % completion)

completion_with_schema_from_prompt = client.chat.completions.create(
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": formatted_prompt.formatted_output_schema,
            "name": "COTReasoning",
        },
    }
    if formatted_prompt.formatted_output_schema
    else openai.NotGiven(),
)
print("Completion with prompt schema: %s" % completion_with_schema_from_prompt)

session = fpclient.sessions.create()
messages = formatted_prompt.all_messages(completion.choices[0].message)
print(f"All messages: {messages}")
call_info = CallInfo.from_prompt_info(
    formatted_prompt.prompt_info,
    start,
    end,
    UsageTokens(completion.usage.prompt_tokens, completion.usage.completion_tokens),
    api_style="batch",
)
response_info = ResponseInfo(is_complete=completion.choices[0].finish_reason == "stop")

print(f"Messages: {messages}")
record_response = fpclient.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
        output_schema=COTResponse.model_json_schema(),
        response_info=response_info,
    )
)
