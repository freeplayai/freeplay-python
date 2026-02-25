import os
import time
from typing import Any

from openai import OpenAI

from freeplay import Freeplay, RecordPayload, CallInfo
from freeplay.resources.recordings import UsageTokens

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {"location": "San Francisco"}

project_id = os.environ["FREEPLAY_PROJECT_ID"]

formatted_prompt = fp_client.prompts.get_formatted(
    project_id=project_id,
    template_name="my-openai-prompt",
    environment="latest",
    variables=input_variables,
)

print(f"Instructions (system): {formatted_prompt.system_content}")
print(f"Input messages: {formatted_prompt.llm_prompt}")
print(f"Tool schema: {formatted_prompt.tool_schema}")
print(f"Output schema: {formatted_prompt.formatted_output_schema}")

# Build the Responses API call
response_params: dict[str, Any] = {
    **formatted_prompt.prompt_info.model_parameters,
}
if formatted_prompt.system_content:
    response_params["instructions"] = formatted_prompt.system_content
if formatted_prompt.tool_schema:
    response_params["tools"] = formatted_prompt.tool_schema
if formatted_prompt.formatted_output_schema:
    response_params["text"] = formatted_prompt.formatted_output_schema

start = time.time()
completion = openai_client.responses.create(
    input=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    **response_params,
)
end = time.time()
print("Completion: %s" % completion)

session = fp_client.sessions.create()
messages = formatted_prompt.all_messages(completion.output)

call_info = CallInfo.from_prompt_info(
    formatted_prompt.prompt_info,
    start,
    end,
    UsageTokens(completion.usage.input_tokens, completion.usage.output_tokens),
    api_style="batch",
)
record_response = fp_client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
    )
)

print(f"Record response: {record_response.completion_id}")
