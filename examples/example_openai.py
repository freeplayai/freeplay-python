import os
import time

from openai import OpenAI

from freeplay import Freeplay, RecordPayload, ResponseInfo, CallInfo
from freeplay.resources.recordings import UsageTokens

# logging.basicConfig(level=logging.NOTSET)

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {"question": "why is the sky blue?"}

project_id = os.environ["FREEPLAY_PROJECT_ID"]
prompt = fp_client.prompts.get(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name="my-openai-prompt",
    environment="latest",
)

print(f"Tool Schema from simple prompt: {prompt.tool_schema}")

formatted_prompt = fp_client.prompts.get_formatted(
    project_id=project_id,
    template_name="my-openai-prompt",
    environment="latest",
    variables=input_variables,
)

print(f"Tool schema: {formatted_prompt.tool_schema}")

start = time.time()
completion = openai_client.chat.completions.create(
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    tools=formatted_prompt.tool_schema,
    **formatted_prompt.prompt_info.model_parameters,
)
end = time.time()
print("Completion: %s" % completion)

session = fp_client.sessions.create()
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
record_response = fp_client.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
        response_info=response_info,
    )
)

print(f"Sending customer feedback for completion id: {record_response.completion_id}")
fp_client.customer_feedback.update(
    project_id,
    record_response.completion_id,
    {"is_it_good": "nah", "count_of_interactions": 123},
)
