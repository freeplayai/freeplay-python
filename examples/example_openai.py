import os
import time
import logging
from openai import OpenAI

from freeplay import Freeplay, RecordPayload, ResponseInfo, CallInfo

# logging.basicConfig(level=logging.NOTSET)

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

input_variables = {'question': "why is the sky blue?"}

prompt = fpclient.prompts.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-openai-prompt',
    environment='latest'
)

print(f"Tool Schema from simple prompt: {prompt.tool_schema}")

formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-openai-prompt',
    environment='latest',
    variables=input_variables
)

print(f"Tool schema: {formatted_prompt.tool_schema}")

start = time.time()
completion = client.chat.completions.create(
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    tools=formatted_prompt.tool_schema,
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion)

session = fpclient.sessions.create()
messages = formatted_prompt.all_messages(completion.choices[0].message)
print(f"All messages: {messages}")
call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
response_info = ResponseInfo(
    is_complete=completion.choices[0].finish_reason == 'stop'
)

print(f"Messages: {messages}")
record_response = fpclient.recordings.create(
    RecordPayload(
        all_messages=messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
        response_info=response_info,
    )
)

print(f"Sending customer feedback for completion id: {record_response.completion_id}")
fpclient.customer_feedback.update(
    record_response.completion_id, {'is_it_good': 'nah', 'count_of_interactions': 123})
