import os
import time
from typing import cast, List

import openai
from openai.types.chat import ChatCompletionMessageParam

from freeplay.thin import Freeplay, RecordPayload, ResponseInfo, CallInfo

API_VERSION_STRING = '2024-02-15-preview'

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)

input_variables = {'input': "Why isn't my door working?"}
formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-chat-template-azure',
    environment='latest',
    variables=input_variables
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

client = openai.AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    api_version=API_VERSION_STRING,
    **formatted_prompt.prompt_info.provider_info,
)

start = time.time()
completion = client.chat.completions.create(
    model=formatted_prompt.prompt_info.model,
    messages=cast(List[ChatCompletionMessageParam], formatted_prompt.llm_prompt),
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.choices[0].message.content)

session = fpclient.sessions.create()
all_messages = formatted_prompt.all_messages(
    new_message={'role': 'Assistant', 'content': completion.choices[0].message.content}
)
call_info = CallInfo(
    formatted_prompt.prompt_info.provider,
    model=formatted_prompt.prompt_info.model,
    start_time=start,
    end_time=end,
    model_parameters=formatted_prompt.prompt_info.model_parameters)

response_info = ResponseInfo(
    is_complete=True
)

record_response = fpclient.recordings.create(
    RecordPayload(
        all_messages=all_messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_info=formatted_prompt.prompt_info,
        call_info=call_info,
        response_info=response_info
    )
)
