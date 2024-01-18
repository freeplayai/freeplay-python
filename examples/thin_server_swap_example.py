import os
import time
from typing import List, cast

from anthropic import Anthropic
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from freeplay.thin import Freeplay, ResponseInfo, RecordPayload

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

input_variables = {'question': "Why isn't my door working?"}
formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-prompt-anthropic',
    environment='prod',
    variables=input_variables
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

start = time.time()
if formatted_prompt.prompt_info.provider == 'anthropic':
    completion = anthropic_client.completions.create(
        model=formatted_prompt.prompt_info.model,
        prompt=str(formatted_prompt.llm_prompt),
        **formatted_prompt.prompt_info.model_parameters
    )
    completion_text = completion.completion
    is_complete = completion.stop_reason == 'stop_sequence'
elif formatted_prompt.prompt_info.provider == 'openai':
    chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=cast(List[ChatCompletionMessageParam], formatted_prompt.llm_prompt)
    )
    completion_text = chat_completion.choices[0].message.content or ''
    is_complete = chat_completion.choices[0].finish_reason == 'stop'
else:
    raise ValueError("Unknown provider: %s" % formatted_prompt.prompt_info.provider)
end = time.time()

print(completion_text)

session = fpclient.sessions.create()
all_messages = formatted_prompt.all_messages(
    new_message={'role': 'Assistant', 'content': completion_text}
)
call_info = formatted_prompt.prompt_info.get_call_info(start, end)
response_info = ResponseInfo(is_complete)

fpclient.recordings.create(
    RecordPayload(
        all_messages=all_messages,
        session_id=session.session_id,
        inputs=input_variables,
        prompt_info=formatted_prompt.prompt_info,
        call_info=call_info,
        response_info=response_info
    )
)
