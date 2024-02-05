import os
import time
from pathlib import Path

from anthropic import Anthropic

from freeplay.thin import Freeplay, RecordPayload, ResponseInfo, CallInfo
from freeplay.thin.resources.prompts import FilesystemTemplateResolver

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
    template_resolver=FilesystemTemplateResolver(Path(os.environ['FREEPLAY_TEMPLATE_DIRECTORY']))
)
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
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
completion = client.completions.create(
    model=formatted_prompt.prompt_info.model,
    prompt=str(formatted_prompt.llm_prompt),
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.completion)

session = fpclient.sessions.create()
all_messages = formatted_prompt.all_messages(
    new_message={'role': 'Assistant', 'content': completion.completion}
)
call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
response_info = ResponseInfo(
    is_complete=completion.stop_reason == 'stop_sequence'
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
