import os
import time

from anthropic import Anthropic

from src.freeplay.freeplay_thin import FreeplayThin, RecordPayload, ResponseInfo

fpclient = FreeplayThin(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

input_variables = {'question': "Why isn't my door working?"}
formatted_prompt = fpclient.get_formatted_prompt(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-prompt-anthropic',
    environment='prod',
    variables=input_variables
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

start = time.time()
completion = client.completions.create(
    model=formatted_prompt.prompt_info.model,
    prompt=formatted_prompt.llm_prompt,
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.completion)

session = fpclient.create_session()
all_messages = formatted_prompt.all_messages(
    new_message={'role': 'Assistant', 'content': completion.completion}
)
call_info = formatted_prompt.prompt_info.get_call_info(start, end)
response_info = ResponseInfo(
    is_complete=completion.stop_reason == 'stop_sequence'
)

fpclient.record_call(
    RecordPayload(
        all_messages=all_messages,
        session_id=session.session_id,
        inputs=input_variables,
        prompt_info=formatted_prompt.prompt_info,
        call_info=call_info,
        response_info=response_info
    )
)
