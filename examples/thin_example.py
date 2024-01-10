import os
import time

from anthropic import Anthropic

from freeplay.freeplay_thin import FreeplayThin, RecordPayload, CallInfo, ResponseInfo

fpclient = FreeplayThin(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)

prompt_info, messages = fpclient.get_prompt(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-prompt-anthropic',
    environment='prod',
    variables={'question': "Why isn't my door working?"}
)

ready_for_llm = str(fpclient.format(prompt_info.flavor_name, messages))
print(f"Ready for LLM: {ready_for_llm}")

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

start = time.time()
completion = client.completions.create(
    model=prompt_info.model,
    prompt=ready_for_llm,
    **prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.completion)

messages.append({'role': 'Assistant', 'content': completion.completion})

session = fpclient.create_session()
call_info = CallInfo(
    provider=prompt_info.provider,
    model=prompt_info.model,
    start_time=start,
    end_time=end,
    model_parameters=prompt_info.model_parameters
)
response_info = ResponseInfo(
    is_complete=completion.stop_reason == 'stop_sequence'
)

fpclient.record_call(
    RecordPayload(
        all_messages=messages,
        session_id=session.session_id,
        prompt_info=prompt_info,
        call_info=call_info,
        response_info=response_info
    )
)
