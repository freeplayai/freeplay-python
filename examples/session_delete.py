import os
import time

from anthropic import Anthropic, NotGiven

from freeplay import Freeplay, RecordPayload, ResponseInfo, CallInfo

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

input_variables = {'question': "why is the sky blue?"}
formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-anthropic-prompt',
    environment='latest',
    variables=input_variables
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

start = time.time()
completion = client.messages.create(
    system=formatted_prompt.system_content or NotGiven(),
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.content[0].text)

session = fpclient.sessions.create()
all_messages = formatted_prompt.all_messages(
    new_message={'role': 'assistant', 'content': completion.content[0].text}
)
call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
response_info = ResponseInfo(
    is_complete=completion.stop_reason == 'stop_sequence'
)

record_response = fpclient.recordings.create(
    RecordPayload(
        project_id=os.environ['FREEPLAY_PROJECT_ID'],
        all_messages=all_messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_info=formatted_prompt.prompt_info,
        call_info=call_info,
        response_info=response_info
    )
)
print(f"Recorded Session ID: {session.session_id}")

print(f"Sending customer feedback for completion id: {record_response.completion_id}")
fpclient.customer_feedback.update(
    record_response.completion_id, {'is_it_good': 'nah', 'count_of_interactions': 123})

fpclient.sessions.delete(os.environ['FREEPLAY_PROJECT_ID'], session.session_id)
print(f"Deleted Session ID: {session.session_id}")