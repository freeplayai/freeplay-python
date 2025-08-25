import os
import time

from anthropic import Anthropic, NotGiven

from freeplay import Freeplay, RecordPayload, ResponseInfo, CallInfo, UsageTokens

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

input_variables = {'question': "why is the sky blue?"}

prompt = fpclient.prompts.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-anthropic-prompt',
    environment='latest'
)

print(f"Tool Schema from simple prompt: {prompt.tool_schema}")

formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-anthropic-prompt',
    environment='latest',
    variables=input_variables
)

print(f"Tool schema: {formatted_prompt.tool_schema}")

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

start = time.time()
completion = client.messages.create(
    system=formatted_prompt.system_content or NotGiven(),
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    tools=formatted_prompt.tool_schema,
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.content[0])

session = fpclient.sessions.create()
messages = formatted_prompt.all_messages({
    'content': completion.content,
    'role': completion.role,
})
print(f"All messages: {messages}")
call_info = CallInfo.from_prompt_info(
    formatted_prompt.prompt_info,
    start,
    end,
    usage=UsageTokens(completion.usage.input_tokens, completion.usage.output_tokens)
)
response_info = ResponseInfo(
    is_complete=completion.stop_reason == 'stop_sequence'
)

print(f"Messages: {messages}")
record_response = fpclient.recordings.create(
    RecordPayload(
        project_id=os.environ['FREEPLAY_PROJECT_ID'],
        all_messages=messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
        response_info=response_info
    )
)

print(f"Sending customer feedback for completion id: {record_response.completion_id}")
fpclient.customer_feedback.update(
    record_response.completion_id, {'is_it_good': 'nah', 'count_of_interactions': 123})
