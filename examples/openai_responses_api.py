import os
import time

from freeplay import Freeplay, RecordPayload, CallInfo
from freeplay.resources.recordings import UsageTokens
from openai import OpenAI
from openai.types.responses import WebSearchToolParam

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {
    "question": "search the internet and tell me about Freeplay's latest funding round"
}

project_id = os.environ["FREEPLAY_PROJECT_ID"]

formatted_prompt = fp_client.prompts.get_formatted(
    project_id=project_id,
    template_name="witty-question",
    environment="latest",
    variables=input_variables,
)

start = time.time()
completion = openai_client.responses.create(
    input=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    include=["code_interpreter_call.outputs"],
    tools=[WebSearchToolParam(type="web_search_preview")],
    # TODO: Tool schema from prompt can't be used -- format has changed from chat completions API...
    # FIX => format tool schema for responses API. Likely need a new flavor for the Openai Responses API
    # tools=formatted_prompt.tool_schema,
    **formatted_prompt.prompt_info.model_parameters,
)
end = time.time()
print("Completion: %s" % completion)

session = fp_client.sessions.create()
# TODO: Rough edge: requires constructing a message format from text. This would drop tool calls, etc.
# Fix => We could update our record payload to accept these messages/tool calls, etc.
out_msg = {"role": "assistant", "content": completion.output_text}

messages = formatted_prompt.all_messages(out_msg)
print(f"All messages: {messages}")
call_info = CallInfo.from_prompt_info(
    formatted_prompt.prompt_info,
    start,
    end,
    UsageTokens(completion.usage.input_tokens, completion.usage.output_tokens),
    api_style="batch",
)
print(f"Messages: {messages}")
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

print(f"Sending customer feedback for completion id: {record_response.completion_id}")
fp_client.customer_feedback.update(
    project_id,
    record_response.completion_id,
    {"is_it_good": "nah", "count_of_interactions": 123},
)
