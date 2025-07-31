import os
import time

from openai import OpenAI

from freeplay.model import OpenAIFunctionCall
from freeplay import Freeplay, RecordPayload, ResponseInfo, CallInfo

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

input_variables = {"pop_star": "Bruno Mars"}
formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='album_bot',
    environment='latest',
    variables=input_variables
)

openai_function_definition = {
    "name": "get_album_tracklist",
    "description": "Given an album name and genre, return a list of songs.",
    "parameters": {
        "type": "object",
        "properties": {
            "album_name": {
                "type": "string",
                "description": "Name of album from which to retrieve tracklist."
            },
            "genre": {
                "type": "string",
                "description": "Album genre"
            }
        }
    }
}

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

start = time.time()

completion = client.chat.completions.create(  # type: ignore
    model=formatted_prompt.prompt_info.model,
    messages=formatted_prompt.messages,
    functions=[
        openai_function_definition
    ],
    function_call={"name": "get_album_tracklist"},
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.choices[0].message)

session = fpclient.sessions.create(
    custom_metadata={
        'metadata_key': 'metadata_value'
    }
)
all_messages = formatted_prompt.all_messages(
    new_message={'role': 'assistant', 'content': ""}
)
call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)

# Get function call from completion
function_call_response = None
if completion.choices[0].message.function_call:
    function_call_response = OpenAIFunctionCall(
        name=completion.choices[0].message.function_call.name,
        arguments=completion.choices[0].message.function_call.arguments
    )

response_info = ResponseInfo(
    is_complete=completion.choices[0].finish_reason == 'stop',
    function_call_response=function_call_response
)

fpclient.recordings.create(
    RecordPayload(
        project_id=os.environ['FREEPLAY_PROJECT_ID'],
        all_messages=all_messages,
        inputs=input_variables,
        session_info=session.session_info,
        prompt_info=formatted_prompt.prompt_info,
        call_info=call_info,
        response_info=response_info
    )
)
