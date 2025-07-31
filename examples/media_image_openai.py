import os
import time

from openai import OpenAI

from freeplay import Freeplay, CallInfo, ResponseInfo, RecordPayload
from freeplay.resources.prompts import MediaInputUrl

fpclient = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {"question": "Describe what's in this image"}

image_url = "https://images.pexels.com/photos/30614903/pexels-photo-30614903/free-photo-of-aerial-view-of-bilbao-city-and-guggenheim-museum.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

media_inputs = {
    'some-image': MediaInputUrl(
        type="url",
        url=image_url
    )
}
formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name="media",
    environment="latest",
    variables=input_variables,
    media_inputs=media_inputs
)

start = time.time()
completion = client.chat.completions.create(
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    **formatted_prompt.prompt_info.model_parameters,
)
end = time.time()

response_content = completion.choices[0].message.content
print("Completion:", response_content)

session = fpclient.sessions.create()
call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
response_info = ResponseInfo(is_complete=completion.choices[0].finish_reason == "stop")

record_response = fpclient.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=[
            *formatted_prompt.llm_prompt,
            {"role": "assistant", "content": response_content}
        ],
        session_info=session.session_info,
        inputs=input_variables,
        media_inputs=media_inputs,
        prompt_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
        response_info=response_info,
    )
)

print(f"Recorded completion ID: {record_response.completion_id}")
