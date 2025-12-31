import os
import time

from openai import OpenAI

from freeplay import Freeplay, CallInfo, ResponseInfo, RecordPayload
from freeplay.resources.prompts import MediaInputUrl
from freeplay.resources.test_cases import DatasetTestCase

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {"question": "Describe what's in this image"}

image_url = "https://images.pexels.com/photos/30614903/pexels-photo-30614903/free-photo-of-aerial-view-of-bilbao-city-and-guggenheim-museum.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

media_inputs = {"some-image": MediaInputUrl(type="url", url=image_url)}
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name="media",
    environment="latest",
    variables=input_variables,
    media_inputs=media_inputs,
)

start = time.time()
completion = openai_client.chat.completions.create(
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    **formatted_prompt.prompt_info.model_parameters,
)
end = time.time()

response_content = completion.choices[0].message.content
print("Completion:", response_content)

session = fp_client.sessions.create()
call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
response_info = ResponseInfo(is_complete=completion.choices[0].finish_reason == "stop")

record_response = fp_client.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=[
            *formatted_prompt.llm_prompt,
            {"role": "assistant", "content": response_content},
        ],
        session_info=session.session_info,
        inputs=input_variables,
        media_inputs=media_inputs,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
        response_info=response_info,
    )
)

fp_client.test_cases.create_many(
    os.environ["FREEPLAY_PROJECT_ID"],
    os.environ["FREEPLAY_DATASET_ID"],
    [DatasetTestCase(input_variables, response_content, [], {}, media_inputs)],
)

print(f"Recorded completion ID: {record_response.completion_id}")
