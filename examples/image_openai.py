import os
import time

from openai import OpenAI

from freeplay import CallInfo, Freeplay, RecordPayload, ResponseInfo

fpclient = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {"question": "What's in this image?"}

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"

formatted_prompt = fpclient.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name="media",
    environment="latest",
    variables=input_variables,
    history=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}
            ],
        }
    ],
)

start = time.time()
completion = client.chat.completions.create(
    messages=formatted_prompt.llm_prompt,
    model=formatted_prompt.prompt_info.model,
    **formatted_prompt.prompt_info.model_parameters,
)
end = time.time()

print("Completion:", completion.choices[0].message.content)

session = fpclient.sessions.create()
all_messages = formatted_prompt.all_messages(completion.choices[0].message)
call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
response_info = ResponseInfo(is_complete=completion.choices[0].finish_reason == "stop")

record_response = fpclient.recordings.create(
    RecordPayload(
        project_id=os.environ["FREEPLAY_PROJECT_ID"],
        all_messages=all_messages,
        session_info=session.session_info,
        inputs=input_variables,
        prompt_version_info=formatted_prompt.prompt_info,
        call_info=call_info,
        tool_schema=formatted_prompt.tool_schema,
        response_info=response_info,
    )
)

print(f"Recorded completion ID: {record_response.completion_id}")
