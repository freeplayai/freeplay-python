import base64
import os
import time

import requests
from openai import OpenAI

from freeplay import Freeplay, CallInfo, ResponseInfo, RecordPayload
from freeplay.resources.prompts import MediaInputBase64

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

input_variables = {"query": "Describe this document"}

response = requests.get("https://arxiv.org/pdf/1706.03762")
response.raise_for_status()
encoded_pdf = base64.b64encode(response.content).decode("utf-8")

media_inputs = {
    "some-file": MediaInputBase64(
        type="base64", content_type="application/pdf", data=encoded_pdf
    )
}
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name="pdf",
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

print(f"Recorded completion ID: {record_response.completion_id}")
