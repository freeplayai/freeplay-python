"""
Multi-turn conversation with media_inputs.

This example:
1. Records Turn 1 with media_inputs (using media slot in template)
2. Adds response to history
3. Records Turn 2 asking for more details with history (media now in history)
"""

import copy
import os
import time

from openai import OpenAI

from freeplay import CallInfo, Freeplay, RecordPayload, ResponseInfo
from freeplay.model import MediaInputMap, MediaInputUrl

fpclient = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

project_id = os.environ["FREEPLAY_PROJECT_ID"]

# Use a real image URL
image_url = "https://images.pexels.com/photos/30614903/pexels-photo-30614903/free-photo-of-aerial-view-of-bilbao-city-and-guggenheim-museum.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

print("=" * 80)
print("TURN 1: Recording with media_inputs")
print("=" * 80)

# Create session for multi-turn conversation
session = fpclient.sessions.create()
print(f"Created session ID: {session.session_info.session_id}")

# Turn 1: User provides image via media_inputs
input_variables_1 = {"query": "What do you see in this image?"}
media_inputs_1: MediaInputMap = {
    "media_slot_1": MediaInputUrl(type="url", url=image_url)
}
environment = os.environ.get("FREEPLAY_ENVIRONMENT", "latest")
template_name = os.environ.get("FREEPLAY_PROMPT_TEMPLATE_NAME", "image_generation")

formatted_prompt_1 = fpclient.prompts.get_formatted(
    project_id=project_id,
    template_name=template_name,
    environment=environment,
    variables=input_variables_1,
    media_inputs=media_inputs_1,
    history=[],
)

# Start history from llm_prompt
history = copy.deepcopy(formatted_prompt_1.llm_prompt)

print(f"Turn 1: {len(history)} messages with media_inputs")

# Call OpenAI for Turn 1
# Filter parameters to only those supported by the provider/model

start_1 = time.time()
completion_1 = openai_client.chat.completions.create(
    messages=history,
    model=formatted_prompt_1.prompt_info.model,
    **formatted_prompt_1.prompt_info.model_parameters,
)
end_1 = time.time()

assistant_response_1 = {
    "role": "assistant",
    "content": completion_1.choices[0].message.content,
}

print(f"Assistant: {assistant_response_1['content']}")

# Add response to history
history.append(assistant_response_1)

# Record Turn 1 with media_inputs
call_info_1 = CallInfo.from_prompt_info(formatted_prompt_1.prompt_info, start_1, end_1)
response_info_1 = ResponseInfo(
    is_complete=completion_1.choices[0].finish_reason == "stop"
)

record_response_1 = fpclient.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=history,
        session_info=session.session_info,
        inputs=input_variables_1,
        media_inputs=media_inputs_1,
        prompt_version_info=formatted_prompt_1.prompt_info,
        call_info=call_info_1,
        response_info=response_info_1,
    )
)

print(f"Recorded Turn 1 completion ID: {record_response_1.completion_id}")

# Turn 2: User asks for more details about the same image (now in history)
print("\n" + "=" * 80)
print("TURN 2: Recording follow-up asking for more details")
print("=" * 80)

input_variables_2 = {
    "query": "Tell me more about the architectural features of the building in the image."
}

# Add Turn 2 user message to history
history.append({"role": "user", "content": input_variables_2["query"]})

print(f"Turn 2: {len(history)} messages (includes Turn 1 with media in history)")

# Call OpenAI for Turn 2
# Filter parameters to only those supported by the provider/model

start_2 = time.time()
completion_2 = openai_client.chat.completions.create(
    messages=history,
    model=formatted_prompt_1.prompt_info.model,
    **formatted_prompt_1.prompt_info.model_parameters,
)
end_2 = time.time()

assistant_response_2 = {
    "role": "assistant",
    "content": completion_2.choices[0].message.content,
}

print(f"Assistant: {assistant_response_2['content'][:100]}...")

# Add response to history
history.append(assistant_response_2)

# Record Turn 2 - media is now in history, not media_inputs
call_info_2 = CallInfo.from_prompt_info(formatted_prompt_1.prompt_info, start_2, end_2)
response_info_2 = ResponseInfo(
    is_complete=completion_2.choices[0].finish_reason == "stop"
)

record_response_2 = fpclient.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=history,
        session_info=session.session_info,
        inputs=input_variables_2,
        prompt_version_info=formatted_prompt_1.prompt_info,
        call_info=call_info_2,
        response_info=response_info_2,
    )
)

print(f"Recorded Turn 2 completion ID: {record_response_2.completion_id}")

print("\n" + "=" * 80)
print("Multi-turn recording complete!")
print(f"Session ID: {session.session_info.session_id}")
print(
    f"View in UI: {os.environ.get('FREEPLAY_API_URL', 'http://localhost:8080')}/projects/{project_id}/sessions"
)
print("=" * 80)
