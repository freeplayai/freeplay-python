import os

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

from freeplay import Freeplay, RecordPayload, CallInfo
from freeplay.utils import convert_provider_message_to_dict

# Initialize Freeplay and get formatted prompt
fp_client = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)

input_variables = {'location': "Boulder"}
formatted_prompt = fp_client.prompts.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-openai-prompt',
    environment='latest'
).bind(input_variables, history=[]).format()

# Initialize Vertex AI
vertexai.init(project="fp-d-int-069c", location="us-central1")

model = GenerativeModel(
    model_name=formatted_prompt.prompt_info.model,
    system_instruction=formatted_prompt.system_content,
    generation_config=formatted_prompt.prompt_info.model_parameters,
    tools=formatted_prompt.tool_schema
)

# Start a chat session for function calling
chat = model.start_chat()

# Extract the user message from Freeplay's formatted prompt
initial_message = formatted_prompt.llm_prompt[0]['parts'][0]['text']
response = chat.send_message(initial_message)

# Handle the response
content = response.candidates[0].content

# Check if the model wants to call a function
if content.parts[0].function_call:
    function_call = content.parts[0].function_call
    
    # Execute your function (replace with your actual function logic)
    function_result = {"weather": "sunny", "temperature": 72}
    
    # Send function result back to the model
    function_response_part = Part.from_function_response(
        name=function_call.name,
        response=function_result
    )
    function_response = chat.send_message(function_response_part)
    
    # Build complete message history for recording
    all_messages = list(formatted_prompt.llm_prompt)  # Start with initial messages
    all_messages.append(content)
    all_messages.append(Content(
        role="user",
        parts=[function_response_part]
    ))
    all_messages.append(function_response.candidates[0].content)
else:
    # For non-function-call responses
    all_messages = formatted_prompt.all_messages(content)

# Convert all messages to dicts for recording
all_messages_dict = [convert_provider_message_to_dict(msg) for msg in all_messages]

fp_client.recordings.create(
    RecordPayload(
        project_id=os.environ['FREEPLAY_PROJECT_ID'],
        all_messages=all_messages_dict,
        inputs=input_variables,
        prompt_info=formatted_prompt.prompt_info,
        tool_schema=formatted_prompt.tool_schema,
        call_info=CallInfo(
            provider="vertex",
            model=formatted_prompt.prompt_info.model,
        )
    )
)