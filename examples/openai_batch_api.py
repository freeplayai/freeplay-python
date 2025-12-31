import json
import os
from openai import OpenAI
from freeplay import (
    Freeplay,
    RecordPayload,
    CallInfo,
)
import time

from freeplay.resources.recordings import RecordUpdatePayload


fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
project_id = os.environ["FREEPLAY_PROJECT_ID"]
environment = "latest"

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

questions = [
    "What is the capital of France?",
    "Who is the president of the United States?",
    "What is the population of Tokyo?",
    "Who was the star of the movie 'The Matrix'?",
    "What is the capital of Japan?",
    "What is the highest mountain in the world?",
    "Who is the author of 'To Kill a Mockingbird'?",
]

## Create the batch file ##
# fetch a prompt template from freeplay
prompt_template = fp_client.prompts.get(
    project_id=project_id,
    template_name="basic_trivia_bot",
    environment=environment,
)

# loop over each input and create a completion in freeplay as well as a line in the batch file
batch_file_data = []
for question in questions:
    # format the prompt with the input
    input_vars = {
        "question": question,
    }
    formatted_prompt = prompt_template.bind(input_vars).format()
    # create the completion in freeplay in order to get a completion id
    session_id = fp_client.sessions.create()
    completion_info = fp_client.recordings.create(
        RecordPayload(
            project_id=project_id,
            all_messages=formatted_prompt.messages,
            inputs=input_vars,
            session_info=session_id,
            prompt_info=prompt_template.prompt_info,
            call_info=CallInfo.from_prompt_info(
                prompt_template.prompt_info,
                start_time=time.time(),
                end_time=time.time(),
            ),  # need to figure out a good way to handle latency here since normal latency is not relevant
        )
    )
    # add a line to the batch file using the completion id as the id
    batch_file_data.append(
        {
            "custom_id": completion_info.completion_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": prompt_template.prompt_info.model,
                "messages": formatted_prompt.messages,
                **formatted_prompt.prompt_info.model_parameters,
            },
        }
    )

# write the batch file somewhere persistent
with open("batch_file.jsonl", "w") as f:
    for line in batch_file_data:
        json.dump(line, f)
        f.write("\n")

## Upload the batch file to the OpenAI API ##
batch_input_file = openai_client.files.create(
    file=open("batch_file.jsonl", "rb"), purpose="batch"
)

## Create the batch request ##
batch_request = openai_client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "nightly job"},
)
print(batch_request)

### NEW SERVICE POLLS FOR COMPLETION ###
## New service does the polling to check on the status of the batch request ##
batch_status = batch_request.status
while batch_status != "completed":
    time.sleep(10)
    batch_request = openai_client.batches.retrieve(batch_request.id)
    batch_status = batch_request.status
    print(batch_status)

## Read the batch file and update in freeplay ##
file_response = openai_client.files.content(batch_request.output_file_id)

# Do whatever you want with the response
# will write to a file for now
with open("batch_output.jsonl", "wb") as f:
    f.write(file_response.content)

# loop over each line and update the completion in freeplay
for line in file_response.text.strip().split("\n"):
    response_data = json.loads(line)
    completion_id = response_data[
        "custom_id"
    ]  # this is the completion id for the partial completion in freeplay
    output = response_data["response"]["body"]["choices"][0]["message"]["content"]
    print(completion_id, output)
    # this is new functionality to update the completion output in freeplay
    fp_client.recordings.update(
        RecordUpdatePayload(
            project_id=project_id,
            completion_id=completion_id,
            new_messages=[
                {
                    "role": "assistant",
                    "content": output,
                }
            ],
            eval_results={"is_batch_completion": True},
        )
    )
