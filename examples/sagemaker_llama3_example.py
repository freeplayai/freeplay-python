import json
import os
import time
from typing import Dict, List

import boto3

from customer_utils import get_freeplay_thin_client, record_results_from_bound


# ** NOTE **
# The keys used by the boto3 client MUST be for a service account, not a regular user account. Otherwise you need a
# session token, which is temporary. You should see this account in AWS's 'IAM' section of the console, not 'IAM
# Identity Center'. (Note the keys are picked up from the environment.) See this page for how it finds credentials:
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

def format_llama3(messages: List[Dict[str, str]]) -> str:
    if len(messages) < 1:
        raise ValueError("Must have at least one message to format")

    formatted = "<|begin_of_text|>"
    for message in messages:
        formatted += f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>"

    return formatted


fp_client = get_freeplay_thin_client()

input_variables = {'question': 'Why is the sky blue?'}
bound_prompt = fp_client.prompts.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-sagemaker-llama-3-prompt',
    environment='latest'
).bind(input_variables)

prompt = format_llama3(bound_prompt.messages)

print(f"Ready for LLM: {prompt}")

client = boto3.client(
    'sagemaker-runtime', 'us-east-1'
)

custom_attributes = ""  # An example of a trace ID.
endpoint_name = bound_prompt.prompt_info.provider_info['endpoint_name']
inference_component_name = bound_prompt.prompt_info.provider_info['inference_component_name']
content_type = "application/json"  # The MIME type of the input data in the request body.
accept = "application/json"  # The desired MIME type of the inference in the response.
payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 64,
        # "top_p": 0.9,
        # "temperature": 0.6
    }
}

payload_str = json.dumps(payload)

start = time.time()
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    InferenceComponentName=inference_component_name,
    CustomAttributes=custom_attributes,
    ContentType=content_type,
    Accept=accept,
    Body=json.dumps(payload)
)
end = time.time()

json = json.loads(response['Body'].read().decode("utf-8"))
response_content = json['generated_text']
print(response_content)

record_results_from_bound(
    fp_client,
    bound_prompt.prompt_info,
    bound_prompt.messages,
    response_content,
    input_variables,
    fp_client.sessions.create(),
    start,
    end
)
