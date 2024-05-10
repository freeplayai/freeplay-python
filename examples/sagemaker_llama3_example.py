import json
import os
import time

import boto3

from customer_utils import get_freeplay_thin_client, record_results_from_bound

# ** NOTE **
# The keys used by the boto3 client MUST be for a service account, not a regular user account. Otherwise you need a
# session token, which is temporary. You should see this account in AWS's 'IAM' section of the console, not 'IAM
# Identity Center'. (Note the keys are picked up from the environment.) See this page for how it finds credentials:
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

fp_client = get_freeplay_thin_client()

input_variables = {'question': 'Why is the sky blue?'}
formatted_prompt = fp_client.prompts.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-sagemaker-llama-3-prompt',
    environment='latest'
).bind(input_variables).format()

print(f"Ready for LLM: {formatted_prompt.llm_prompt_text}")

client = boto3.client(
    'sagemaker-runtime', 'us-east-1'
)

custom_attributes = ""  # An example of a trace ID.
endpoint_name = formatted_prompt.prompt_info.provider_info['endpoint_name']
inference_component_name = formatted_prompt.prompt_info.provider_info['inference_component_name']
content_type = "application/json"  # The MIME type of the input data in the request body.
accept = "application/json"  # The desired MIME type of the inference in the response.
payload = {
    "inputs": formatted_prompt.llm_prompt_text,
    "parameters": {
        **formatted_prompt.prompt_info.model_parameters
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
    formatted_prompt.prompt_info,
    formatted_prompt.messages,
    response_content,
    input_variables,
    fp_client.sessions.create(),
    start,
    end
)
