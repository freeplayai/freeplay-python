import os
import time

from anthropic import NotGiven
from anthropic.lib.bedrock import AnthropicBedrock

from customer_utils import get_freeplay_thin_client, record_results

# ** NOTE **
# The keys used below MUST be for a service account, not a regular user account. Otherwise you need a session token,
# which is temporary. You should see this account in AWS's 'IAM' section of the console, not 'IAM Identity Center'.

# Setup guide (including AWS credentials): https://docs.anthropic.com/claude/reference/claude-on-amazon-bedrock

# These live at ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

fp_client = get_freeplay_thin_client()

input_variables = {"question": "why is the sky blue?"}
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name="my-anthropic-prompt",
    environment="latest",
    variables=input_variables,
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

client = AnthropicBedrock(
    aws_access_key=AWS_ACCESS_KEY_ID,
    aws_secret_key=AWS_SECRET_ACCESS_KEY,
    aws_region="us-east-1",
)
bedrock_claude_model_name = "anthropic.claude-v2:1"

start = time.time()
response = client.messages.create(
    model=bedrock_claude_model_name,
    system=formatted_prompt.system_content or NotGiven(),
    messages=formatted_prompt.llm_prompt,
    **formatted_prompt.prompt_info.model_parameters,
)
end = time.time()

print(response.content)
record_results(
    fp_client,
    formatted_prompt,
    response.content[0].text,
    input_variables,
    fp_client.sessions.create(),
    start,
    end,
)
