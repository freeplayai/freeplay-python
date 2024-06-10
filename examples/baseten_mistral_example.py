import os
import time

from openai import OpenAI

from customer_utils import get_freeplay_thin_client, record_results_from_bound

fp_client = get_freeplay_thin_client()

input_variables = {'question': 'Why is the sky blue?'}
formatted_prompt = fp_client.prompts.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-baseten-mistral-prompt',
    environment='latest'
).bind(input_variables).format()

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

model_id = os.environ["BASETEN_MODEL_ID"]
client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/{model_id}/v1"
)

start = time.time()
# Call model endpoint
res = client.chat.completions.create(
    model=formatted_prompt.prompt_info.model,
    messages=formatted_prompt.llm_prompt,
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()

response_content = res.choices[0].message.content
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
