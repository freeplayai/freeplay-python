import os

import vertexai
from vertexai.generative_models import GenerativeModel

from examples.customer_utils import get_freeplay_thin_client

fp_client = get_freeplay_thin_client()

input_variables = {'question': "Why isn't my router working"}
formatted_prompt = fp_client.prompts.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-gemini-prompt',
    environment='latest'
).bind(input_variables).format()

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

project_id = os.environ["EXAMPLES_VERTEX_PROJECT_ID"]

vertexai.init(project=project_id, location="us-central1")
model = GenerativeModel(
    model_name=formatted_prompt.prompt_info.model,
    system_instruction=formatted_prompt.system_content,
    generation_config=formatted_prompt.prompt_info.model_parameters
)

response = model.generate_content(
    formatted_prompt.llm_prompt
)

print(response.text)
