import os
import time
from pathlib import Path

from anthropic import Anthropic

from customer_utils import record_results
from freeplay import Freeplay
from freeplay.resources.prompts import FilesystemTemplateResolver

fp_client = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
    template_resolver=FilesystemTemplateResolver(Path(os.environ['FREEPLAY_TEMPLATE_DIRECTORY']))
)
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

input_variables = {'question': "Why isn't my door working?"}
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-prompt-anthropic',
    environment='prod',
    variables=input_variables
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

start = time.time()
completion = client.completions.create(
    model=formatted_prompt.prompt_info.model,
    prompt=str(formatted_prompt.llm_prompt),
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.completion)

session = fp_client.sessions.create()

record_results(
    fp_client,
    formatted_prompt,
    completion.completion,
    input_variables,
    session,
    start,
    end
)
