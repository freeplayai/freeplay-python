import os
import time
from pathlib import Path

from anthropic import Anthropic

from customer_utils import record_results, format_anthropic_messages
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
system_message_content, other_messages = format_anthropic_messages(formatted_prompt)
completion = client.messages.create(
    system=system_message_content,
    messages=other_messages,
    model=formatted_prompt.prompt_info.model,
    **formatted_prompt.prompt_info.model_parameters
)
end = time.time()
print("Completion: %s" % completion.content[0].text)

session = fp_client.sessions.create()

record_results(
    fp_client,
    formatted_prompt,
    completion.content[0].text,
    input_variables,
    session,
    start,
    end
)
