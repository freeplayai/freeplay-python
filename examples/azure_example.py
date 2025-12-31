import os
import time
from typing import cast, List

import openai
from openai.types.chat import ChatCompletionMessageParam

from customer_utils import get_freeplay_thin_client, record_results

API_VERSION_STRING = "2024-02-15-preview"

fp_client = get_freeplay_thin_client()

input_variables = {"input": "Why isn't my door working?"}
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=os.environ["FREEPLAY_PROJECT_ID"],
    template_name="my-chat-template-azure",
    environment="latest",
    variables=input_variables,
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

client = openai.AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    api_version=API_VERSION_STRING,
    **formatted_prompt.prompt_info.provider_info,
)

start = time.time()
completion = client.chat.completions.create(
    model=formatted_prompt.prompt_info.model,
    messages=cast(List[ChatCompletionMessageParam], formatted_prompt.llm_prompt),
    **formatted_prompt.prompt_info.model_parameters,
)
end = time.time()
print("Completion: %s" % completion.choices[0].message.content)

session = fp_client.sessions.create()

record_results(
    fp_client,
    formatted_prompt,
    completion.choices[0].message.content,
    input_variables,
    session.session_info,
    start,
    end,
)
