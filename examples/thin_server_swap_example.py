import os
import time
from typing import List, cast

from anthropic import Anthropic
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from examples.customer_utils import record_results, get_freeplay_thin_client

fp_client = get_freeplay_thin_client()
anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

input_variables = {'question': "Why isn't my door working?"}
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    template_name='my-anthropic-prompt',
    environment='latest',
    variables=input_variables
)

print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

start = time.time()
if formatted_prompt.prompt_info.provider == 'anthropic':
    completion = anthropic_client.completions.create(
        model=formatted_prompt.prompt_info.model,
        prompt=str(formatted_prompt.llm_prompt),
        **formatted_prompt.prompt_info.model_parameters
    )
    completion_text = completion.completion
    is_complete = completion.stop_reason == 'stop_sequence'
elif formatted_prompt.prompt_info.provider == 'openai':
    chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=cast(List[ChatCompletionMessageParam], formatted_prompt.llm_prompt)
    )
    completion_text = chat_completion.choices[0].message.content or ''
    is_complete = chat_completion.choices[0].finish_reason == 'stop'
else:
    raise ValueError("Unknown provider: %s" % formatted_prompt.prompt_info.provider)
end = time.time()

print(completion_text)
session = fp_client.sessions.create()
record_results(
    fp_client,
    formatted_prompt,
    completion_text,
    input_variables,
    session,
    start,
    end
)
