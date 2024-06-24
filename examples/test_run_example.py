import os
import time
from uuid import uuid4

from anthropic import Anthropic, NotGiven

from customer_utils import get_freeplay_thin_client, record_results, format_anthropic_messages

fp_client = get_freeplay_thin_client()
anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

project_id = os.environ['FREEPLAY_PROJECT_ID']
template_prompt = fp_client.prompts.get(
    project_id=project_id,
    template_name='my-prompt-anthropic',
    environment='prod'
)

test_run = fp_client.test_runs.create(
    project_id, "core-tests", name=f'Test run: {uuid4()}', description='Run from Python examples')
for test_case in test_run.test_cases:
    formatted_prompt = template_prompt.bind(test_case.variables).format()
    print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

    system_message_content, other_messages = format_anthropic_messages(formatted_prompt)
    start = time.time()
    completion = anthropic_client.messages.create(
        system=formatted_prompt.system_content or NotGiven(),
        messages=formatted_prompt.llm_prompt,
        model=formatted_prompt.prompt_info.model,
        **formatted_prompt.prompt_info.model_parameters
    )
    end = time.time()
    print("Completion: %s" % completion.content[0].text)

    session = fp_client.sessions.create()
    test_run_info = test_run.get_test_run_info(test_case.id)
    record_results(
        fp_client,
        formatted_prompt,
        completion.content[0].text,
        test_case.variables,
        session,
        start,
        end,
        test_run_info,
        {
            'f1-score': 0.48,
            'is_non_empty': True
        }
    )


# wait 5 sec and get the results
time.sleep(5)
results = fp_client.test_runs.get(project_id, test_run.test_run_id)
print(f"Test run results")
print(results.test_run_id)
print(results.name)
print(results.description)
print(results.summary_statistics)
