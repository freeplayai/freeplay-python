import os
import time
from uuid import uuid4

from openai import OpenAI

from customer_utils import get_freeplay_thin_client, record_results_messages
from freeplay import ResponseInfo, CallInfo, RecordPayload

fp_client = get_freeplay_thin_client()
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

project_id = os.environ['FREEPLAY_PROJECT_ID']
template_prompt = fp_client.prompts.get(
    project_id=project_id,
    template_name='toolio-openai',
    environment='latest'
)

test_run = fp_client.test_runs.create(
    project_id,
    "Tools",
    include_outputs=True,
    name=f'Test run: {uuid4()}',
    description='Run from Python examples',
    flavor_name=template_prompt.prompt_info.flavor_name
)
for test_case in test_run.test_cases:
    formatted_prompt = template_prompt.bind(test_case.variables, history=test_case.history).format()
    print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

    start = time.time()
    completion = openai_client.chat.completions.create(
        messages=formatted_prompt.llm_prompt,
        model=formatted_prompt.prompt_info.model,
        tools=formatted_prompt.tool_schema,
        **formatted_prompt.prompt_info.model_parameters
    )
    end = time.time()
    print("Completion: %s" % completion)

    session = fp_client.sessions.create()
    test_run_info = test_run.get_test_run_info(test_case.id)

    all_messages = formatted_prompt.all_messages(completion.choices[0].message)

    record_results_messages(
        fp_client,
        project_id,
        all_messages,
        test_case.variables,
        session,
        start,
        end,
        test_run_info=test_run_info,
        eval_results={
            'f1-score': 0.48,
            'is_non_empty': True
        },
        formatted_prompt=formatted_prompt
    )

    new_prompt = template_prompt.bind(test_case.variables, history=all_messages).format()

    start = time.time()
    second_completion = openai_client.chat.completions.create(
        messages=new_prompt.llm_prompt,
        model=new_prompt.prompt_info.model,
        tools=new_prompt.tool_schema,
        **new_prompt.prompt_info.model_parameters
    )
    end = time.time()
    print("Completion: %s" % second_completion.choices[0].message.content)

    final_messages = formatted_prompt.all_messages(second_completion.choices[0].message)

    call_info = CallInfo(
        formatted_prompt.prompt_info.provider,
        model=formatted_prompt.prompt_info.model,
        start_time=start,
        end_time=end,
        model_parameters=formatted_prompt.prompt_info.model_parameters)

    response_info = ResponseInfo(
        is_complete=True
    )

    fp_client.recordings.create(
        RecordPayload(
            project_id=project_id,
            all_messages=final_messages,
            session_info=session.session_info,
            inputs=test_case.variables,
            prompt_info=formatted_prompt.prompt_info,
            call_info=call_info,
            response_info=response_info,
            test_run_info=test_run_info,
            eval_results={
                'f1-score': 0.48,
                'is_non_empty': True
            }
        )
    )

# wait 5 sec and get the results
time.sleep(5)
results = fp_client.test_runs.get(project_id, test_run.test_run_id)
print(f"Test run results")
print(results.test_run_id)
print(results.name)
print(results.description)
print(results.summary_statistics)
