import os
import time
from typing import cast
from uuid import uuid4

from customer_utils import get_freeplay_thin_client
from openai import OpenAI

fp_client = get_freeplay_thin_client()
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

project_id = os.environ["FREEPLAY_PROJECT_ID"]
template_prompt = fp_client.prompts.get(
    project_id=project_id, template_name="media-prompt", environment="latest"
)

test_run = fp_client.test_runs.create(
    project_id,
    "media-1",
    include_outputs=True,
    name=f"Media Test Run: {uuid4()}",
    description="Test run with media inputs from Python SDK",
    flavor_name=template_prompt.prompt_info.flavor_name,
)

for test_case in test_run.test_cases:
    print(f"Processing test case: {test_case.id}")

    # Get media inputs from test case
    media_inputs = test_case.media_variables

    formatted_prompt = template_prompt.bind(
        test_case.variables, history=test_case.history, media_inputs=media_inputs
    ).format()

    print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

    start = time.time()
    completion = openai_client.chat.completions.create(
        messages=formatted_prompt.llm_prompt,
        model=formatted_prompt.prompt_info.model,
        **formatted_prompt.prompt_info.model_parameters,
    )
    end = time.time()

    response_content = cast(str, completion.choices[0].message.content)
    print(f"Completion: {response_content}")

    session = fp_client.sessions.create()
    test_run_info = test_run.get_test_run_info(test_case.id)

    # Manually construct messages to avoid JSON serialization issues with media
    all_messages = [
        *formatted_prompt.llm_prompt,
        {"role": "assistant", "content": response_content},
    ]

    # Use the manual recording approach instead of record_results_messages helper
    # since media inputs require special handling
    from freeplay import CallInfo, RecordPayload, ResponseInfo

    call_info = CallInfo(
        formatted_prompt.prompt_info.provider,
        model=formatted_prompt.prompt_info.model,
        start_time=start,
        end_time=end,
        model_parameters=formatted_prompt.prompt_info.model_parameters,
    )

    response_info = ResponseInfo(
        is_complete=completion.choices[0].finish_reason == "stop"
    )

    fp_client.recordings.create(
        RecordPayload(
            project_id=project_id,
            all_messages=all_messages,
            session_info=session.session_info,
            inputs=test_case.variables,
            media_inputs=media_inputs,
            prompt_info=formatted_prompt.prompt_info,
            call_info=call_info,
            response_info=response_info,
            test_run_info=test_run_info,
            eval_results={
                "has_media": media_inputs is not None,
                "media_count": len(media_inputs or {}),
                "response_length": len(response_content) if response_content else 0,
            },
        )
    )

# wait 5 sec and get the results
time.sleep(5)
results = fp_client.test_runs.get(project_id, test_run.test_run_id)
print(results.test_run_id)
print(results.name)
print(results.description)
print(results.summary_statistics)
