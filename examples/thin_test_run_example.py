import os
import time

from anthropic import Anthropic

from src.freeplay.freeplay_thin import FreeplayThin, RecordPayload, ResponseInfo

fpclient = FreeplayThin(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

project_id = os.environ['FREEPLAY_PROJECT_ID']
template_prompt = fpclient.get_prompt(
    project_id=project_id,
    template_name='my-prompt-anthropic',
    environment='prod'
)

test_run = fpclient.create_test_run(project_id, "core-tests")
for variables in test_run.inputs:
    formatted_prompt = template_prompt.bind(variables).format()
    print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

    start = time.time()
    completion = anthropic_client.completions.create(
        model=formatted_prompt.prompt_info.model,
        prompt=formatted_prompt.llm_prompt,
        **formatted_prompt.prompt_info.model_parameters
    )
    end = time.time()
    print("Completion: %s" % completion.completion)

    all_messages = formatted_prompt.all_messages(
        {'role': 'Assistant', 'content': completion.completion}
    )

    session = fpclient.create_session()
    call_info = formatted_prompt.prompt_info.get_call_info(start, end, test_run.test_run_id)
    # Anthropic-specific
    response_info = ResponseInfo(
        is_complete=completion.stop_reason == 'stop_sequence'
    )

    fpclient.record_call(
        RecordPayload(
            all_messages=all_messages,
            inputs=variables,
            session_id=session.session_id,
            prompt_info=formatted_prompt.prompt_info,
            call_info=call_info,
            response_info=response_info,
        )
    )
