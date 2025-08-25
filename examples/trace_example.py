import os
import random
import time
from typing import Optional

from anthropic import Anthropic, NotGiven

from freeplay import (
    CallInfo,
    Freeplay,
    RecordPayload,
    ResponseInfo,
    SessionInfo,
    TraceInfo,
)

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)
project_id = os.environ['FREEPLAY_PROJECT_ID']

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


def call_and_record(
        project_id: str,
        template_name: str,
        env: str,
        input_variables: dict,
        session_info: SessionInfo,
        trace_info: Optional[TraceInfo] = None
) -> dict:
    formatted_prompt = fpclient.prompts.get_formatted(
        project_id=project_id,
        template_name=template_name,
        environment=env,
        variables=input_variables
    )

    print(f"Ready for LLM: {formatted_prompt.llm_prompt}")

    start = time.time()
    completion = client.messages.create(
        system=formatted_prompt.system_content or NotGiven(),
        messages=formatted_prompt.llm_prompt,
        model=formatted_prompt.prompt_info.model,
        **formatted_prompt.prompt_info.model_parameters
    )
    end = time.time()

    llm_response = completion.content[0].text
    print("Completion: %s" % llm_response)

    all_messages = formatted_prompt.all_messages(
        new_message={'role': 'assistant', 'content': llm_response}
    )
    call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
    response_info = ResponseInfo(
        is_complete=completion.stop_reason == 'stop_sequence'
    )

    record_response = fpclient.recordings.create(
        RecordPayload(
            project_id=project_id,
            all_messages=all_messages,
            session_info=session_info,
            inputs=input_variables,
            prompt_version_info=formatted_prompt.prompt_info,
            call_info=call_info,
            response_info=response_info,
            trace_info=trace_info,
        )
    )

    return {'completion_id': record_response.completion_id, 'llm_response': llm_response}


# send 3 questions to the model encapsulated into a trace
user_questions = ["answer life's most existential questions", "what is sand?", "how tall are lions?"]

session = fpclient.sessions.create({"metadata_123": "blah"})
for question in user_questions:
    trace_info = session.create_trace(agent_name="mr-secret-agent", input=question, custom_metadata={"metadata_key": "hello"})
    bot_response = call_and_record(
        project_id=project_id,
        template_name='my-anthropic-prompt',
        env='latest',
        input_variables={'question': question},
        session_info=session.session_info,
        trace_info=trace_info
    )
    categorization_result = call_and_record(
        project_id=project_id,
        template_name='question-classifier',
        env='latest',
        input_variables={'question': question},
        session_info=session.session_info,
        trace_info=trace_info
    )

    print(f"Sending customer feedback for completion id: {bot_response['completion_id']}")
    fpclient.customer_feedback.update(
        project_id=project_id,
        completion_id=bot_response["completion_id"],
        feedback={
            "is_it_good": random.choice(["nah", "yuh"]),
            "topic": categorization_result["llm_response"],
        },
    )

    trace_info.record_output(
        project_id,
        bot_response['llm_response'],
        eval_results={"bool_field": False, "num_field": 0.9})
    # record feedback for the trace
    trace_feedback = {
        'is_it_good': random.choice([True, False]),
        'freeplay_feedback': random.choice(["positive", "negative"])
    }
    fpclient.customer_feedback.update_trace(project_id, trace_info.trace_id, trace_feedback)
    print(f"Trace info id: {trace_info.trace_id}")
