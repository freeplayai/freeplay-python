import json
import os
import time
from copy import deepcopy
from typing import Optional

import boto3
from anthropic import Anthropic, NotGiven
from openai import OpenAI

from freeplay import (
    Freeplay,
    RecordPayload,
    ResponseInfo,
    CallInfo,
    SessionInfo,
    TraceInfo,
)

fpclient = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
project_id = os.environ["FREEPLAY_PROJECT_ID"]
enviroment = "dev"

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
sagemaker_client = boto3.client(
    "sagemaker-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)


articles = [
    "george washington was the first president of the united states",
    "the sky is blue",
    "the earth is round",
    "",
]
questions = [
    "who was the first president of the united states?",
    "what color is the sky?",
    "what shape is the earth?",
    "repeat the first question and answer",
]
input_pairs = list(zip(articles, questions))

template_prompt = fpclient.prompts.get(
    project_id=project_id, template_name="History-QA", environment=enviroment
)
print("Template Prompt messages")
print(json.dumps(template_prompt.messages, indent=2))
print("\n")


def call_and_record(
    project_id: str,
    template_name: str,
    env: str,
    history: list,
    input_variables: dict,
    session_info: SessionInfo,
    trace_info: Optional[TraceInfo] = None,
) -> dict:
    formatted_prompt = fpclient.prompts.get_formatted(
        project_id=project_id,
        template_name=template_name,
        environment=env,
        variables=input_variables,
        history=history,
    )

    start = time.time()
    if formatted_prompt.prompt_info.provider == "openai":
        print("LLM Input")
        print(json.dumps(formatted_prompt.messages, indent=2))
        completion = openai_client.chat.completions.create(  # type: ignore
            model=formatted_prompt.prompt_info.model,
            messages=formatted_prompt.messages,
            **formatted_prompt.prompt_info.model_parameters,
        )
        end = time.time()

        llm_response = completion.choices[0].message.content
    elif formatted_prompt.prompt_info.provider == "anthropic":
        print("LLM Input")
        print(formatted_prompt.system_content)
        print(json.dumps(formatted_prompt.llm_prompt, indent=2))
        completion = anthropic_client.messages.create(
            system=formatted_prompt.system_content or NotGiven(),
            messages=formatted_prompt.llm_prompt,
            model=formatted_prompt.prompt_info.model,
            **formatted_prompt.prompt_info.model_parameters,
        )
        end = time.time()
        llm_response = completion.content[0].text
    elif formatted_prompt.prompt_info.provider == "sagemaker":
        print("LLM Input")
        print(formatted_prompt.llm_prompt_text)
        model_params = deepcopy(formatted_prompt.prompt_info.model_parameters)
        model_params.update({"stop": ["<|eot_id|>"]})
        completion = sagemaker_client.invoke_endpoint(
            EndpointName=formatted_prompt.prompt_info.provider_info["endpoint_name"],
            InferenceComponentName=formatted_prompt.prompt_info.provider_info[
                "inference_component_name"
            ],
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(
                {"inputs": formatted_prompt.llm_prompt_text, "parameters": model_params}
            ),
        )
        end = time.time()
        llm_response = json.loads(completion["Body"].read().decode("utf-8"))[
            "generated_text"
        ]
    else:
        raise ValueError(
            f"Unsupported provider: {formatted_prompt.prompt_info.provider}"
        )

    print("Completion: %s" % llm_response)

    assistant_response = {"role": "assistant", "content": llm_response}
    all_messages = formatted_prompt.all_messages(new_message=assistant_response)

    call_info = CallInfo.from_prompt_info(formatted_prompt.prompt_info, start, end)
    response_info = ResponseInfo(
        is_complete=True,
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

    return {
        "completion_id": record_response.completion_id,
        "llm_response": assistant_response,
        "all_messages": all_messages,
    }


session = fpclient.sessions.create()
print("session id %s" % session.session_info.session_id)
history = []
for inputs in input_pairs:
    input_vars = {"question": inputs[1], "article": inputs[0]}
    record_response = call_and_record(
        project_id=project_id,
        template_name="History-QA",
        env=enviroment,
        history=history,
        input_variables=input_vars,
        session_info=session.session_info,
    )
    # this would work for 80% of cases
    history = [
        msg for msg in record_response["all_messages"] if msg["role"] != "system"
    ]
    print("\n")
