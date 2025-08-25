# This file is for re-usable utils an example customer might write to re-use their code.
# These are not part of the Freeplay SDK.
import os
from typing import Optional, Dict, Tuple, List, Union, Any, Mapping

from anthropic import NotGiven
from anthropic.types import MessageParam

from freeplay import Freeplay, CallInfo, ResponseInfo, RecordPayload
from freeplay.resources.prompts import FormattedPrompt, PromptInfo
from freeplay.resources.recordings import RecordResponse, TestRunInfo
from freeplay.resources.sessions import Session, TraceInfo


def get_freeplay_thin_client() -> Freeplay:
    return Freeplay(
        freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
        api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
    )


def record_results_messages(
        client: Freeplay,
        project_id: str,
        all_messages: Any,
        variables: Mapping[str, Any],
        session: Session,
        start: float,
        end: float,
        trace_info: Optional[TraceInfo] = None,
        test_run_info: Optional[TestRunInfo] = None,
        eval_results: Optional[Dict[str, Union[bool, float]]] = None,
        formatted_prompt: Optional[FormattedPrompt] = None
) -> RecordResponse:
    if formatted_prompt is not None or (start is not None and end is not None):
        call_info = CallInfo(
            formatted_prompt.prompt_info.provider if formatted_prompt is not None else None,
            model=formatted_prompt.prompt_info.model if formatted_prompt is not None else None,
            start_time=start,
            end_time=end,
            model_parameters=formatted_prompt.prompt_info.model_parameters if formatted_prompt is not None else None)

    response_info = ResponseInfo(
        is_complete=True
    )

    return client.recordings.create(
        RecordPayload(
            project_id=project_id,
            all_messages=all_messages,
            session_info=session.session_info,
            inputs=variables,
            prompt_version_info=formatted_prompt.prompt_info if formatted_prompt is not None else None,
            call_info=call_info,
            response_info=response_info,
            test_run_info=test_run_info,
            eval_results=eval_results,
            trace_info=trace_info
        )
    )


def record_results(
        client: Freeplay,
        project_id: str,
        formatted_prompt: FormattedPrompt,
        completion_content: MessageParam,
        variables: Dict[str, str],
        session: Session,
        start: float,
        end: float,
        test_run_info: Optional[TestRunInfo] = None,
        eval_results: Optional[Dict[str, Union[bool, float]]] = None
) -> RecordResponse:
    all_messages = formatted_prompt.all_messages(
        new_message=completion_content
    )
    call_info = CallInfo(
        formatted_prompt.prompt_info.provider,
        model=formatted_prompt.prompt_info.model,
        start_time=start,
        end_time=end,
        model_parameters=formatted_prompt.prompt_info.model_parameters)

    response_info = ResponseInfo(
        is_complete=True
    )

    return client.recordings.create(
        RecordPayload(
            project_id=project_id,
            all_messages=all_messages,
            session_info=session.session_info,
            inputs=variables,
            prompt_version_info=formatted_prompt.prompt_info,
            call_info=call_info,
            response_info=response_info,
            test_run_info=test_run_info,
            eval_results=eval_results
        )
    )


def record_results_from_bound(
        client: Freeplay,
        project_id: str,
        prompt_info: PromptInfo,
        prompt_messages: List[Dict[str, str]],
        completion_content: str,
        variables: Dict[str, str],
        session: Session,
        start: float,
        end: float,
        test_run_info: Optional[TestRunInfo] = None
) -> RecordResponse:
    all_messages = prompt_messages + [{'role': 'Assistant', 'content': completion_content}]

    call_info = CallInfo(
        prompt_info.provider,
        model=prompt_info.model,
        start_time=start,
        end_time=end,
        model_parameters=prompt_info.model_parameters)

    response_info = ResponseInfo(
        is_complete=True
    )

    return client.recordings.create(
        RecordPayload(
            project_id=project_id,
            all_messages=all_messages,
            session_info=session.session_info,
            inputs=variables,
            prompt_info=prompt_info,
            call_info=call_info,
            response_info=response_info,
            test_run_info=test_run_info
        )
    )


def format_anthropic_messages(formatted_prompt: FormattedPrompt) -> Tuple[Union[str, NotGiven], List[Dict[str, str]]]:
    system_message = next((message for message in formatted_prompt.messages if message['role'] == 'system'), None)
    system_message_content = system_message['content'] if system_message else NotGiven()
    other_messages = [message for message in formatted_prompt.messages if message['role'] != 'system']
    return system_message_content, other_messages
