# This file is for re-usable utils an example customer might write to re-use their code.
# These are not part of the Freeplay SDK.
import os
from typing import Optional, Dict

from freeplay import Freeplay, CallInfo, ResponseInfo, RecordPayload
from freeplay.resources.prompts import FormattedPrompt
from freeplay.resources.recordings import RecordResponse, TestRunInfo
from freeplay.resources.sessions import Session


def get_freeplay_thin_client() -> Freeplay:
    return Freeplay(
        freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
        api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
    )


def record_results(
        client: Freeplay,
        formatted_prompt: FormattedPrompt,
        completion_content: str,
        variables: Dict[str, str],
        session: Session,
        start: float,
        end: float,
        test_run_info: Optional[TestRunInfo] = None
) -> RecordResponse:
    all_messages = formatted_prompt.all_messages(
        new_message={'role': 'Assistant', 'content': completion_content}
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
            all_messages=all_messages,
            session_info=session.session_info,
            inputs=variables,
            prompt_info=formatted_prompt.prompt_info,
            call_info=call_info,
            response_info=response_info,
            test_run_info=test_run_info
        )
    )
