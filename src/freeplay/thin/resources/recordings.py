import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

from requests import HTTPError

from freeplay import api_support
from freeplay.completions import PromptTemplateWithMetadata, OpenAIFunctionCall
from freeplay.errors import FreeplayClientError, FreeplayError
from freeplay.llm_parameters import LLMParameters
from freeplay.model import InputVariables
from freeplay.support import CallSupport
from freeplay.thin.resources.prompts import PromptInfo
from freeplay.thin.resources.sessions import SessionInfo

logger = logging.getLogger(__name__)


@dataclass
class CallInfo:
    provider: str
    model: str
    start_time: float
    end_time: float
    model_parameters: LLMParameters

    @staticmethod
    def from_prompt_info(prompt_info: PromptInfo, start_time: float, end_time: float) -> 'CallInfo':
        return CallInfo(
            prompt_info.provider,
            prompt_info.model,
            start_time,
            end_time,
            prompt_info.model_parameters
        )


@dataclass
class ResponseInfo:
    is_complete: bool
    function_call_response: Optional[OpenAIFunctionCall] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None


@dataclass
class TestRunInfo:
    test_run_id: str
    test_case_id: str


@dataclass
class RecordPayload:
    all_messages: List[Dict[str, str]]
    inputs: InputVariables

    session_info: SessionInfo
    prompt_info: PromptInfo
    call_info: CallInfo
    response_info: ResponseInfo
    test_run_info: Optional[TestRunInfo] = None


@dataclass
class RecordResponse:
    completion_id: str


class Recordings:
    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def create(self, record_payload: RecordPayload) -> RecordResponse:
        if len(record_payload.all_messages) < 1:
            raise FreeplayClientError("Messages list must have at least one message. "
                                      "The last message should be the current response.")

        completion = record_payload.all_messages[-1]
        history_as_string = json.dumps(record_payload.all_messages[0:-1])

        template = PromptTemplateWithMetadata(
            prompt_template_id=record_payload.prompt_info.prompt_template_id,
            prompt_template_version_id=record_payload.prompt_info.prompt_template_version_id,
            name=record_payload.prompt_info.template_name,
            content=history_as_string,
            flavor_name=record_payload.prompt_info.flavor_name,
            params=record_payload.prompt_info.model_parameters
        )

        record_api_payload = {
            "session_id": record_payload.session_info.session_id,
            "project_version_id": template.prompt_template_version_id,
            "prompt_template_id": template.prompt_template_id,
            "start_time": record_payload.call_info.start_time,
            "end_time": record_payload.call_info.end_time,
            "tag": record_payload.prompt_info.environment,
            "inputs": record_payload.inputs,
            "prompt_content": history_as_string,
            # Content may not be set for function calls, but it is required in the record API payload.
            "return_content": completion.get('content', ''),
            "format_type": None,
            "is_complete": record_payload.response_info.is_complete,
            "model": record_payload.call_info.model,
            "provider": record_payload.call_info.provider,
            "llm_parameters": record_payload.call_info.model_parameters,
        }

        if record_payload.session_info.custom_metadata is not None:
            record_api_payload['custom_metadata'] = record_payload.session_info.custom_metadata

        if record_payload.response_info.function_call_response is not None:
            record_api_payload['function_call_response'] = record_payload.response_info.function_call_response

        if record_payload.test_run_info is not None:
            record_api_payload['test_run_id'] = record_payload.test_run_info.test_run_id

        if record_payload.test_run_info is not None:
            record_api_payload['test_case_id'] = record_payload.test_run_info.test_case_id

        try:
            recorded_response = api_support.post_raw(
                api_key=self.call_support.freeplay_api_key,
                url=f'{self.call_support.api_base}/v1/record',
                payload=record_api_payload
            )
            recorded_response.raise_for_status()
            json_dom = recorded_response.json()
            return RecordResponse(completion_id=str(json_dom['completion_id']))
        except HTTPError as e:
            message = f'There was an error recording to Freeplay. Call will not be logged. ' \
                      f'Status: {e.response.status_code}. '

            if e.response.content:
                try:
                    content = e.response.content
                    json_body = json.loads(content)
                    if 'message' in json_body:
                        message += json_body['message']
                except:
                    pass
            else:
                message += f'{e.__class__}'

            raise FreeplayError(message) from e

        except Exception as e:
            status_code = -1
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code

            message = f'There was an error recording to Freeplay. Call will not be logged. ' \
                      f'Status: {status_code}. {e.__class__}'

            raise FreeplayError(message) from e
