import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from requests import HTTPError

from freeplay import api_support
from freeplay.errors import FreeplayClientError, FreeplayError
from freeplay.llm_parameters import LLMParameters
from freeplay.model import InputVariables, OpenAIFunctionCall
from freeplay.resources.prompts import PromptInfo
from freeplay.resources.sessions import SessionInfo, TraceInfo
from freeplay.support import CallSupport

logger = logging.getLogger(__name__)


@dataclass
class UsageTokens:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class CallInfo:
    provider: str
    model: str
    start_time: float
    end_time: float
    model_parameters: LLMParameters
    provider_info: Optional[Dict[str, Any]] = None
    usage: Optional[UsageTokens] = None

    @staticmethod
    def from_prompt_info(
            prompt_info: PromptInfo,
            start_time: float,
            end_time: float,
            usage: Optional[UsageTokens] = None
    ) -> 'CallInfo':
        return CallInfo(
            provider=prompt_info.provider,
            model=prompt_info.model,
            start_time=start_time,
            end_time=end_time,
            model_parameters=prompt_info.model_parameters,
            provider_info=prompt_info.provider_info,
            usage=usage
        )


@dataclass
class ResponseInfo:
    is_complete: Optional[bool] = None
    function_call_response: Optional[OpenAIFunctionCall] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None


@dataclass
class TestRunInfo:
    test_run_id: str
    test_case_id: str


@dataclass
class RecordPayload:
    all_messages: List[Dict[str, Any]]
    inputs: InputVariables

    session_info: SessionInfo
    prompt_info: PromptInfo
    call_info: CallInfo
    tool_schema: Optional[List[Dict[str, Any]]] = None
    response_info: Optional[ResponseInfo] = None
    test_run_info: Optional[TestRunInfo] = None
    eval_results: Optional[Dict[str, Union[bool, float]]] = None
    trace_info: Optional[TraceInfo] = None
    completion_id: Optional[UUID] = None


@dataclass
class RecordUpdatePayload:
    project_id: str
    completion_id: str
    new_messages: Optional[List[Dict[str, Any]]] = None
    eval_results: Optional[Dict[str, Union[bool, float]]] = None


@dataclass
class RecordResponse:
    completion_id: str


class Recordings:
    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def create(self, record_payload: RecordPayload) -> RecordResponse:  # type: ignore
        if len(record_payload.all_messages) < 1:
            raise FreeplayClientError("Messages list must have at least one message. "
                                      "The last message should be the current response.")

        record_api_payload: Dict[str, Any] = {
            "messages": record_payload.all_messages,
            "inputs": record_payload.inputs,
            "tool_schema": record_payload.tool_schema,
            "session_info": {"custom_metadata": record_payload.session_info.custom_metadata},
            "prompt_info": {
                "environment": record_payload.prompt_info.environment,
                "prompt_template_version_id": record_payload.prompt_info.prompt_template_version_id,
            },
            "call_info": {
                "start_time": record_payload.call_info.start_time,
                "end_time": record_payload.call_info.end_time,
                "model": record_payload.call_info.model,
                "provider": record_payload.call_info.provider,
                "provider_info": record_payload.call_info.provider_info,
                "llm_parameters": record_payload.call_info.model_parameters,
            }
        }

        if record_payload.completion_id is not None:
            record_api_payload['completion_id'] = str(record_payload.completion_id)

        if record_payload.session_info.custom_metadata is not None:
            record_api_payload['custom_metadata'] = record_payload.session_info.custom_metadata

        if record_payload.response_info is not None:
            if record_payload.response_info.function_call_response is not None:
                record_api_payload['response_info'] = {
                    "function_call_response": {
                        "name": record_payload.response_info.function_call_response["name"],
                        "arguments": record_payload.response_info.function_call_response["arguments"],
                    }
                }

        if record_payload.test_run_info is not None:
            record_api_payload['test_run_info'] = {
                "test_run_id": record_payload.test_run_info.test_run_id,
                "test_case_id": record_payload.test_run_info.test_case_id
            }

        if record_payload.eval_results is not None:
            record_api_payload['eval_results'] = record_payload.eval_results

        if record_payload.trace_info is not None:
            record_api_payload['trace_info'] = {
                "trace_id": record_payload.trace_info.trace_id
            }

        if record_payload.call_info.usage is not None:
            record_api_payload['call_info']['usage'] = {
                "prompt_tokens": record_payload.call_info.usage.prompt_tokens,
                "completion_tokens": record_payload.call_info.usage.completion_tokens,
            }

        try:
            recorded_response = api_support.post_raw(
                api_key=self.call_support.freeplay_api_key,
                url=f'{self.call_support.api_base}/v2/projects/{record_payload.prompt_info.project_id}/sessions/{record_payload.session_info.session_id}/completions',
                payload=record_api_payload
            )
            recorded_response.raise_for_status()
            json_dom = recorded_response.json()
            return RecordResponse(completion_id=str(json_dom['completion_id']))
        except HTTPError as e:
            message = f'There was an error recording to Freeplay. Call will not be logged. ' \
                      f'Status: {e.response.status_code}. '

            self.__handle_and_raise_api_error(e, message)

        except Exception as e:
            status_code = -1
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code

            message = f'There was an error recording to Freeplay. Call will not be logged. ' \
                      f'Status: {status_code}. {e.__class__}'

            raise FreeplayError(message) from e

    def update(self, record_update_payload: RecordUpdatePayload) -> RecordResponse:  # type: ignore
        record_update_api_payload: Dict[str, Any] = {
            "new_messages": record_update_payload.new_messages,
            "eval_results": record_update_payload.eval_results,
        }

        try:
            record_update_response = api_support.post_raw(
                api_key=self.call_support.freeplay_api_key,
                url=f'{self.call_support.api_base}/v2/projects/{record_update_payload.project_id}/completions/{record_update_payload.completion_id}',
                payload=record_update_api_payload
            )
            record_update_response.raise_for_status()
            json_dom = record_update_response.json()
            return RecordResponse(completion_id=str(json_dom['completion_id']))
        except HTTPError as e:
            message = f'There was an error updating the completion. Status: {e.response.status_code}.'
            self.__handle_and_raise_api_error(e, message)

    @staticmethod
    def __handle_and_raise_api_error(e: HTTPError, messages: str) -> None:
        if e.response.content:
            try:
                content = e.response.content
                json_body = json.loads(content)
                if 'message' in json_body:
                    messages += json_body['message']
            except:
                pass
        else:
            messages += f'{e.__class__}'
        raise FreeplayError(messages) from e
