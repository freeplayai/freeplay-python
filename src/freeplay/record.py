import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict, Union
from typing import Optional

from . import api_support
from .completions import PromptTemplateWithMetadata, OpenAIFunctionCall
from .llm_parameters import LLMParameters
from .model import InputVariables

logger = logging.getLogger(__name__)


@dataclass
class RecordCallFields:
    completion_content: str
    completion_is_complete: bool
    end: float
    formatted_prompt: str
    session_id: str
    start: float
    target_template: PromptTemplateWithMetadata
    variables: InputVariables
    tag: str
    test_run_id: Optional[str]
    test_case_id: Optional[str]
    record_format_type: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    llm_parameters: Optional[LLMParameters]
    function_call_response: Optional[OpenAIFunctionCall] = None
    custom_metadata: Optional[Dict[str, Union[str, int, float]]] = None


class RecordProcessor(ABC):
    @abstractmethod
    def record_call(
            self,
            record_call: RecordCallFields,
    ) -> None:
        pass


class NoOpRecorder(RecordProcessor):
    def record_call(
            self,
            record_call: RecordCallFields,
    ) -> None:
        pass


no_op_recorder = NoOpRecorder()


class DefaultRecordProcessor(RecordProcessor):

    def __init__(
            self,
            freeplay_api_key: str,
            api_base: str
    ) -> None:
        self.api_base = api_base
        self.freeplay_api_key = freeplay_api_key

    def record_call(
            self,
            record_call: RecordCallFields
    ) -> None:
        record_payload = {
            "session_id": record_call.session_id,
            "project_version_id": record_call.target_template.prompt_template_version_id,
            "prompt_template_id": record_call.target_template.prompt_template_id,
            "start_time": record_call.start,
            "end_time": record_call.end,
            "tag": record_call.tag,
            "inputs": record_call.variables,
            "prompt_content": record_call.formatted_prompt,
            "return_content": record_call.completion_content,
            "format_type": record_call.record_format_type,
            "is_complete": record_call.completion_is_complete,
            "model": record_call.model,
            "provider": record_call.provider,
            "llm_parameters": record_call.llm_parameters,
        }

        if record_call.custom_metadata is not None:
            record_payload['custom_metadata'] = record_call.custom_metadata

        if record_call.function_call_response is not None:
            record_payload['function_call_response'] = record_call.function_call_response

        if record_call.test_run_id is not None:
            record_payload['test_run_id'] = record_call.test_run_id

        if record_call.test_case_id is not None:
            record_payload['test_case_id'] = record_call.test_case_id

        try:
            recorded_response = api_support.post_raw(
                api_key=self.freeplay_api_key,
                url=f'{self.api_base}/v1/record',
                payload=record_payload
            )
            recorded_response.raise_for_status()
        except Exception as e:
            status_code = -1
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            logger.warning(
                f'There was an error recording to Freeplay. Call will not be logged. '
                f'Status: {status_code}. {e.__class__}'
            )
