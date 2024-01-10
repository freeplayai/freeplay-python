import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .completions import PromptTemplates, ChatMessage, OpenAIFunctionCall, PromptTemplateWithMetadata
from .errors import FreeplayConfigurationError, FreeplayClientError
from .flavors import Flavor
from .model import InputVariables
from .support import CallSupport
from .llm_parameters import LLMParameters
from .record import DefaultRecordProcessor, RecordCallFields
from .utils import bind_template_variables


@dataclass
class Session:
    session_id: str


@dataclass
class PromptInfo:
    prompt_template_id: str
    prompt_template_version_id: str
    template_name: str
    environment: str
    variables: InputVariables
    model_parameters: LLMParameters
    provider: str
    model: str
    flavor_name: str


@dataclass
class CallInfo:
    provider: str
    model: str
    start_time: float
    end_time: float
    model_parameters: LLMParameters
    test_run_id: Optional[str] = None


@dataclass
class ResponseInfo:
    is_complete: bool
    function_call_response: Optional[OpenAIFunctionCall] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None


@dataclass
class RecordPayload:
    all_messages: List[ChatMessage]
    session_id: str

    prompt_info: PromptInfo
    call_info: CallInfo
    response_info: ResponseInfo


class FreeplayThin:
    def __init__(
            self,
            freeplay_api_key: str,
            api_base: str
    ) -> None:
        if not freeplay_api_key or not freeplay_api_key.strip():
            raise FreeplayConfigurationError("Freeplay API key not set. It must be set to the Freeplay API.")

        self.call_support = CallSupport(freeplay_api_key, api_base,
                                        DefaultRecordProcessor(freeplay_api_key, api_base))
        self.freeplay_api_key = freeplay_api_key
        self.api_base = api_base

    # noinspection PyMethodMayBeStatic
    def create_session(self) -> Session:
        return Session(session_id=str(uuid.uuid4()))

    def get_prompts(self, project_id: str, tag: str) -> PromptTemplates:
        return self.call_support.get_prompts(project_id=project_id, tag=tag)

    def get_prompt(
            self,
            project_id: str,
            template_name: str,
            environment: str,
            variables: InputVariables
    ) -> Tuple[PromptInfo, List[ChatMessage]]:
        prompt_template = self.call_support.get_prompt(
            project_id=project_id,
            template_name=template_name,
            environment=environment
        )

        messages: List[Dict[str, str]] = json.loads(prompt_template.content)
        bound_messages: List[ChatMessage] = [
            {
                "content": bind_template_variables(message['content'], variables),
                "role": message['role']
            } for message in messages
        ]

        params = prompt_template.get_params()
        model = params.pop('model')

        if not prompt_template.flavor_name:
            raise FreeplayConfigurationError(
                "Flavor must be configured in the Freeplay UI. Unable to fulfill request.")

        flavor = Flavor.get_by_name(prompt_template.flavor_name)

        prompt_info = PromptInfo(
            prompt_template_id=prompt_template.prompt_template_id,
            prompt_template_version_id=prompt_template.prompt_template_version_id,
            template_name=prompt_template.name,
            environment=environment,
            variables=variables,
            model_parameters=params,
            provider=flavor.provider,
            model=model,
            flavor_name=prompt_template.flavor_name
        )

        return prompt_info, bound_messages

    # noinspection PyMethodMayBeStatic
    def format(self, flavor_name: str, messages: List[ChatMessage]) -> str | List[ChatMessage]:
        flavor = Flavor.get_by_name(flavor_name)
        return flavor.to_llm_syntax(messages)

    def record_call(self, record_payload: RecordPayload) -> None:
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

        self.call_support.record_processor.record_call(
            RecordCallFields(
                formatted_prompt=history_as_string,
                completion_content=completion['content'],
                completion_is_complete=record_payload.response_info.is_complete,
                start=record_payload.call_info.start_time,
                end=record_payload.call_info.end_time,
                session_id=record_payload.session_id,
                target_template=template,
                variables=record_payload.prompt_info.variables,
                tag=record_payload.prompt_info.environment,
                test_run_id=record_payload.call_info.test_run_id,
                model=record_payload.call_info.model,
                provider=record_payload.prompt_info.provider,
                llm_parameters=record_payload.call_info.model_parameters,
                record_format_type=None  # This is deprecated and unused in the API
            )
        )
