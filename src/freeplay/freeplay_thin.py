import json
import uuid
from dataclasses import dataclass
from typing import List, Optional, cast

from .completions import PromptTemplates, ChatMessage, OpenAIFunctionCall, PromptTemplateWithMetadata
from .errors import FreeplayConfigurationError, FreeplayClientError
from .flavors import Flavor
from .llm_parameters import LLMParameters
from .model import InputVariables
from .record import DefaultRecordProcessor, RecordCallFields
from .support import CallSupport
from .utils import bind_template_variables


@dataclass
class Session:
    session_id: str


@dataclass
class CallInfo:
    provider: str
    model: str
    start_time: float
    end_time: float
    model_parameters: LLMParameters


@dataclass
class PromptInfo:
    prompt_template_id: str
    prompt_template_version_id: str
    template_name: str
    environment: str
    model_parameters: LLMParameters
    provider: str
    model: str
    flavor_name: str

    def get_call_info(self, start_time: float, end_time: float) -> CallInfo:
        return CallInfo(
            self.provider,
            self.model,
            start_time,
            end_time,
            self.model_parameters
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


class FormattedPrompt:
    def __init__(
            self,
            prompt_info: PromptInfo,
            messages: List[dict[str, str]],
            formatted_prompt: str | list[dict[str, str]]
    ):
        self.prompt_info = prompt_info
        self.messages = messages
        self.llm_prompt = formatted_prompt

    def all_messages(
            self,
            new_message: dict[str, str]
    ) -> list[dict[str, str]]:
        return self.messages + [new_message]


class BoundPrompt:
    def __init__(
            self,
            prompt_info: PromptInfo,
            messages: List[dict[str, str]]
    ):
        self.prompt_info = prompt_info
        self.messages = messages

    def format(
            self,
            flavor_name: Optional[str] = None
    ) -> FormattedPrompt:
        final_flavor = flavor_name or self.prompt_info.flavor_name
        flavor = Flavor.get_by_name(final_flavor)
        llm_format = flavor.to_llm_syntax(cast(list[ChatMessage], self.messages))

        return FormattedPrompt(
            self.prompt_info,
            self.messages,
            cast(str | list[dict[str, str]], llm_format)
        )


class TemplatePrompt:
    def __init__(
            self,
            prompt_info: PromptInfo,
            messages: List[dict[str, str]]
    ):
        self.prompt_info = prompt_info
        self.messages = messages

    def bind(self, variables: InputVariables) -> BoundPrompt:
        bound_messages = [
            {'role': message['role'], 'content': bind_template_variables(message['content'], variables)}
            for message in self.messages
        ]
        return BoundPrompt(self.prompt_info, bound_messages)


@dataclass
class RecordPayload:
    all_messages: list[dict[str, str]]
    inputs: InputVariables
    session_id: str

    prompt_info: PromptInfo
    call_info: CallInfo
    response_info: ResponseInfo
    test_run_info: Optional[TestRunInfo] = None


@dataclass
class TestCase:
    def __init__(self, test_case_id: str, variables: InputVariables):
        self.id = test_case_id
        self.variables = variables


@dataclass
class TestRun:
    def __init__(
            self,
            test_run_id: str,
            test_cases: List[TestCase]
    ):
        self.test_run_id = test_run_id
        self.test_cases = test_cases

    def get_test_cases(self) -> List[TestCase]:
        return self.test_cases

    def get_test_run_info(self, test_case_id: str) -> TestRunInfo:
        return TestRunInfo(self.test_run_id, test_case_id)


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
            environment: str
    ) -> TemplatePrompt:
        prompt_template = self.call_support.get_prompt(
            project_id=project_id,
            template_name=template_name,
            environment=environment
        )

        messages = json.loads(prompt_template.content)

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
            model_parameters=params,
            provider=flavor.provider,
            model=model,
            flavor_name=prompt_template.flavor_name
        )

        return TemplatePrompt(prompt_info, messages)

    def get_bound_prompt(
            self,
            project_id: str,
            template_name: str,
            environment: str,
            variables: InputVariables
    ) -> BoundPrompt:
        template_prompt = self.get_prompt(
            project_id=project_id,
            template_name=template_name,
            environment=environment
        )

        return template_prompt.bind(variables)

    def get_formatted_prompt(
            self,
            project_id: str,
            template_name: str,
            environment: str,
            variables: InputVariables,
            flavor_name: Optional[str] = None
    ) -> FormattedPrompt:
        bound_prompt = self.get_bound_prompt(
            project_id=project_id,
            template_name=template_name,
            environment=environment,
            variables=variables
        )

        return bound_prompt.format(flavor_name)

    def create_test_run(self, project_id: str, testlist: str) -> TestRun:
        test_run = self.call_support.create_test_run(project_id, testlist)
        test_cases = [
            TestCase(test_case_id=test_case.id, variables=test_case.variables)
            for test_case in test_run.test_cases
        ]

        return TestRun(test_run.test_run_id, test_cases)

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
                variables=record_payload.inputs,
                tag=record_payload.prompt_info.environment,
                test_run_id=record_payload.test_run_info.test_run_id if record_payload.test_run_info else None,
                test_case_id=record_payload.test_run_info.test_case_id if record_payload.test_run_info else None,
                model=record_payload.call_info.model,
                provider=record_payload.prompt_info.provider,
                llm_parameters=record_payload.call_info.model_parameters,
                record_format_type=None  # This is deprecated and unused in the API
            )
        )
