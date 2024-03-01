import json
import time
from copy import copy
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List, Generator
from uuid import uuid4

from freeplay import api_support
from freeplay.api_support import try_decode
from freeplay.completions import PromptTemplates, PromptTemplateWithMetadata, ChatMessage, ChatCompletionResponse, \
    CompletionChunk, CompletionResponse
from freeplay.errors import FreeplayConfigurationError, freeplay_response_error, FreeplayServerError
from freeplay.flavors import ChatFlavor, Flavor, pick_flavor_from_config
from freeplay.llm_parameters import LLMParameters
from freeplay.model import InputVariables
from freeplay.provider_config import ProviderConfig
from freeplay.record import RecordProcessor, RecordCallFields

JsonDom = Dict[str, Any]


class TestCaseTestRunResponse:
    def __init__(self, test_case: JsonDom):
        self.variables: InputVariables = test_case['variables']
        self.id: str = test_case['id']


class TestRunResponse:
    def __init__(
            self,
            test_run_id: str,
            test_cases: List[JsonDom]
    ):
        self.test_cases = [
            TestCaseTestRunResponse(test_case)
            for test_case in test_cases
        ]
        self.test_run_id = test_run_id


@dataclass
class PromptTemplateMetadata:
    provider: Optional[str]
    flavor: Optional[str]
    model: Optional[str]
    params: Optional[Dict[str, Any]] = None
    provider_info: Optional[Dict[str, Any]] = None


@dataclass
class PromptTemplate:
    prompt_template_id: str
    prompt_template_version_id: str
    prompt_template_name: str
    content: List[Dict[str, str]]
    metadata: PromptTemplateMetadata
    format_version: int


class CallSupport:
    def __init__(
            self,
            freeplay_api_key: str,
            api_base: str,
            record_processor: RecordProcessor,
            **kwargs: Any
    ) -> None:
        self.api_base = api_base
        self.freeplay_api_key = freeplay_api_key
        self.client_params = LLMParameters(kwargs)
        self.record_processor = record_processor

    @staticmethod
    def find_template_by_name(prompts: PromptTemplates, template_name: str) -> PromptTemplateWithMetadata:
        templates = [t for t in prompts.templates if t.name == template_name]
        if len(templates) == 0:
            raise FreeplayConfigurationError(f'Could not find template with name "{template_name}"')
        return templates[0]

    @staticmethod
    def create_session_id() -> str:
        return str(uuid4())

    @staticmethod
    def check_all_values_string_or_number(metadata: Optional[Dict[str, Union[str, int, float]]]) -> None:
        if metadata:
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float)):
                    raise FreeplayConfigurationError(f"Invalid value for key {key}: Value must be a string or number.")

    def update_customer_feedback(
            self,
            completion_id: str,
            feedback: Dict[str, Union[bool, str, int, float]]
    ) -> None:
        response = api_support.put_raw(
            self.freeplay_api_key,
            f'{self.api_base}/v1/completion_feedback/{completion_id}',
            feedback
        )
        if response.status_code != 201:
            raise freeplay_response_error("Error updating customer feedback", response)

    def get_prompt(self, project_id: str, template_name: str, environment: str) -> PromptTemplate:
        response = api_support.get_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/v2/projects/{project_id}/prompt-templates/name/{template_name}'
        )

        if response.status_code != 200:
            raise freeplay_response_error(
                f"Error getting prompt template {template_name} in project {project_id} "
                f"and environment {environment}",
                response
            )

        maybe_prompt = try_decode(PromptTemplate, response.content)
        if maybe_prompt is None:
            raise FreeplayServerError(
                f"Error handling prompt {template_name} in project {project_id} "
                f"and environment {environment}"
            )

        return maybe_prompt

    def get_prompts(self, project_id: str, tag: str) -> PromptTemplates:
        response = api_support.get_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/projects/{project_id}/templates/all/{tag}'
        )

        if response.status_code != 200:
            raise freeplay_response_error("Error getting prompt templates", response)

        maybe_prompts = try_decode(PromptTemplates, response.content)
        if maybe_prompts is None:
            raise FreeplayServerError(f'Failed to parse prompt templates from server')

        return maybe_prompts

    def create_test_run(self, project_id: str, testlist: str) -> TestRunResponse:
        response = api_support.post_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/projects/{project_id}/test-runs-cases',
            payload={'testlist_name': testlist},
        )

        if response.status_code != 201:
            raise freeplay_response_error('Error while creating a test run.', response)

        json_dom = response.json()

        return TestRunResponse(json_dom['test_run_id'], json_dom['test_cases'])

    # noinspection PyUnboundLocalVariable
    def prepare_and_make_chat_call(
            self,
            session_id: str,
            flavor: ChatFlavor,
            provider_config: ProviderConfig,
            tag: str,
            target_template: PromptTemplateWithMetadata,
            variables: InputVariables,
            message_history: List[ChatMessage],
            new_messages: Optional[List[ChatMessage]],
            test_run_id: Optional[str] = None,
            completion_parameters: Optional[LLMParameters] = None,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> ChatCompletionResponse:
        # make call
        start = time.time()
        params = target_template.get_params() \
            .merge_and_override(self.client_params) \
            .merge_and_override(completion_parameters)
        prompt_messages = copy(message_history)
        if new_messages is not None:
            prompt_messages.extend(new_messages)
        completion_response = flavor.continue_chat(messages=prompt_messages,
                                                   provider_config=provider_config,
                                                   llm_parameters=params)
        end = time.time()

        model = flavor.get_model_params(params).get('model')
        formatted_prompt = json.dumps(prompt_messages)
        # record data
        record_call_fields = RecordCallFields(
            completion_content=completion_response.content,
            completion_is_complete=completion_response.is_complete,
            end=end,
            formatted_prompt=formatted_prompt,
            session_id=session_id,
            start=start,
            target_template=target_template,
            variables=variables,
            record_format_type=flavor.record_format_type,
            tag=tag,
            test_run_id=test_run_id,
            test_case_id=None,
            model=model,
            provider=flavor.provider,
            llm_parameters=params,
            custom_metadata=metadata,
        )
        self.record_processor.record_call(record_call_fields)

        return completion_response

    # noinspection PyUnboundLocalVariable
    def prepare_and_make_chat_call_stream(
            self,
            session_id: str,
            flavor: ChatFlavor,
            provider_config: ProviderConfig,
            tag: str,
            target_template: PromptTemplateWithMetadata,
            variables: InputVariables,
            message_history: List[ChatMessage],
            test_run_id: Optional[str] = None,
            completion_parameters: Optional[LLMParameters] = None,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> Generator[CompletionChunk, None, None]:
        # make call
        start = time.time()
        prompt_messages = copy(message_history)
        params = target_template.get_params() \
            .merge_and_override(self.client_params) \
            .merge_and_override(completion_parameters)
        completion_response = flavor.continue_chat_stream(prompt_messages, provider_config, llm_parameters=params)

        str_content = ''
        last_is_complete = False
        for chunk in completion_response:
            str_content += chunk.text or ''
            last_is_complete = chunk.is_complete
            yield chunk
        # End time must be logged /after/ streaming the response above, or else OpenAI latency will not be captured.
        end = time.time()

        model = flavor.get_model_params(params).get('model')
        formatted_prompt = json.dumps(prompt_messages)
        record_call_fields = RecordCallFields(
            completion_content=str_content,
            completion_is_complete=last_is_complete,
            end=end,
            formatted_prompt=formatted_prompt,
            session_id=session_id,
            start=start,
            target_template=target_template,
            variables=variables,
            record_format_type=flavor.record_format_type,
            tag=tag,
            test_run_id=test_run_id,
            test_case_id=None,
            model=model,
            provider=flavor.provider,
            llm_parameters=params,
            custom_metadata=metadata,
        )
        self.record_processor.record_call(record_call_fields)

    # noinspection PyUnboundLocalVariable
    def prepare_and_make_call(
            self,
            session_id: str,
            prompts: PromptTemplates,
            template_name: str,
            variables: InputVariables,
            flavor: Optional[Flavor],
            provider_config: ProviderConfig,
            tag: str,
            test_run_id: Optional[str] = None,
            completion_parameters: Optional[LLMParameters] = None,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> CompletionResponse:
        target_template = self.find_template_by_name(prompts, template_name)
        params = target_template.get_params() \
            .merge_and_override(self.client_params) \
            .merge_and_override(completion_parameters)

        final_flavor = pick_flavor_from_config(flavor, target_template.flavor_name)
        formatted_prompt = final_flavor.format(target_template, variables)

        # make call
        start = time.time()
        completion_response = final_flavor.call_service(formatted_prompt=formatted_prompt,
                                                        provider_config=provider_config,
                                                        llm_parameters=params)
        end = time.time()

        model = final_flavor.get_model_params(params).get('model')

        # record data
        record_call_fields = RecordCallFields(
            completion_content=completion_response.content,
            completion_is_complete=completion_response.is_complete,
            end=end,
            formatted_prompt=formatted_prompt,
            session_id=session_id,
            start=start,
            target_template=target_template,
            variables=variables,
            record_format_type=final_flavor.record_format_type,
            tag=tag,
            test_run_id=test_run_id,
            test_case_id=None,
            model=model,
            provider=final_flavor.provider,
            llm_parameters=params,
            custom_metadata=metadata,
        )
        self.record_processor.record_call(record_call_fields)

        return completion_response

    def prepare_and_make_call_stream(
            self,
            session_id: str,
            prompts: PromptTemplates,
            template_name: str,
            variables: InputVariables,
            flavor: Optional[Flavor],
            provider_config: ProviderConfig,
            tag: str,
            test_run_id: Optional[str] = None,
            completion_parameters: Optional[LLMParameters] = None,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> Generator[CompletionChunk, None, None]:
        target_template = self.find_template_by_name(prompts, template_name)
        params = target_template.get_params() \
            .merge_and_override(self.client_params) \
            .merge_and_override(completion_parameters)

        final_flavor = pick_flavor_from_config(flavor, target_template.flavor_name)
        formatted_prompt = final_flavor.format(target_template, variables)

        # make call
        start = int(time.time())
        completion_response = final_flavor.call_service_stream(
            formatted_prompt=formatted_prompt, provider_config=provider_config, llm_parameters=params)
        text_chunks = []
        last_is_complete = False
        for chunk in completion_response:
            text_chunks.append(chunk.text)
            last_is_complete = chunk.is_complete
            yield chunk
        # End time must be logged /after/ streaming the response above, or else OpenAI latency will not be captured.
        end = int(time.time())

        model = final_flavor.get_model_params(params).get('model')

        record_call_fields = RecordCallFields(
            completion_content=''.join(text_chunks),
            completion_is_complete=last_is_complete,
            end=end,
            formatted_prompt=formatted_prompt,
            session_id=session_id,
            start=start,
            target_template=target_template,
            variables=variables,
            record_format_type=final_flavor.record_format_type,
            tag=tag,
            test_run_id=test_run_id,
            test_case_id=None,
            model=model,
            provider=final_flavor.provider,
            llm_parameters=params,
            custom_metadata=metadata,
        )
        self.record_processor.record_call(record_call_fields)
