import json
from abc import abstractmethod, ABC
from copy import copy
from typing import Any, Dict, Generator, List, Optional

import anthropic  # type: ignore
import openai
from openai.error import AuthenticationError, InvalidRequestError

from .completions import CompletionChunk, PromptTemplateWithMetadata, CompletionResponse, ChatCompletionResponse, \
    ChatMessage
from .errors import FreeplayConfigurationError, LLMClientError, LLMServerError, FreeplayError
from .llm_parameters import LLMParameters
from .provider_config import ProviderConfig, OpenAIConfig, AnthropicConfig
from .utils import format_template_variables


class Flavor(ABC):
    @classmethod
    def get_by_name(cls, flavor_name: str) -> 'Flavor':
        if flavor_name == OpenAIChat.record_format_type:
            return OpenAIChat()
        elif flavor_name == AzureOpenAIChat.record_format_type:
            return AzureOpenAIChat()
        elif flavor_name == AnthropicClaudeChat.record_format_type:
            return AnthropicClaudeChat()
        else:
            raise FreeplayConfigurationError(
                'Configured flavor not found in SDK. Please update your SDK version or configure '
                'a different model in the Freeplay UI.')

    @property
    @abstractmethod
    def provider(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def record_format_type(self) -> str:
        raise NotImplementedError()

    @property
    def _model_params_with_defaults(self) -> LLMParameters:
        return LLMParameters.empty()

    @abstractmethod
    def format(self, prompt_template: PromptTemplateWithMetadata, variables: Dict[str, str]) -> str:
        pass

    @abstractmethod
    def call_service(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> CompletionResponse:
        pass

    @abstractmethod
    def call_service_stream(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        pass

    def get_model_params(self, llm_parameters: LLMParameters) -> LLMParameters:
        return self._model_params_with_defaults.merge_and_override(llm_parameters)


class ChatFlavor(Flavor, ABC):
    @abstractmethod
    def continue_chat(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def continue_chat_stream(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        pass


class OpenAI(Flavor, ABC):
    def configure_openai(
            self,
            openai_config: Optional[OpenAIConfig],
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            api_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not openai_config:
            raise FreeplayConfigurationError(
                "Missing OpenAI key. Use a ProviderConfig to specify keys prior to getting completion.")

        if api_base:
            openai.api_base = api_base
        elif openai_config.api_base:
            openai.api_base = openai_config.api_base

        if api_type:
            openai.api_type = api_type

        if api_version:
            openai.api_version = api_version

        if not openai_config.api_key or not openai_config.api_key.strip():
            raise FreeplayConfigurationError("OpenAI API key is not set. It must be set to make calls to the service.")

        openai.api_key = openai_config.api_key

    @property
    def provider(self) -> str:
        return "openai"


class OpenAIChat(OpenAI, ChatFlavor):
    record_format_type = "openai_chat"
    _model_params_with_defaults = LLMParameters({
        "model": "gpt-3.5-turbo"
    })

    def format(self, prompt_template: PromptTemplateWithMetadata, variables: Dict[str, str]) -> str:
        # Extract messages JSON to enable formatting of individual content fields of each message. If we do not
        # extract the JSON, current variable interpolation will fail on JSON curly braces.
        messages_as_json: List[Dict[str, str]] = json.loads(prompt_template.content)
        formatted_messages = [
            {
                "content": format_template_variables(message['content'], variables), "role": message['role']
            } for message in messages_as_json]
        return json.dumps(formatted_messages)

    def call_service(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> CompletionResponse:
        messages = json.loads(formatted_prompt)
        completion = self._call_openai(messages, provider_config, llm_parameters, stream=False)
        return CompletionResponse(
            content=completion.choices[0].message.content or '',
            is_complete=completion.choices[0].finish_reason == 'stop',
            openai_function_call=completion.choices[0].message.get('function_call')
        )

    def call_service_stream(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        messages = json.loads(formatted_prompt)
        completion_stream = self._call_openai(messages, provider_config, llm_parameters, stream=True)
        for chunk in completion_stream:
            yield CompletionChunk(
                text=chunk.choices[0].delta.get('content') or '',
                is_complete=chunk.choices[0].finish_reason == 'stop',
                openai_function_call=chunk.choices[0].delta.get('function_call')
            )

    def continue_chat(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> ChatCompletionResponse:
        completion = self._call_openai(messages, provider_config, llm_parameters, stream=False)

        message_history = copy(messages)
        message_history.append(completion.choices[0].message.to_dict())
        return ChatCompletionResponse(
            content=completion.choices[0].message.content,
            message_history=message_history,
            is_complete=completion.choices[0].finish_reason == "stop"
        )

    def continue_chat_stream(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        completion_stream = self._call_openai(messages, provider_config, llm_parameters, stream=True)
        for chunk in completion_stream:
            yield CompletionChunk(
                text=chunk.choices[0].delta.get('content', ''),
                is_complete=chunk.choices[0].finish_reason == "stop"
            )

    def _call_openai(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters,
            stream: bool
    ) -> Any:
        self.configure_openai(provider_config.openai)
        llm_parameters.pop('endpoint')
        try:
            return openai.ChatCompletion.create(
                messages=messages,
                **self.get_model_params(llm_parameters),
                stream=stream,
            )  # type: ignore
        except (InvalidRequestError, AuthenticationError) as e:
            raise LLMClientError("Unable to call OpenAI") from e
        except Exception as e:
            raise LLMServerError("Unable to call OpenAI") from e


class AzureOpenAIChat(OpenAIChat):
    record_format_type = "azure_openai_chat"

    def _call_openai(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters,
            stream: bool
    ) -> Any:
        api_version = llm_parameters.get('api_version')
        deployment_id = llm_parameters.get('deployment_id')
        resource_name = llm_parameters.get('resource_name')
        endpoint = f'https://{resource_name}.openai.azure.com'
        self.configure_openai(
            provider_config.azure,
            api_base=endpoint,
            api_type='azure',
            api_version=api_version
        )
        llm_parameters.pop('resource_name')
        try:
            return openai.ChatCompletion.create(
                messages=messages,
                **self.get_model_params(llm_parameters),
                engine=deployment_id,
                stream=stream,
            )  # type: ignore
        except (InvalidRequestError, AuthenticationError) as e:
            raise LLMClientError("Unable to call OpenAI") from e
        except Exception as e:
            raise LLMServerError("Unable to call OpenAI") from e

    @property
    def provider(self) -> str:
        return "azure"


class AnthropicClaudeText(Flavor):
    record_format_type = "anthropic_text"
    _model_params_with_defaults = LLMParameters({
        "model": "claude-v1",
        "max_tokens_to_sample": 100
    })

    def __init__(self) -> None:
        self.client: Optional[anthropic.Client] = None

    @property
    def provider(self) -> str:
        return "anthropic"

    def get_anthropic_client(self, anthropic_config: Optional[AnthropicConfig]) -> Any:
        if self.client:
            return self.client

        if not anthropic_config:
            raise FreeplayConfigurationError(
                "Missing Anthropic key. Use a ProviderConfig to specify keys prior to getting completion.")

        self.client = anthropic.Client(api_key=anthropic_config.api_key)
        return self.client

    def format(self, prompt_template: PromptTemplateWithMetadata, variables: Dict[str, str]) -> str:
        interpolated_prompt = format_template_variables(prompt_template.content, variables)
        # Anthropic expects a specific Chat format "Human: $PROMPT_TEXT\n\nAssistant:". We add the wrapping for Text.
        chat_formatted_prompt = f"{anthropic.HUMAN_PROMPT} {interpolated_prompt} {anthropic.AI_PROMPT}"
        return chat_formatted_prompt

    def call_service(self, formatted_prompt: str, provider_config: ProviderConfig,
                     llm_parameters: LLMParameters) -> CompletionResponse:
        try:
            client = self.get_anthropic_client(provider_config.anthropic)
            anthropic_response = client.completion(
                prompt=formatted_prompt,
                **self.get_model_params(llm_parameters)
            )
            return CompletionResponse(
                content=anthropic_response['completion'],
                is_complete=anthropic_response['stop_reason'] == 'stop_sequence'
            )
        except anthropic.APIError as e:
            raise FreeplayError("Error calling Anthropic") from e

    def call_service_stream(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        try:
            client = self.get_anthropic_client(provider_config.anthropic)
            anthropic_response = client.completion_stream(
                prompt=formatted_prompt,
                **self.get_model_params(llm_parameters)
            )

            # Yield incremental text completions. Claude returns the full text output in every chunk.
            # We want to predictably return a stream like we do for OpenAI.
            prev_chunk = ''
            for chunk in anthropic_response:
                if len(prev_chunk) != 0:
                    incremental_new_text = chunk['completion'].split(prev_chunk)[1]
                else:
                    incremental_new_text = chunk['completion']

                prev_chunk = chunk['completion']
                yield CompletionChunk(
                    text=incremental_new_text,
                    is_complete=chunk['stop_reason'] == 'stop_sequence'
                )
        except anthropic.APIError as e:
            raise FreeplayError("Error calling Anthropic") from e


class AnthropicClaudeChat(ChatFlavor):
    record_format_type = "anthropic_chat"
    _model_params_with_defaults = LLMParameters({
        "model": "claude-2",
        "max_tokens_to_sample": 100
    })

    def __init__(self) -> None:
        self.client: Optional[anthropic.Anthropic] = None

    @property
    def provider(self) -> str:
        return "anthropic"

    def get_anthropic_client(self, anthropic_config: Optional[AnthropicConfig]) -> anthropic.Client:
        if self.client:
            return self.client

        if not anthropic_config:
            raise FreeplayConfigurationError(
                "Missing Anthropic key. Use a ProviderConfig to specify keys prior to getting completion.")

        self.client = anthropic.Client(api_key=anthropic_config.api_key)
        return self.client

    # This just formats the prompt for uploading to the record endpoint.
    # TODO: Move this to a base class.
    def format(self, prompt_template: PromptTemplateWithMetadata, variables: Dict[str, str]) -> str:
        # Extract messages JSON to enable formatting of individual content fields of each message. If we do not
        # extract the JSON, current variable interpolation will fail on JSON curly braces.
        messages_as_json: List[Dict[str, str]] = json.loads(prompt_template.content)
        formatted_messages = [
            {
                "content": format_template_variables(message['content'], variables),
                "role": self.__to_anthropic_role(message['role'])
            } for message in messages_as_json]
        return json.dumps(formatted_messages)

    @staticmethod
    def __to_anthropic_role(role: str) -> str:
        if role == 'Human':
            return 'Human'
        elif role == 'assistant' or role == 'Assistant':
            return 'Assistant'
        else:
            # Anthropic does not support system role for now.
            return 'Human'

    @staticmethod
    def __to_anthropic_chat_format(messages: List[ChatMessage]) -> str:
        formatted_messages = []
        for message in messages:
            formatted_messages.append(f"{message['role']}: {message['content']}")
        formatted_messages.append('Assistant:')

        return "\n\n" + "\n\n".join(formatted_messages)

    def continue_chat(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> ChatCompletionResponse:
        formatted_prompt = self.__to_anthropic_chat_format(messages)
        try:
            client = self.get_anthropic_client(provider_config.anthropic)
            completion = client.completions.create(
                prompt=formatted_prompt,
                **self.get_model_params(llm_parameters)
            )
            content = completion.completion
            message_history = messages + [{"role": "assistant", "content": content}]
            return ChatCompletionResponse(
                content=content,
                is_complete=completion.stop_reason == 'stop_sequence',
                message_history=message_history,
            )
        except anthropic.APIError as e:
            raise FreeplayError("Error calling Anthropic") from e

    def continue_chat_stream(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        formatted_prompt = self.__to_anthropic_chat_format(messages)
        try:
            client = self.get_anthropic_client(provider_config.anthropic)
            anthropic_response = client.completions.create(
                prompt=formatted_prompt,
                stream=True,
                **self.get_model_params(llm_parameters)
            )

            for chunk in anthropic_response:
                yield CompletionChunk(
                    text=chunk.completion,
                    is_complete=chunk.stop_reason == 'stop_sequence'
                )
        except anthropic.APIError as e:
            raise FreeplayError("Error calling Anthropic") from e

    def call_service(self, formatted_prompt: str, provider_config: ProviderConfig,
                     llm_parameters: LLMParameters) -> CompletionResponse:
        messages = json.loads(formatted_prompt)
        completion = self.continue_chat(messages, provider_config, llm_parameters)
        return CompletionResponse(
            content=completion.content,
            is_complete=completion.is_complete,
        )

    def call_service_stream(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        messages = json.loads(formatted_prompt)
        return self.continue_chat_stream(messages, provider_config, llm_parameters)
