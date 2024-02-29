import json
from abc import abstractmethod, ABC
from copy import copy
from typing import cast, Any, Dict, Generator, List, Optional, Union

import anthropic
import openai
from openai import AuthenticationError, BadRequestError, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

from .completions import CompletionChunk, PromptTemplateWithMetadata, CompletionResponse, ChatCompletionResponse, \
    ChatMessage, OpenAIFunctionCall
from .errors import FreeplayConfigurationError, LLMClientError, LLMServerError, FreeplayError
from .llm_parameters import LLMParameters
from .model import InputVariables
from .provider_config import AnthropicConfig, AzureConfig, OpenAIConfig, ProviderConfig
from .utils import bind_template_variables


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
                f'Configured flavor ({flavor_name}) not found in SDK. Please update your SDK version or configure '
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
    def format(self, prompt_template: PromptTemplateWithMetadata, variables: InputVariables) -> str:
        pass

    @abstractmethod
    def to_llm_syntax(self, messages: List[ChatMessage]) -> Union[str, List[ChatMessage]]:
        raise NotImplementedError()

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


class OpenAIChatFlavor(ChatFlavor, ABC):

    @abstractmethod
    def _call_openai(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters,
            stream: bool
    ) -> Union[ChatCompletion, openai.Stream[ChatCompletionChunk]]:
        pass

    def format(self, prompt_template: PromptTemplateWithMetadata, variables: InputVariables) -> str:
        # Extract messages JSON to enable formatting of individual content fields of each message. If we do not
        # extract the JSON, current variable interpolation will fail on JSON curly braces.
        messages_as_json: List[Dict[str, str]] = json.loads(prompt_template.content)
        formatted_messages = [
            {
                "content": bind_template_variables(message['content'], variables), "role": message['role']
            } for message in messages_as_json]
        return json.dumps(formatted_messages)

    def to_llm_syntax(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        return messages

    def call_service(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> CompletionResponse:
        messages = json.loads(formatted_prompt)
        completion = cast(ChatCompletion, self._call_openai(messages, provider_config, llm_parameters, stream=False))

        return CompletionResponse(
            content=completion.choices[0].message.content or '',
            is_complete=completion.choices[0].finish_reason == 'stop',
            openai_function_call=self.__maybe_function_call(completion),
        )

    # noinspection PyMethodMayBeStatic
    def __maybe_function_call(self, completion: ChatCompletion) -> Optional[OpenAIFunctionCall]:
        maybe_function_call = completion.choices[0].message.function_call
        if maybe_function_call:
            return OpenAIFunctionCall(
                name=maybe_function_call.name,
                arguments=maybe_function_call.arguments
            )
        return None

    def call_service_stream(
            self,
            formatted_prompt: str,
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        messages = json.loads(formatted_prompt)
        completion_stream = cast(Stream[ChatCompletionChunk],
                                 self._call_openai(messages, provider_config, llm_parameters, stream=True))
        for chunk in completion_stream:
            yield CompletionChunk(
                text=chunk.choices[0].delta.content or '',
                is_complete=chunk.choices[0].finish_reason == 'stop',
                openai_function_call=chunk.choices[0].delta.function_call
            )

    def continue_chat(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> ChatCompletionResponse:
        completion = cast(ChatCompletion, self._call_openai(messages, provider_config, llm_parameters, stream=False))

        message_history = copy(messages)
        message = completion.choices[0].message
        message_history.append(ChatMessage(
            role=message.role or '',
            content=message.content or ''
        ))
        return ChatCompletionResponse(
            content=message.content or '',
            message_history=message_history,
            is_complete=completion.choices[0].finish_reason == "stop"
        )

    def continue_chat_stream(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters
    ) -> Generator[CompletionChunk, None, None]:
        completion_stream = cast(Stream[ChatCompletionChunk],
                                 self._call_openai(messages, provider_config, llm_parameters, stream=True))
        for chunk in completion_stream:
            yield CompletionChunk(
                text=chunk.choices[0].delta.content or '',
                is_complete=chunk.choices[0].finish_reason == "stop"
            )


class OpenAIChat(OpenAIChatFlavor):
    record_format_type = "openai_chat"
    _model_params_with_defaults = LLMParameters({
        "model": "gpt-3.5-turbo"
    })

    def __init__(self) -> None:
        self.client: Optional[openai.OpenAI] = None

    @property
    def provider(self) -> str:
        return "openai"

    def get_openai_client(self, openai_config: Optional[OpenAIConfig]) -> openai.OpenAI:
        if self.client:
            return self.client

        if not openai_config:
            raise FreeplayConfigurationError(
                "Missing OpenAI key. Use a ProviderConfig to specify keys prior to getting completion.")

        self.client = openai.OpenAI(api_key=openai_config.api_key, base_url=openai_config.base_url)
        return self.client

    def _call_openai(
            self,
            messages: List[ChatMessage],
            provider_config: ProviderConfig,
            llm_parameters: LLMParameters,
            stream: bool
    ) -> Union[ChatCompletion, openai.Stream[ChatCompletionChunk]]:
        client = self.get_openai_client(provider_config.openai)
        try:
            return client.chat.completions.create(
                messages=cast(List[ChatCompletionMessageParam], messages),
                **self.get_model_params(llm_parameters),
                stream=stream,
            )
        except (BadRequestError, AuthenticationError) as e:
            raise LLMClientError("Unable to call OpenAI") from e
        except Exception as e:
            raise LLMServerError("Unable to call OpenAI") from e


class AzureOpenAIChat(OpenAIChatFlavor):
    record_format_type = "azure_openai_chat"

    def __init__(self) -> None:
        self.client: Optional[openai.AzureOpenAI] = None

    @property
    def provider(self) -> str:
        return "azure"

    def get_azure_client(
            self,
            azure_config: Optional[AzureConfig],
            api_version: Optional[str] = None,
            endpoint: Optional[str] = None,
            deployment: Optional[str] = None,
    ) -> openai.AzureOpenAI:
        if self.client:
            return self.client

        if not azure_config:
            raise FreeplayConfigurationError(
                "Missing Azure key. Use a ProviderConfig to specify keys prior to getting completion.")

        self.client = openai.AzureOpenAI(
            api_key=azure_config.api_key,
            api_version=api_version,
            azure_endpoint=endpoint or '',
            azure_deployment=deployment,
        )
        return self.client

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
        llm_parameters.pop('resource_name')

        client = self.get_azure_client(
            azure_config=provider_config.azure,
            api_version=api_version,
            endpoint=endpoint,
            deployment=deployment_id,
        )

        try:
            return client.chat.completions.create(
                messages=cast(List[ChatCompletionMessageParam], messages),
                **self.get_model_params(llm_parameters),
                stream=stream,
            )
        except (BadRequestError, AuthenticationError) as e:
            raise LLMClientError("Unable to call Azure") from e
        except Exception as e:
            raise LLMServerError("Unable to call Azure") from e


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
    def format(self, prompt_template: PromptTemplateWithMetadata, variables: InputVariables) -> str:
        # Extract messages JSON to enable formatting of individual content fields of each message. If we do not
        # extract the JSON, current variable interpolation will fail on JSON curly braces.
        messages_as_json: List[Dict[str, str]] = json.loads(prompt_template.content)
        formatted_messages = [
            {
                "content": bind_template_variables(message['content'], variables),
                "role": self.__to_anthropic_role(message['role'])
            } for message in messages_as_json]
        return json.dumps(formatted_messages)

    def to_llm_syntax(self, messages: List[ChatMessage]) -> str:
        formatted_messages = [
            ChatMessage(
                content=message['content'],
                role=self.__to_anthropic_role(message['role'])
            ) for message in messages
        ]
        return self.__to_anthropic_chat_format(formatted_messages)

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


def pick_flavor_from_config(completion_flavor: Optional[Flavor], ui_flavor_name: Optional[str]) -> Flavor:
    ui_flavor = Flavor.get_by_name(ui_flavor_name) if ui_flavor_name else None
    flavor = completion_flavor or ui_flavor

    if flavor is None:
        raise FreeplayConfigurationError(
            "Flavor must be configured on either the Freeplay client, completion call, "
            "or in the Freeplay UI. Unable to fulfill request.")

    return flavor


def get_chat_flavor_from_config(completion_flavor: Optional[Flavor], ui_flavor_name: Optional[str]) -> ChatFlavor:
    flavor = pick_flavor_from_config(completion_flavor, ui_flavor_name)
    return require_chat_flavor(flavor)


def require_chat_flavor(flavor: Flavor) -> ChatFlavor:
    if not isinstance(flavor, ChatFlavor):
        raise FreeplayConfigurationError('A Chat flavor is required to start a chat session.')

    return flavor
