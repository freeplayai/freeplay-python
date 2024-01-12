import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from .completions import (
    PromptTemplates,
    CompletionResponse,
    CompletionChunk,
    ChatCompletionResponse,
    ChatMessage
)
from .errors import FreeplayConfigurationError
from .flavors import Flavor, ChatFlavor, require_chat_flavor, get_chat_flavor_from_config
from .llm_parameters import LLMParameters
from .model import InputVariables
from .provider_config import ProviderConfig
from .record import (
    RecordProcessor,
    DefaultRecordProcessor
)
from .support import CallSupport

logger = logging.getLogger(__name__)
default_tag = 'latest'


class Session:
    def __init__(
            self,
            call_support: CallSupport,
            session_id: str,
            prompts: PromptTemplates,
            flavor: Optional[Flavor],
            provider_config: ProviderConfig,
            tag: str = default_tag,
            test_run_id: Optional[str] = None
    ) -> None:
        self.tag = tag
        self.call_support = call_support
        self.session_flavor = flavor
        self.provider_config = provider_config
        self.session_id = session_id
        self.prompts = prompts
        self.test_run_id = test_run_id

    def get_completion(
            self,
            template_name: str,
            variables: InputVariables,
            flavor: Optional[Flavor] = None,
            **kwargs: Any
    ) -> CompletionResponse:
        completion_flavor = flavor or self.session_flavor
        return self.call_support.prepare_and_make_call(self.session_id,
                                                       self.prompts,
                                                       template_name,
                                                       variables,
                                                       completion_flavor,
                                                       self.provider_config,
                                                       self.tag,
                                                       self.test_run_id,
                                                       completion_parameters=LLMParameters(kwargs))

    def get_completion_stream(
            self,
            template_name: str,
            variables: InputVariables,
            flavor: Optional[Flavor] = None,
            **kwargs: Any
    ) -> Generator[CompletionChunk, None, None]:
        completion_flavor = flavor or self.session_flavor
        return self.call_support.prepare_and_make_call_stream(self.session_id,
                                                              self.prompts,
                                                              template_name,
                                                              variables,
                                                              completion_flavor,
                                                              self.provider_config,
                                                              self.tag,
                                                              self.test_run_id,
                                                              completion_parameters=LLMParameters(kwargs))


class ChatSession(Session):
    def __init__(
            self,
            call_support: CallSupport,
            session_id: str,
            prompts: PromptTemplates,
            flavor: Optional[ChatFlavor],
            provider_config: ProviderConfig,
            template_name: str,
            variables: InputVariables,
            tag: str = default_tag,
            test_run_id: Optional[str] = None,
            messages: Optional[List[ChatMessage]] = None,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None,
    ) -> None:
        super().__init__(call_support, session_id, prompts, flavor, provider_config, tag, test_run_id)
        # A Chat Session tracks the template_name and variables for a set of chat completions.
        # Assumes these will be the same for subsequent chat messages.
        self.message_history = messages or []
        self.variables = variables
        self.metadata = metadata
        self.target_template = self.call_support.find_template_by_name(self.prompts, template_name)
        self.flavor = get_chat_flavor_from_config(flavor, self.target_template.flavor_name)
        self.__initial_messages = json.loads(self.flavor.format(self.target_template, self.variables))

    def last_message(self) -> Optional[ChatMessage]:
        return self.message_history[len(self.message_history) - 1]

    def store_new_messages(self, new_messages: List[ChatMessage]) -> None:
        for message in new_messages:
            self.message_history.append({
                "role": message["role"],
                "content": message["content"]
            })

    def start_chat(self, **kwargs: Any) -> ChatCompletionResponse:
        response = self.call_support.prepare_and_make_chat_call(
            self.session_id,
            flavor=self.flavor,
            provider_config=self.provider_config,
            tag=self.tag,
            test_run_id=self.test_run_id,
            target_template=self.target_template,
            variables=self.variables,
            message_history=self.__initial_messages,
            new_messages=None,
            completion_parameters=LLMParameters(kwargs),
            metadata=self.metadata,
        )

        self.store_new_messages(response.message_history)
        return response

    def start_chat_stream(self, **kwargs: Any) -> Generator[CompletionChunk, None, None]:
        return self.continue_chat_stream(new_messages=None, **kwargs)

    def aggregate_message_from_response(
            self,
            response: Generator[CompletionChunk, None, None]
    ) -> Generator[CompletionChunk, Any, None]:
        message: ChatMessage = {
            "role": "assistant",
            "content": ""
        }

        for chunk in response:
            message["content"] += chunk.text
            yield chunk

        self.message_history.append(message)

    def continue_chat(
            self,
            new_messages: Optional[List[ChatMessage]] = None,
            **kwargs: Any
    ) -> ChatCompletionResponse:

        response = self.call_support.prepare_and_make_chat_call(
            self.session_id,
            flavor=self.flavor,
            provider_config=self.provider_config,
            tag=self.tag,
            test_run_id=self.test_run_id,
            target_template=self.target_template,
            variables=self.variables,
            message_history=self.message_history,
            new_messages=new_messages,
            completion_parameters=LLMParameters(kwargs),
            metadata=self.metadata,
        )

        if new_messages is not None:
            self.store_new_messages(new_messages)
        if response.content:
            self.message_history.append(response.message_history[-1])

        return response

    def continue_chat_stream(
            self,
            new_messages: Optional[List[ChatMessage]] = None,
            **kwargs: Any
    ) -> Generator[CompletionChunk, None, None]:
        new_messages = new_messages or []
        if len(self.message_history) == 0:
            self.message_history = self.__initial_messages

        response = self.call_support.prepare_and_make_chat_call_stream(
            self.session_id,
            flavor=self.flavor,
            provider_config=self.provider_config,
            tag=self.tag,
            target_template=self.target_template,
            variables=self.variables,
            message_history=self.message_history,
            test_run_id=self.test_run_id,
            completion_parameters=LLMParameters(kwargs),
            metadata=self.metadata,
        )

        self.store_new_messages(new_messages)
        yield from self.aggregate_message_from_response(response)


@dataclass()
class FreeplayTestRun:
    def __init__(
            self,
            call_support: CallSupport,
            flavor: Optional[Flavor],
            provider_config: ProviderConfig,
            test_run_id: str,
            inputs: List[InputVariables]
    ):
        self.call_support = call_support
        self.flavor = flavor
        self.provider_config = provider_config
        self.test_run_id = test_run_id
        self.inputs = inputs

    def get_inputs(self) -> List[InputVariables]:
        return self.inputs

    def create_session(self, project_id: str, tag: str = default_tag) -> Session:
        session_id = self.call_support.create_session_id()
        prompts = self.call_support.get_prompts(project_id, tag)
        return Session(self.call_support, session_id, prompts, self.flavor, self.provider_config,
                       tag, self.test_run_id)


# This SDK prototype does not support full functionality of either OpenAI's API or Freeplay's
# The simplifications are:
#  - Always assumes there is a single choice returned, does not support multiple
#  - Does not support an "escape hatch" to allow use of features we don't explicitly expose


class Freeplay:
    def __init__(
            self,
            freeplay_api_key: str,
            api_base: str,
            provider_config: ProviderConfig,
            flavor: Optional[Flavor] = None,
            record_processor: Optional[RecordProcessor] = None,
            **kwargs: Any
    ) -> None:
        if not freeplay_api_key or not freeplay_api_key.strip():
            raise FreeplayConfigurationError("Freeplay API key not set. It must be set to use the Freeplay API.")
        provider_config.validate()

        self.__record_processor = record_processor or DefaultRecordProcessor(freeplay_api_key, api_base)
        self.call_support = CallSupport(freeplay_api_key, api_base, self.__record_processor, **kwargs)
        self.provider_config = provider_config
        self.client_flavor = flavor
        self.freeplay_api_key = freeplay_api_key
        self.api_base = api_base

    def create_session(self, project_id: str, tag: str = default_tag) -> Session:
        session_id = self.call_support.create_session_id()
        prompts = self.call_support.get_prompts(project_id, tag)
        return Session(
            self.call_support,
            session_id,
            prompts,
            self.client_flavor,
            self.provider_config,
            tag)

    def restore_session(
            self,
            project_id: str,
            session_id: str,
            template_name: str,
            variables: InputVariables,
            tag: str = default_tag,
            flavor: Optional[Flavor] = None,
            **kwargs: Any
    ) -> CompletionResponse:
        prompts = self.call_support.get_prompts(project_id, tag)
        completion_flavor = flavor or self.client_flavor
        return self.call_support.prepare_and_make_call(
            session_id=session_id,
            prompts=prompts,
            template_name=template_name,
            variables=variables,
            flavor=completion_flavor,
            provider_config=self.provider_config,
            tag=tag,
            completion_parameters=LLMParameters(kwargs),
        )

    def get_completion(
            self,
            project_id: str,
            template_name: str,
            variables: InputVariables,
            tag: str = default_tag,
            flavor: Optional[Flavor] = None,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None,
            **kwargs: Any
    ) -> CompletionResponse:
        self.call_support.check_all_values_string_or_number(metadata)
        session_id = self.call_support.create_session_id()
        prompts = self.call_support.get_prompts(project_id, tag)
        completion_flavor = flavor or self.client_flavor

        return self.call_support.prepare_and_make_call(session_id,
                                                       prompts,
                                                       template_name,
                                                       variables,
                                                       completion_flavor,
                                                       self.provider_config,
                                                       tag,
                                                       completion_parameters=LLMParameters(kwargs),
                                                       metadata=metadata)

    def get_completion_stream(
            self,
            project_id: str,
            template_name: str,
            variables: InputVariables,
            tag: str = default_tag,
            flavor: Optional[Flavor] = None,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None,
            **kwargs: Any
    ) -> Generator[CompletionChunk, None, None]:
        self.call_support.check_all_values_string_or_number(metadata)
        session_id = self.call_support.create_session_id()
        prompts = self.call_support.get_prompts(project_id, tag)
        completion_flavor = flavor or self.client_flavor

        return self.call_support.prepare_and_make_call_stream(session_id,
                                                              prompts,
                                                              template_name,
                                                              variables,
                                                              completion_flavor,
                                                              self.provider_config,
                                                              tag,
                                                              completion_parameters=LLMParameters(kwargs),
                                                              metadata=metadata)

    def start_chat(
            self,
            project_id: str,
            template_name: str,
            variables: InputVariables,
            tag: str = default_tag,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None,
            **kwargs: Any
    ) -> Tuple[ChatSession, ChatCompletionResponse]:
        session = self.__create_chat_session(project_id, tag, template_name, variables, metadata)
        completion_response = session.start_chat(**kwargs)
        return session, completion_response

    def restore_chat_session(
            self,
            project_id: str,
            template_name: str,
            session_id: str,
            variables: InputVariables,
            tag: str = default_tag,
            messages: Optional[List[ChatMessage]] = None,
            flavor: Optional[ChatFlavor] = None) -> ChatSession:
        prompts = self.call_support.get_prompts(project_id, tag)
        chat_flavor = flavor or require_chat_flavor(self.client_flavor) if self.client_flavor else None
        return ChatSession(
            call_support=self.call_support,
            session_id=session_id,
            prompts=prompts,
            flavor=chat_flavor,
            provider_config=self.provider_config,
            template_name=template_name,
            variables=variables,
            tag=tag,
            messages=messages or []
        )

    def start_chat_stream(
            self,
            project_id: str,
            template_name: str,
            variables: InputVariables,
            tag: str = default_tag,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None,
            **kwargs: Any
    ) -> Tuple[ChatSession, Generator[CompletionChunk, None, None]]:
        """Returns a chat session, the base prompt template messages, and a streamed response from the LLM."""
        session = self.__create_chat_session(project_id, tag, template_name, variables, metadata)
        completion_response = session.start_chat_stream(**kwargs)
        return session, completion_response

    def create_test_run(self, project_id: str, testlist: str) -> FreeplayTestRun:
        test_run_response = self.call_support.create_test_run(project_id=project_id, testlist=testlist)

        return FreeplayTestRun(
            self.call_support,
            self.client_flavor,
            self.provider_config,
            test_run_response.test_run_id,
            [test_case.variables for test_case in test_run_response.test_cases]
        )

    def __create_chat_session(
            self,
            project_id: str,
            tag: str,
            template_name: str,
            variables: InputVariables,
            metadata: Optional[Dict[str, Union[str, int, float]]] = None) -> ChatSession:
        chat_flavor = require_chat_flavor(self.client_flavor) if self.client_flavor else None

        session_id = self.call_support.create_session_id()
        prompts = self.call_support.get_prompts(project_id, tag)
        return ChatSession(
            self.call_support,
            session_id,
            prompts,
            chat_flavor,
            self.provider_config,
            template_name,
            variables,
            tag,
            metadata=metadata)
