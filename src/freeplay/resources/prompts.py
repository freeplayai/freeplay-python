import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Union, cast, Any

from freeplay.errors import FreeplayConfigurationError, FreeplayClientError
from freeplay.llm_parameters import LLMParameters
from freeplay.model import InputVariables
from freeplay.support import PromptTemplate, PromptTemplates, PromptTemplateMetadata
from freeplay.support import CallSupport
from freeplay.utils import bind_template_variables


class MissingFlavorError(FreeplayConfigurationError):
    def __init__(self, flavor_name: str):
        super().__init__(
            f'Configured flavor ({flavor_name}) not found in SDK. Please update your SDK version or configure '
            'a different model in the Freeplay UI.'
        )


# SDK-Exposed Classes
@dataclass
class PromptInfo:
    prompt_template_id: str
    prompt_template_version_id: str
    template_name: str
    environment: str
    model_parameters: LLMParameters
    provider_info: Optional[Dict[str, Any]]
    provider: str
    model: str
    flavor_name: str


class FormattedPrompt:
    def __init__(
            self,
            prompt_info: PromptInfo,
            messages: List[Dict[str, str]],
            formatted_prompt: Union[str, List[Dict[str, str]]]
    ):
        self.prompt_info = prompt_info
        self.messages = messages
        self.llm_prompt = formatted_prompt

    def all_messages(
            self,
            new_message: Dict[str, str]
    ) -> List[Dict[str, str]]:
        return self.messages + [new_message]


class BoundPrompt:
    def __init__(
            self,
            prompt_info: PromptInfo,
            messages: List[Dict[str, str]]
    ):
        self.prompt_info = prompt_info
        self.messages = messages

    @staticmethod
    def __to_anthropic_role(role: str) -> str:
        if role == 'assistant' or role == 'Assistant':
            return 'Assistant'
        else:
            # Anthropic does not support system role for now.
            return 'Human'

    @staticmethod
    def __format_messages_for_flavor(flavor_name: str, messages: List[Dict[str, str]]) -> Union[
        str, List[Dict[str, str]]]:
        if flavor_name == 'azure_openai_chat' or flavor_name == 'openai_chat':
            return messages
        elif flavor_name == 'anthropic_chat':
            formatted_messages = []
            for message in messages:
                role = BoundPrompt.__to_anthropic_role(message['role'])
                formatted_messages.append(f"{role}: {message['content']}")
            formatted_messages.append('Assistant:')

            return "\n\n" + "\n\n".join(formatted_messages)
        raise MissingFlavorError(flavor_name)

    def format(
            self,
            flavor_name: Optional[str] = None
    ) -> FormattedPrompt:
        final_flavor = flavor_name or self.prompt_info.flavor_name
        llm_format = BoundPrompt.__format_messages_for_flavor(final_flavor, self.messages)

        return FormattedPrompt(
            self.prompt_info,
            self.messages,
            llm_format,
        )


class TemplatePrompt:
    def __init__(
            self,
            prompt_info: PromptInfo,
            messages: List[Dict[str, str]]
    ):
        self.prompt_info = prompt_info
        self.messages = messages

    def bind(self, variables: InputVariables) -> BoundPrompt:
        bound_messages = [
            {'role': message['role'], 'content': bind_template_variables(message['content'], variables)}
            for message in self.messages
        ]
        return BoundPrompt(self.prompt_info, bound_messages)


class TemplateResolver(ABC):
    @abstractmethod
    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        pass

    @abstractmethod
    def get_prompt(self, project_id: str, template_name: str, environment: str) -> PromptTemplate:
        pass


class FilesystemTemplateResolver(TemplateResolver):
    # If you think you need a change here, be sure to check the server as the translations must match. Once we have
    # all the SDKs and all customers on the new common format, this translation can go away.
    __role_translations = {
        'system': 'system',
        'user': 'user',
        'assistant': 'assistant',
        'Assistant': 'assistant',
        'Human': 'user'  # Don't think we ever store this, but in case...
    }

    def __init__(self, freeplay_directory: Path):
        FilesystemTemplateResolver.__validate_freeplay_directory(freeplay_directory)
        self.prompts_directory = freeplay_directory / "freeplay" / "prompts"

    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        self.__validate_prompt_directory(project_id, environment)

        directory = self.prompts_directory / project_id / environment
        prompt_file_paths = directory.glob("*.json")

        prompt_list = []
        for prompt_file_path in prompt_file_paths:
            json_dom = json.loads(prompt_file_path.read_text())
            prompt_list.append(self.__render_into_v2(json_dom))

        return PromptTemplates(prompt_list)

    def get_prompt(self, project_id: str, template_name: str, environment: str) -> PromptTemplate:
        self.__validate_prompt_directory(project_id, environment)

        expected_file: Path = self.prompts_directory / project_id / environment / f"{template_name}.json"

        if not expected_file.exists():
            raise FreeplayClientError(
                f"Could not find prompt with name {template_name} for project "
                f"{project_id} in environment {environment}"
            )

        json_dom = json.loads(expected_file.read_text())
        return self.__render_into_v2(json_dom)

    @staticmethod
    def __render_into_v2(json_dom: Dict[str, Any]) -> PromptTemplate:
        format_version = json_dom.get('format_version')

        if format_version == 2:
            metadata = json_dom['metadata']
            flavor_name = metadata.get('flavor')
            model = metadata.get('model')

            return PromptTemplate(
                format_version=2,
                prompt_template_id=json_dom.get('prompt_template_id'),  # type: ignore
                prompt_template_version_id=json_dom.get('prompt_template_version_id'),  # type: ignore
                prompt_template_name=json_dom.get('prompt_template_name'),  # type: ignore
                content=FilesystemTemplateResolver.__normalize_roles(json_dom['content']),
                metadata=PromptTemplateMetadata(
                    provider=FilesystemTemplateResolver.__flavor_to_provider(flavor_name),
                    flavor=flavor_name,
                    model=model,
                    params=metadata.get('params'),
                    provider_info=metadata.get('provider_info')
                )
            )
        else:
            metadata = json_dom['metadata']

            flavor_name = metadata.get('flavor_name')
            params = metadata.get('params')
            model = params.pop('model') if 'model' in params else None

            return PromptTemplate(
                format_version=2,
                prompt_template_id=json_dom.get('prompt_template_id'),  # type: ignore
                prompt_template_version_id=json_dom.get('prompt_template_version_id'),  # type: ignore
                prompt_template_name=json_dom.get('name'),  # type: ignore
                content=FilesystemTemplateResolver.__normalize_roles(json.loads(str(json_dom['content']))),
                metadata=PromptTemplateMetadata(
                    provider=FilesystemTemplateResolver.__flavor_to_provider(flavor_name),
                    flavor=flavor_name,
                    model=model,
                    params=params,
                    provider_info=None
                )
            )

    @staticmethod
    def __normalize_roles(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        normalized = []
        for message in messages:
            role = FilesystemTemplateResolver.__role_translations.get(message['role']) or message['role']
            normalized.append({'role': role, 'content': message['content']})
        return normalized

    @staticmethod
    def __validate_freeplay_directory(freeplay_directory: Path) -> None:
        if not freeplay_directory.is_dir():
            raise FreeplayConfigurationError(
                "Path for prompt templates is not a valid directory (%s)" % freeplay_directory
            )

        prompts_directory = freeplay_directory / "freeplay" / "prompts"
        if not prompts_directory.is_dir():
            raise FreeplayConfigurationError(
                "Invalid path for prompt templates (%s). "
                "Did not find a freeplay/prompts directory underneath." % freeplay_directory
            )

    def __validate_prompt_directory(self, project_id: str, environment: str) -> None:
        maybe_prompt_dir = self.prompts_directory / project_id / environment
        if not maybe_prompt_dir.is_dir():
            raise FreeplayConfigurationError(
                "Could not find prompt template directory for project ID %s and environment %s." %
                (project_id, environment)
            )

    @staticmethod
    def __flavor_to_provider(flavor: str) -> str:
        flavor_provider = {
            'azure_openai_chat': 'azure',
            'anthropic_chat': 'anthropic',
            'openai_chat': 'openai',
        }
        provider = flavor_provider.get(flavor)
        if not provider:
            raise MissingFlavorError(flavor)
        return provider


class APITemplateResolver(TemplateResolver):

    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        return self.call_support.get_prompts(
            project_id=project_id,
            environment=environment
        )

    def get_prompt(self, project_id: str, template_name: str, environment: str) -> PromptTemplate:
        return self.call_support.get_prompt(
            project_id=project_id,
            template_name=template_name,
            environment=environment
        )


class Prompts:
    def __init__(self, call_support: CallSupport, template_resolver: TemplateResolver) -> None:
        self.call_support = call_support
        self.template_resolver = template_resolver

    def get_all(self, project_id: str, environment: str) -> PromptTemplates:
        return self.call_support.get_prompts(project_id=project_id, environment=environment)

    def get(self, project_id: str, template_name: str, environment: str) -> TemplatePrompt:
        prompt = self.template_resolver.get_prompt(project_id, template_name, environment)

        params = prompt.metadata.params
        model = prompt.metadata.model

        if not model:
            raise FreeplayConfigurationError(
                "Model must be configured in the Freeplay UI. Unable to fulfill request.")

        if not prompt.metadata.flavor:
            raise FreeplayConfigurationError(
                "Flavor must be configured in the Freeplay UI. Unable to fulfill request.")

        if not prompt.metadata.provider:
            raise FreeplayConfigurationError(
                "Provider must be configured in the Freeplay UI. Unable to fulfill request.")

        prompt_info = PromptInfo(
            prompt_template_id=prompt.prompt_template_id,
            prompt_template_version_id=prompt.prompt_template_version_id,
            template_name=prompt.prompt_template_name,
            environment=environment,
            model_parameters=cast(LLMParameters, params) or LLMParameters({}),
            provider=prompt.metadata.provider,
            model=model,
            flavor_name=prompt.metadata.flavor,
            provider_info=prompt.metadata.provider_info
        )

        return TemplatePrompt(prompt_info, prompt.content)

    def get_formatted(
            self,
            project_id: str,
            template_name: str,
            environment: str,
            variables: InputVariables,
            flavor_name: Optional[str] = None
    ) -> FormattedPrompt:
        bound_prompt = self.get(
            project_id=project_id,
            template_name=template_name,
            environment=environment
        ).bind(variables=variables)

        return bound_prompt.format(flavor_name)
