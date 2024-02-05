import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union, cast

from freeplay.completions import PromptTemplates, ChatMessage, PromptTemplateWithMetadata
from freeplay.errors import FreeplayConfigurationError
from freeplay.flavors import Flavor
from freeplay.llm_parameters import LLMParameters
from freeplay.model import InputVariables
from freeplay.support import CallSupport
from freeplay.utils import bind_template_variables


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


class FormattedPrompt:
    def __init__(
            self,
            prompt_info: PromptInfo,
            messages: List[dict[str, str]],
            formatted_prompt: Union[str, List[dict[str, str]]]
    ):
        self.prompt_info = prompt_info
        self.messages = messages
        self.llm_prompt = formatted_prompt

    def all_messages(
            self,
            new_message: dict[str, str]
    ) -> List[dict[str, str]]:
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
        llm_format = flavor.to_llm_syntax(cast(List[ChatMessage], self.messages))

        return FormattedPrompt(
            self.prompt_info,
            self.messages,
            cast(Union[str, List[dict[str, str]]], llm_format)
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


class TemplateResolver(ABC):
    @abstractmethod
    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        pass


class FilesystemTemplateResolver(TemplateResolver):

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

            prompt_list.append(PromptTemplateWithMetadata(
                prompt_template_id=json_dom.get('prompt_template_id'),
                prompt_template_version_id=json_dom.get('prompt_template_version_id'),
                name=json_dom.get('name'),
                content=json_dom.get('content'),
                flavor_name=json_dom.get('metadata').get('flavor_name'),
                params=json_dom.get('metadata').get('params')
            ))

        return PromptTemplates(prompt_list)

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


class APITemplateResolver(TemplateResolver):

    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        return self.call_support.get_prompts(
            project_id=project_id,
            tag=environment
        )


class Prompts:
    def __init__(self, call_support: CallSupport, template_resolver: TemplateResolver) -> None:
        self.call_support = call_support
        self.template_resolver = template_resolver

    def get_all(self, project_id: str, environment: str) -> PromptTemplates:
        return self.call_support.get_prompts(project_id=project_id, tag=environment)

    def get(self, project_id: str, template_name: str, environment: str) -> TemplatePrompt:
        prompt_templates = self.template_resolver.get_prompts(project_id, environment)
        prompt_template = self.call_support.find_template_by_name(prompt_templates, template_name)

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
