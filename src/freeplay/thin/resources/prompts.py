import json
from dataclasses import dataclass
from typing import Optional, List, Union, cast

from freeplay.completions import PromptTemplates, ChatMessage
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


class Prompts:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def get_all(self, project_id: str, environment: str) -> PromptTemplates:
        return self.call_support.get_prompts(project_id=project_id, tag=environment)

    def get(self, project_id: str, template_name: str, environment: str) -> TemplatePrompt:
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
