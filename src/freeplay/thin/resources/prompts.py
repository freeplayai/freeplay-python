import json
from typing import Optional

from freeplay.completions import PromptTemplates
from freeplay.errors import FreeplayConfigurationError
from freeplay.flavors import Flavor
from freeplay.model import InputVariables
from freeplay.support import CallSupport
from freeplay.thin.model import TemplatePrompt, PromptInfo, FormattedPrompt


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
