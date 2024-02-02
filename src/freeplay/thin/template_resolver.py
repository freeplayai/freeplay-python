import json
from abc import ABC, abstractmethod
from pathlib import Path

from freeplay.completions import PromptTemplates, PromptTemplateWithMetadata
from freeplay.errors import FreeplayConfigurationError
from freeplay.support import CallSupport


class TemplateResolver(ABC):
    @abstractmethod
    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        pass


class APITemplateResolver(TemplateResolver):

    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        return self.call_support.get_prompts(
            project_id=project_id,
            tag=environment
        )


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
