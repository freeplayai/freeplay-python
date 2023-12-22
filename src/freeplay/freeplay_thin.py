from .completions import PromptTemplates
from .errors import FreeplayConfigurationError
from .freeplay import CallSupport
from .record import DefaultRecordProcessor


class FreeplayThin:
    def __init__(
            self,
            freeplay_api_key: str,
            api_base: str
    ) -> None:
        if not freeplay_api_key or not freeplay_api_key.strip():
            raise FreeplayConfigurationError("Freeplay API key not set. It must be set to the Freeplay API.")

        self.call_support = CallSupport(freeplay_api_key, api_base, DefaultRecordProcessor(freeplay_api_key, api_base))
        self.freeplay_api_key = freeplay_api_key
        self.api_base = api_base

    def get_prompts(self, project_id: str, tag: str) -> PromptTemplates:
        return self.call_support.get_prompts(project_id=project_id, tag=tag)