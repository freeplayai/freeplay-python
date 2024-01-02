from dataclasses import dataclass
from typing import Optional

from .errors import FreeplayConfigurationError


@dataclass
class OpenAIConfig:
    api_key: str
    base_url: Optional[str] = None

    def validate(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise FreeplayConfigurationError("OpenAI API key not set. It must be set to make calls to the service.")


@dataclass
class AzureConfig(OpenAIConfig):
    engine: Optional[str] = None
    api_version: Optional[str] = None

    def validate(self) -> None:
        super().validate()

        if self.api_version is None:
            raise FreeplayConfigurationError(
                "OpenAI API version not set. It must be set to make calls to the service.")

        if self.engine is None:
            raise FreeplayConfigurationError("Azure engine is not set. It must be set to make calls to the service.")


@dataclass
class AnthropicConfig:
    api_key: str


@dataclass
class ProviderConfig:
    anthropic: Optional[AnthropicConfig] = None
    openai: Optional[OpenAIConfig] = None
    azure: Optional[AzureConfig] = None

    def validate(self) -> None:
        if all(config is None for config in [self.anthropic, self.openai, self.azure]):
            FreeplayConfigurationError("At least one provider key must be set in ProviderConfig.")

        if self.openai is not None:
            self.openai.validate()
