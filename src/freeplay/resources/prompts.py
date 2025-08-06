import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    Union,
    cast,
    runtime_checkable,
)

from freeplay.errors import (
    FreeplayClientError,
    FreeplayConfigurationError,
    log_freeplay_client_warning,
)
from freeplay.llm_parameters import LLMParameters
from freeplay.model import (
    InputVariables,
    MediaInputMap,
    MediaInputUrl,
)
from freeplay.resources.adapters import (
    MediaContentBase64,
    MediaContentUrl,
    MissingFlavorError,
    TextContent,
    adaptor_for_flavor,
)
from freeplay.support import (
    CallSupport,
    HistoryTemplateMessage,
    MediaSlot,
    PromptTemplate,
    PromptTemplateMetadata,
    PromptTemplates,
    Role,
    TemplateChatMessage,
    TemplateMessage,
    ToolSchema,
)
from freeplay.utils import bind_template_variables, convert_provider_message_to_dict

logger = logging.getLogger(__name__)


class UnsupportedToolSchemaError(FreeplayConfigurationError):
    def __init__(self) -> None:
        super().__init__(
            'Tool schema not supported for this model and provider.'
        )
class VertexAIToolSchemaError(FreeplayConfigurationError):
    def __init__(self) -> None:
        super().__init__(
            'Vertex AI SDK not found. Install google-cloud-aiplatform to get proper Tool objects.'
        )


# Models ==

# A content block compatible with stainless generated SDKs (such as Anthropic
# and OpenAI). This lets us generate a dictionary from the stainless classes
# correctly. Intentionally over-permissive to allow schema evolution by the
# providers.
@runtime_checkable
class ProviderMessageProtocol(Protocol):
    def model_dump(self) -> Dict[str, Any]:
        pass


class MessageDict(TypedDict):
    role: str
    content: Any


# This type represents a struct or dict containing a role and content. The role
#  should be one of user, assistant or system. This type should be compatible
#  with OpenAI and Anthropic's message format, as well as most other SDKs. If
#  not using a common provider, use {'content': str, 'role': str} to record. If
#  using a common provider, this is usually the `.content` field.
ProviderMessage = Union[MessageDict, Dict[str, Any], ProviderMessageProtocol]

# DEPRECATED: Use ProviderMessage instead
GenericProviderMessage = ProviderMessage


# SDK-Exposed Classes
@dataclass
class PromptInfo:
    prompt_template_id: str
    prompt_template_version_id: str
    template_name: str
    environment: Optional[str]
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
        formatted_prompt: Optional[List[Dict[str, str]]] = None,
        formatted_prompt_text: Optional[str] = None,
        tool_schema: Optional[List[Dict[str, Any]]] = None
    ):
        # These two definitions allow us to operate on typed fields until we expose them as Any for client use.
        self._llm_prompt = formatted_prompt
        self._tool_schema = tool_schema

        self.prompt_info = prompt_info
        if formatted_prompt_text:
            self.llm_prompt_text = formatted_prompt_text

        maybe_system_content = next(
            (message['content'] for message in messages if message['role'] == 'system'), None)
        self.system_content = maybe_system_content

        self._messages = messages

    @property
    def messages(self) -> List[Dict[str, str]]:
        warnings.warn(
            "The 'messages' attribute is deprecated and will be removed in a future version. It is not formatted for the provider. Use 'llm_prompt' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._messages

    @property
    def llm_prompt(self) -> Any:
        return self._llm_prompt

    @property
    def tool_schema(self) -> Any:
        return self._tool_schema

    def all_messages(self, new_message: ProviderMessage) -> List[Dict[str, Any]]:
        converted_message = convert_provider_message_to_dict(new_message)
        return self._messages + [converted_message]


class BoundPrompt:
    def __init__(
        self,
        prompt_info: PromptInfo,
        messages: List[Dict[str, Any]],
        tool_schema: Optional[List[ToolSchema]] = None
    ):
        self.prompt_info = prompt_info
        self.messages = messages
        self.tool_schema = tool_schema

    @staticmethod
    def __format_tool_schema(flavor_name: str, tool_schema: List[ToolSchema]) -> Any:
        if flavor_name == 'anthropic_chat':
            return [{
                'name': tool_schema.name,
                'description': tool_schema.description,
                'input_schema': tool_schema.parameters
            } for tool_schema in tool_schema]
        elif flavor_name in ['openai_chat', 'azure_openai_chat']:
            return [
                {
                    'function': asdict(tool_schema),
                    'type': 'function'
                } for tool_schema in tool_schema
            ]
        elif flavor_name == "amazon_bedrock_converse":
            return {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool_schema.name,
                            "description": tool_schema.description,
                            "inputSchema": {"json": tool_schema.parameters},
                        }
                    }
                    for tool_schema in tool_schema
                ]
            }
        elif flavor_name == "gemini_chat":
            try:
                from vertexai.generative_models import Tool, FunctionDeclaration  # type: ignore[import-untyped]
                
                function_declarations = [
                    FunctionDeclaration(
                        name=tool_schema.name,
                        description=tool_schema.description,
                        parameters=tool_schema.parameters
                    )
                    for tool_schema in tool_schema
                ]
                return [Tool(function_declarations=function_declarations)]
            except ImportError:
                raise VertexAIToolSchemaError()

        raise UnsupportedToolSchemaError()

    def format(
        self,
        flavor_name: Optional[str] = None
    ) -> FormattedPrompt:
        final_flavor = flavor_name or self.prompt_info.flavor_name
        adapter = adaptor_for_flavor(final_flavor)
        formatted_prompt = adapter.to_llm_syntax(self.messages)
        formatted_tool_schema = BoundPrompt.__format_tool_schema(
            final_flavor,
            self.tool_schema
        ) if self.tool_schema else None

        if isinstance(formatted_prompt, str):
            return FormattedPrompt(
                prompt_info=self.prompt_info,
                messages=self.messages,
                formatted_prompt_text=formatted_prompt,
                tool_schema=formatted_tool_schema
            )
        else:
            return FormattedPrompt(
                prompt_info=self.prompt_info,
                messages=self.messages,
                formatted_prompt=formatted_prompt,
                tool_schema=formatted_tool_schema
            )




def extract_media_content(media_inputs: MediaInputMap, media_slots: List[MediaSlot]) -> List[
    Union[MediaContentBase64, MediaContentUrl]]:
    media_content: List[Union[MediaContentBase64, MediaContentUrl]] = []
    for slot in media_slots:
        file = media_inputs.get(slot.placeholder_name, None)
        if file is None:
            continue
        if isinstance(file, MediaInputUrl):
            media_content.append(MediaContentUrl(type=slot.type, url=file.url, slot_name=slot.placeholder_name))
        else:
            media_content.append(MediaContentBase64(type=slot.type, content_type=file.content_type, data=file.data, slot_name=slot.placeholder_name))

    return media_content


class TemplatePrompt:
    def __init__(
        self,
        prompt_info: PromptInfo,
        messages: List[TemplateMessage],
        tool_schema: Optional[List[ToolSchema]] = None
    ):
        self.prompt_info = prompt_info
        self.tool_schema = tool_schema
        self.messages = messages

    def bind(
        self,
        variables: InputVariables,
        history: Optional[Sequence[ProviderMessage]] = None,
        media_inputs: Optional[MediaInputMap] = None
    ) -> BoundPrompt:
        # check history for a system message
        history_clean = []
        if history:
            template_messages_contain_system = any(
                message.role == 'system' for message in self.messages if isinstance(message, TemplateChatMessage))
            history_dict = [convert_provider_message_to_dict(msg) for msg in history]
            for msg in history_dict:
                history_has_system = msg.get('role', None) == 'system'
                if history_has_system and template_messages_contain_system:
                    log_freeplay_client_warning("System message found in history, and prompt template."
                                                "Removing system message from the history")
                else:
                    history_clean.append(msg)

        has_history_placeholder = any(isinstance(message, HistoryTemplateMessage) for message in self.messages)
        if history and not has_history_placeholder:
            raise FreeplayClientError(
                "History provided for prompt that does not expect history")
        if has_history_placeholder and not history:
            log_freeplay_client_warning("History missing for prompt that expects history")

        bound_messages: List[Dict[str, Any]] = []
        if not media_inputs:
            media_inputs = {}
        for msg in self.messages:
            if isinstance(msg, HistoryTemplateMessage):
                bound_messages.extend(history_clean)
            else:
                media_content = extract_media_content(media_inputs, msg.media_slots)
                content = bind_template_variables(msg.content, variables)

                if media_content:
                    bound_messages.append({
                        'role': msg.role,
                        'content': [
                            TextContent(text=content),
                            *media_content
                        ],
                        'has_media': True,
                    })
                else:
                    bound_messages.append({
                        'role': msg.role,
                        'content': content},
                    )

        return BoundPrompt(self.prompt_info, bound_messages, self.tool_schema)


class TemplateResolver(ABC):
    @abstractmethod
    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        pass

    @abstractmethod
    def get_prompt(self, project_id: str, template_name: str, environment: str) -> PromptTemplate:
        pass

    @abstractmethod
    def get_prompt_version_id(self, project_id: str, template_id: str, version_id: str) -> PromptTemplate:
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

    def get_prompt_version_id(self, project_id: str, template_id: str, version_id: str) -> PromptTemplate:

        expected_file: Path = self.prompts_directory / project_id

        if not expected_file.exists():
            raise FreeplayClientError(
                f"Could not find project id {project_id}"
            )

        # read all files in the project directory
        prompt_file_paths = expected_file.glob("**/*.json")
        # find the file with the matching version id
        for prompt_file_path in prompt_file_paths:
            json_dom = json.loads(prompt_file_path.read_text())
            if json_dom.get('prompt_template_version_id') == version_id:
                return self.__render_into_v2(json_dom)

        raise FreeplayClientError(
            f"Could not find prompt with version id {version_id} for project {project_id}"
        )

    @staticmethod
    def __render_into_v2(json_dom: Dict[str, Any]) -> PromptTemplate:
        format_version = json_dom.get('format_version')

        if format_version and format_version >= 2:
            metadata = json_dom['metadata']
            flavor_name = metadata.get('flavor')
            model = metadata.get('model')

            return PromptTemplate(
                format_version=format_version,
                prompt_template_id=json_dom.get('prompt_template_id'),  # type: ignore
                prompt_template_version_id=json_dom.get('prompt_template_version_id'),  # type: ignore
                prompt_template_name=json_dom.get('prompt_template_name'),  # type: ignore
                content=FilesystemTemplateResolver.__normalize_messages(json_dom['content']),
                metadata=PromptTemplateMetadata(
                    provider=FilesystemTemplateResolver.__flavor_to_provider(flavor_name),
                    flavor=flavor_name,
                    model=model,
                    params=metadata.get('params'),
                    provider_info=metadata.get('provider_info')
                ),
                project_id=str(json_dom.get('project_id')),
                tool_schema=[ToolSchema(
                    name=schema.get('name'),
                    description=schema.get('description'),
                    parameters=schema.get('parameters')
                ) for schema in json_dom.get('tool_schema', [])] if json_dom.get('tool_schema') else None
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
                content=FilesystemTemplateResolver.__normalize_messages(json.loads(str(json_dom['content']))),
                metadata=PromptTemplateMetadata(
                    provider=FilesystemTemplateResolver.__flavor_to_provider(flavor_name),
                    flavor=flavor_name,
                    model=model,
                    params=params,
                    provider_info=None
                ),
                project_id=str(json_dom.get('project_id'))
            )

    @staticmethod
    def __normalize_messages(messages: List[Dict[str, Any]]) -> List[TemplateMessage]:
        normalized: List[TemplateMessage] = []
        for message in messages:
            if 'kind' in message:
                normalized.append(HistoryTemplateMessage(kind="history"))
            else:
                role = FilesystemTemplateResolver.__role_translations.get(message['role']) or message['role']
                media_slots: List[MediaSlot] = cast(List[MediaSlot], message.get('media_slots', []))
                normalized.append(
                    TemplateChatMessage(role=cast(Role, role), content=message['content'], media_slots=media_slots))
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

    def get_prompt_version_id(self, project_id: str, template_id: str, version_id: str) -> PromptTemplate:
        return self.call_support.get_prompt_version_id(
            project_id=project_id,
            template_id=template_id,
            version_id=version_id
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
            provider_info=prompt.metadata.provider_info,
        )

        return TemplatePrompt(prompt_info, prompt.content, prompt.tool_schema)

    def get_all_for_environment(self, environment: str) -> PromptTemplates:
        return self.call_support.get_prompts_for_environment(environment=environment)

    def get_by_version_id(self, project_id: str, template_id: str, version_id: str) -> TemplatePrompt:
        prompt = self.template_resolver.get_prompt_version_id(project_id, template_id, version_id)

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
            environment=prompt.environment if prompt.environment else '',
            model_parameters=cast(LLMParameters, params) or LLMParameters({}),
            provider=prompt.metadata.provider,
            model=model,
            flavor_name=prompt.metadata.flavor,
            provider_info=prompt.metadata.provider_info,
        )

        return TemplatePrompt(prompt_info, prompt.content, prompt.tool_schema)

    def get_formatted(
        self,
        project_id: str,
        template_name: str,
        environment: str,
        variables: InputVariables,
        history: Optional[Sequence[ProviderMessage]] = None,
        flavor_name: Optional[str] = None,
        media_inputs: Optional[MediaInputMap] = None,
    ) -> FormattedPrompt:
        bound_prompt = self.get(
            project_id=project_id,
            template_name=template_name,
            environment=environment
        ).bind(variables=variables, history=history, media_inputs=media_inputs)

        return bound_prompt.format(flavor_name)

    def get_formatted_by_version_id(
        self,
        project_id: str,
        template_id: str,
        version_id: str,
        variables: InputVariables,
        flavor_name: Optional[str] = None,
    ) -> FormattedPrompt:
        bound_prompt = self.get_by_version_id(
            project_id=project_id,
            template_id=template_id,
            version_id=version_id
        ).bind(variables=variables)

        return bound_prompt.format(flavor_name)
