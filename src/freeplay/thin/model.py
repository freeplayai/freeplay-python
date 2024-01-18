from dataclasses import dataclass
from typing import Optional, List, cast, Union

from freeplay.completions import OpenAIFunctionCall, ChatMessage
from freeplay.flavors import Flavor
from freeplay.llm_parameters import LLMParameters
from freeplay.model import InputVariables
from freeplay.utils import bind_template_variables


@dataclass
class CallInfo:
    provider: str
    model: str
    start_time: float
    end_time: float
    model_parameters: LLMParameters


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

    def get_call_info(self, start_time: float, end_time: float) -> CallInfo:
        return CallInfo(
            self.provider,
            self.model,
            start_time,
            end_time,
            self.model_parameters
        )


@dataclass
class ResponseInfo:
    is_complete: bool
    function_call_response: Optional[OpenAIFunctionCall] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None


@dataclass
class TestRunInfo:
    test_run_id: str
    test_case_id: str


@dataclass
class RecordPayload:
    all_messages: List[dict[str, str]]
    inputs: InputVariables
    session_id: str

    prompt_info: PromptInfo
    call_info: CallInfo
    response_info: ResponseInfo
    test_run_info: Optional[TestRunInfo] = None


@dataclass
class Session:
    session_id: str


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


@dataclass
class TestCase:
    def __init__(self, test_case_id: str, variables: InputVariables):
        self.id = test_case_id
        self.variables = variables


@dataclass
class TestRun:
    def __init__(
            self,
            test_run_id: str,
            test_cases: List[TestCase]
    ):
        self.test_run_id = test_run_id
        self.test_cases = test_cases

    def get_test_cases(self) -> List[TestCase]:
        return self.test_cases

    def get_test_run_info(self, test_case_id: str) -> TestRunInfo:
        return TestRunInfo(self.test_run_id, test_case_id)
