from dataclasses import asdict, dataclass, field
from json import JSONEncoder
from typing import Any, Dict, List, Literal, Optional, Union

from freeplay import api_support
from freeplay.api_support import try_decode
from freeplay.errors import FreeplayServerError, freeplay_response_error
from freeplay.model import (
    FeedbackValue,
    InputVariables,
    MediaInputBase64,
    MediaInputUrl,
    NormalizedMessage,
    TestRunInfo,
)

CustomMetadata = Optional[Dict[str, Union[str, int, float, bool]]]


@dataclass
class PromptTemplateMetadata:
    provider: Optional[str]
    flavor: Optional[str]
    model: Optional[str]
    params: Optional[Dict[str, Any]] = None
    provider_info: Optional[Dict[str, Any]] = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]


Role = Literal['system', 'user', 'assistant']


MediaType = Literal["image", "audio", "video", "file"]


@dataclass
class MediaSlot:
    type: MediaType
    placeholder_name: str


@dataclass
class TemplateChatMessage:
    role: Role
    content: str
    media_slots: List[MediaSlot] = field(default_factory=list)


@dataclass
class HistoryTemplateMessage:
    kind: Literal["history"]

TemplateMessage = Union[HistoryTemplateMessage, TemplateChatMessage]


@dataclass
class PromptTemplate:
    prompt_template_id: str
    prompt_template_version_id: str
    prompt_template_name: str
    content: List[TemplateMessage]
    metadata: PromptTemplateMetadata
    project_id: str
    format_version: int
    environment: Optional[str] = None
    tool_schema: Optional[List[ToolSchema]] = None


@dataclass
class PromptTemplates:
    prompt_templates: List[PromptTemplate]


@dataclass
class SummaryStatistics:
    auto_evaluation: Dict[str, Any]
    human_evaluation: Dict[str, Any]


@dataclass
class ProjectInfo:
    id: str
    name: str


@dataclass
class ProjectInfos:
    projects: List[ProjectInfo]


class PromptTemplateEncoder(JSONEncoder):
    def default(self, prompt_template: PromptTemplate) -> Dict[str, Any]:
        return prompt_template.__dict__


class TestCaseTestRunResponse:
    def __init__(self, test_case: Dict[str, Any]):
        self.id: str = test_case['test_case_id']
        self.variables: InputVariables = test_case['variables']
        self.output: Optional[str] = test_case.get('output')
        self.history: Optional[List[Dict[str, Any]]] = test_case.get('history')
        self.custom_metadata: Optional[Dict[str, str]] = test_case.get('custom_metadata')

        if test_case.get("media_variables", None):
            self.media_variables: Optional[
                Dict[str, Union[MediaInputBase64, MediaInputUrl]]
            ] = {}
            for name, media_data in test_case.get("media_variables", {}).items():
                media_type = media_data.get("type", "base64")
                if media_type == "url":
                    self.media_variables[name] = MediaInputUrl(
                        type="url",
                        url=media_data["url"],
                    )
                else:
                    self.media_variables[name] = MediaInputBase64(
                        type="base64",
                        data=media_data["data"],
                        content_type=media_data["content_type"],
                    )
        else:
            self.media_variables = None


class TraceTestCaseTestRunResponse:
    def __init__(self, test_case: Dict[str, Any]):
        self.id: str = test_case['test_case_id']
        self.input: str = test_case['input']
        self.output: Optional[str] = test_case.get('output')
        self.custom_metadata: Optional[Dict[str, str]] = test_case.get('custom_metadata')


class TestRunResponse:
    def __init__(
            self,
            test_run_id: str,
            test_cases: Optional[List[Dict[str, Any]]],
            trace_test_cases: Optional[List[Dict[str, Any]]]
    ):
        if test_cases and trace_test_cases:
            raise ValueError("Test cases and trace test cases cannot both be present.")

        self.test_cases = [
            TestCaseTestRunResponse(test_case)
            for test_case in (test_cases or []) if test_case is not None
        ]
        self.trace_test_cases = [
            TraceTestCaseTestRunResponse(test_case)
            for test_case in (trace_test_cases or []) if test_case is not None
        ]

        self.test_run_id = test_run_id


class TestRunRetrievalResponse:
    def __init__(
            self,
            name: str,
            description: str,
            test_run_id: str,
            summary_statistics: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.test_run_id = test_run_id
        self.summary_statistics = SummaryStatistics(
            auto_evaluation=summary_statistics['auto_evaluation'],
            human_evaluation=summary_statistics['human_evaluation']
        )


class DatasetTestCaseRequest:
    def __init__(self, history: Optional[List[NormalizedMessage]], inputs: InputVariables,
                 metadata: Optional[Dict[str, str]], output: Optional[str]) -> None:
        self.history: Optional[List[NormalizedMessage]] = history
        self.inputs: InputVariables = inputs
        self.metadata: Optional[Dict[str, str]] = metadata
        self.output: Optional[str] = output


class DatasetTestCaseResponse:
    def __init__(self, test_case: Dict[str, Any]):
        self.values: InputVariables = test_case['values']
        self.id: str = test_case['id']
        self.output: Optional[str] = test_case.get('output')
        self.history: Optional[List[NormalizedMessage]] = test_case.get('history')
        self.metadata: Optional[Dict[str, str]] = test_case.get('metadata')


class DatasetTestCasesRetrievalResponse:
    def __init__(self, test_cases: List[Dict[str, Any]]) -> None:
        self.test_cases = [
            DatasetTestCaseResponse(test_case)
            for test_case in test_cases
        ]


class CallSupport:
    def __init__(
            self,
            freeplay_api_key: str,
            api_base: str
    ) -> None:
        self.api_base = api_base
        self.freeplay_api_key = freeplay_api_key

    def get_prompts(self, project_id: str, environment: str) -> PromptTemplates:
        response = api_support.get_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/v2/projects/{project_id}/prompt-templates/all/{environment}'
        )

        if response.status_code != 200:
            raise freeplay_response_error("Error getting prompt templates", response)

        maybe_prompts = try_decode(PromptTemplates, response.content)
        if maybe_prompts is None:
            raise FreeplayServerError('Failed to parse prompt templates from server')

        return maybe_prompts

    def get_prompts_for_environment(self, environment: str) -> PromptTemplates:
        projects_response = api_support.get_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/v2/projects/all'
        )
        if projects_response.status_code != 200:
            raise freeplay_response_error("Error getting prompt templates", projects_response)

        maybe_projects: Optional[ProjectInfos] = try_decode(ProjectInfos, projects_response.content)
        if maybe_projects is None:
            raise FreeplayServerError('Failed to parse list of projects from server')

        prompt_templates = PromptTemplates([])
        for project in maybe_projects.projects:
            prompt_templates.prompt_templates.extend(
                self.get_prompts(project.id, environment).prompt_templates
            )

        return prompt_templates

    def get_prompt(self, project_id: str, template_name: str, environment: str) -> PromptTemplate:
        response = api_support.get_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/v2/projects/{project_id}/prompt-templates/name/{template_name}',
            params={
                'environment': environment
            }
        )

        if response.status_code != 200:
            raise freeplay_response_error(
                f"Error getting prompt template {template_name} in project {project_id} "
                f"and environment {environment}",
                response
            )

        maybe_prompt = try_decode(PromptTemplate, response.content)
        if maybe_prompt is None:
            raise FreeplayServerError(
                f"Error handling prompt {template_name} in project {project_id} "
                f"and environment {environment}"
            )

        return maybe_prompt

    def get_prompt_version_id(self, project_id: str, template_id: str, version_id: str) -> PromptTemplate:
        response = api_support.get_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/v2/projects/{project_id}/prompt-templates/id/{template_id}/versions/{version_id}'
        )

        if response.status_code != 200:
            raise freeplay_response_error(
                f"Error getting version id {version_id} for template {template_id} in project {project_id}",
                response
            )

        maybe_prompt = try_decode(PromptTemplate, response.content)
        if maybe_prompt is None:
            raise FreeplayServerError(
                f"Error handling version id {version_id} for template {template_id} in project {project_id}"
            )

        return maybe_prompt

    def update_customer_feedback(
            self,
            project_id: str,
            completion_id: str,
            feedback: Dict[str, Union[bool, str, int, float]]
    ) -> None:
        response = api_support.post_raw(
            self.freeplay_api_key,
            f'{self.api_base}/v2/projects/{project_id}/completion-feedback/id/{completion_id}',
            feedback
        )
        if response.status_code != 201:
            raise freeplay_response_error("Error updating customer feedback", response)

    def update_trace_feedback(
            self, project_id: str, trace_id: str, feedback: Dict[str, FeedbackValue]
    ) -> None:
        response = api_support.post_raw(
            self.freeplay_api_key,
            f'{self.api_base}/v2/projects/{project_id}/trace-feedback/id/{trace_id}',
            feedback
        )
        if response.status_code != 201:
            raise freeplay_response_error(
                f'Error updating trace feedback for {trace_id} in project {project_id}',
                response
            )

    def create_test_run(
            self,
            project_id: str,
            testlist: str,
            include_outputs: bool = False,
            name: Optional[str] = None,
            description: Optional[str] = None,
            flavor_name: Optional[str] = None
    ) -> TestRunResponse:
        response = api_support.post_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/v2/projects/{project_id}/test-runs',
            payload={
                'dataset_name': testlist,
                'include_outputs': include_outputs,
                'test_run_name': name,
                'test_run_description': description,
                'flavor_name': flavor_name
            },
        )

        if response.status_code != 201:
            raise freeplay_response_error('Error while creating a test run.', response)

        json_dom = response.json()

        return TestRunResponse(json_dom['test_run_id'], json_dom['test_cases'], json_dom['trace_test_cases'])

    def get_test_run_results(
            self,
            project_id: str,
            test_run_id: str,
    ) -> TestRunRetrievalResponse:
        response = api_support.get_raw(
            api_key=self.freeplay_api_key,
            url=f'{self.api_base}/v2/projects/{project_id}/test-runs/id/{test_run_id}'
        )
        if response.status_code != 200:
            raise freeplay_response_error('Error while retrieving test run results.', response)

        json_dom = response.json()

        return TestRunRetrievalResponse(
            name=json_dom['name'],
            description=json_dom['description'],
            test_run_id=json_dom['id'],
            summary_statistics=json_dom['summary_statistics']
        )

    def record_trace(
            self,
            project_id: str,
            session_id: str,
            trace_id: str,
            input: str,
            output: str,
            agent_name: Optional[str] = None,
            custom_metadata: CustomMetadata = None,
            eval_results: Optional[Dict[str, Union[bool, float]]] = None,
            test_run_info: Optional[TestRunInfo] = None
    ) -> None:
        payload = {
            'agent_name': agent_name,
            'input': input,
            'output': output,
            'custom_metadata': custom_metadata,
            'eval_results': eval_results,
            'test_run_info': asdict(test_run_info) if test_run_info else None
        }
        response = api_support.post_raw(
            self.freeplay_api_key,
            f'{self.api_base}/v2/projects/{project_id}/sessions/{session_id}/traces/id/{trace_id}',
            payload
        )
        if response.status_code != 201:
            raise freeplay_response_error('Error while recording trace.', response)

    def delete_session(self, project_id: str, session_id: str) -> None:
        response = api_support.delete_raw(
            self.freeplay_api_key,
            f'{self.api_base}/v2/projects/{project_id}/sessions/{session_id}'
        )
        if response.status_code != 201:
            raise freeplay_response_error('Error while deleting session.', response)

    def create_test_cases(self, project_id: str, dataset_id: str, test_cases: List[DatasetTestCaseRequest]) -> None:
        examples = [
            {
                "history": test_case.history,
                "output": test_case.output,
                "metadata": test_case.metadata,
                "inputs": test_case.inputs
            } for test_case in test_cases]
        payload: Dict[str, Any] = {"examples": examples}
        url = f'{self.api_base}/v2/projects/{project_id}/datasets/id/{dataset_id}/test-cases'

        response = api_support.post_raw(self.freeplay_api_key, url, payload)
        if response.status_code != 201:
            raise freeplay_response_error('Error while creating test cases.', response)

    def get_test_cases(self, project_id: str, dataset_id: str) -> DatasetTestCasesRetrievalResponse:
        url = f'{self.api_base}/v2/projects/{project_id}/datasets/id/{dataset_id}/test-cases'
        response = api_support.get_raw(self.freeplay_api_key, url)

        if response.status_code != 200:
            raise freeplay_response_error('Error while getting test cases.', response)

        json_dom = response.json()

        return DatasetTestCasesRetrievalResponse(
            test_cases=[
                {
                    "history": jsn["history"],
                    "id": jsn["id"],
                    "output": jsn["output"],
                    "values": jsn["values"],
                    "metadata": jsn["metadata"] if 'metadata' in jsn.keys() else None
                } for jsn in json_dom
            ]
        )
