from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union, cast
from uuid import uuid4

T = TypeVar("T")

from freeplay.llm_parameters import LLMParameters
from freeplay.model import (
    MediaInputMap,
    TestRunInfo,
    parse_media_variables,
)
from freeplay.resources.prompts import (
    FormattedPrompt,
    PromptInfo,
    TemplatePrompt,
)
from freeplay.resources.recordings import (
    CallInfo,
    RecordPayload,
    RecordResponse,
    Recordings,
)
from freeplay.resources.sessions import SessionInfo, TraceInfo
from freeplay.resources.test_runs import CompletionTestCase, TraceTestCase
from freeplay.support import (
    CallSupport,
    HistoryTemplateMessage,
    MediaSlot,
    MediaType,
    Role,
    TemplateChatMessage,
    TemplateMessage,
    ToolSchema,
)


@dataclass
class TestSuiteRunResultsSummary:
    human_evaluation: Optional[Dict[str, Any]]
    auto_evaluation: Optional[Dict[str, Any]]
    client_evaluation: Optional[Dict[str, Any]]


@dataclass
class TestSuiteRunResults:
    run_id: str
    suite_id: str
    status: str
    passed: Optional[bool]
    eval_results: Optional[Dict[str, Any]]
    summary_statistics: Optional[TestSuiteRunResultsSummary]


class TestSuiteRun:
    """Represents a running test suite execution. Provides lazy-paginated
    access to test cases and convenience methods for recording results."""

    def __init__(
        self,
        run_id: str,
        suite_id: str,
        project_id: str,
        target_type: str,
        total_test_cases: int,
        template_prompt: Optional[TemplatePrompt],
        call_support: CallSupport,
        recordings: Recordings,
        page_size: int = 50,
    ):
        self.run_id = run_id
        self.suite_id = suite_id
        self.project_id = project_id
        self.target_type = target_type
        self.total_test_cases = total_test_cases
        self._template_prompt = template_prompt
        self._call_support = call_support
        self._recordings = recordings
        self._page_size = page_size

    @property
    def test_cases(self) -> Iterator[CompletionTestCase]:
        """Lazy-paginated iterator over completion test cases.
        Only valid for prompt-type suites."""
        if self.target_type != "prompt":
            raise ValueError("This is an agent suite. Use trace_test_cases instead.")
        return self._paginated_test_cases(_parse_completion_test_case)

    @property
    def trace_test_cases(self) -> Iterator[TraceTestCase]:
        """Lazy-paginated iterator over trace test cases.
        Only valid for agent-type suites."""
        if self.target_type != "agent":
            raise ValueError("This is a prompt suite. Use test_cases instead.")
        return self._paginated_test_cases(_parse_trace_test_case)

    def _paginated_test_cases(
        self, parse_fn: Callable[[Dict[str, Any]], T]
    ) -> Iterator[T]:
        page = 1
        while True:
            data, has_next = self._call_support.get_test_suite_run_test_cases(
                self.project_id,
                self.suite_id,
                self.run_id,
                page=page,
                page_size=self._page_size,
            )
            for tc in data:
                yield parse_fn(tc)
            if not has_next:
                break
            page += 1

    def format_prompt(
        self,
        test_case: CompletionTestCase,
        media_inputs: Optional[MediaInputMap] = None,
    ) -> FormattedPrompt:
        """Bind test case variables to the suite's prompt template and format."""
        if self._template_prompt is None:
            raise ValueError(
                "No prompt template available. "
                "For agent suites, handle prompting in your agent code."
            )
        return self._template_prompt.bind(
            test_case.variables,
            history=test_case.history,
            media_inputs=(
                media_inputs
                if media_inputs is not None
                else test_case.media_variables
            ),
        ).format()

    def record(
        self,
        test_case: CompletionTestCase,
        all_messages: List[Dict[str, Any]],
        call_info: Optional[CallInfo] = None,
        session_info: Optional[SessionInfo] = None,
        eval_results: Optional[Dict[str, Union[bool, float]]] = None,
        **kwargs: Any,
    ) -> RecordResponse:
        """Record a completion result for this test case."""
        if self.target_type != "prompt":
            raise ValueError("This is an agent suite. Use record_trace instead.")
        if session_info is None:
            session_info = SessionInfo(session_id=str(uuid4()), custom_metadata=None)

        prompt_version_info = None
        if self._template_prompt is not None:
            prompt_version_info = self._template_prompt.prompt_info

        return self._recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=all_messages,
                session_info=session_info,
                inputs=test_case.variables,
                prompt_version_info=prompt_version_info,
                call_info=call_info,
                test_run_info=TestRunInfo(self.run_id, test_case.id),
                eval_results=eval_results,
                **kwargs,
            )
        )

    def record_trace(
        self,
        test_case: TraceTestCase,
        trace_info: TraceInfo,
        output: Any,
        eval_results: Optional[Dict[str, Union[bool, float]]] = None,
    ) -> None:
        """Record a trace result for this test case."""
        if self.target_type != "agent":
            raise ValueError("This is a prompt suite. Use record instead.")
        trace_info.record_output(
            self.project_id,
            output,
            eval_results=eval_results,
            test_run_info=TestRunInfo(self.run_id, test_case.id),
        )

    def get_test_run_info(self, test_case_id: str) -> TestRunInfo:
        """Get TestRunInfo for manual recording (escape hatch)."""
        return TestRunInfo(self.run_id, test_case_id)

    def get_results(self) -> TestSuiteRunResults:
        """Fetch pass/fail results for this run."""
        raw = self._call_support.get_test_suite_run_results(
            self.project_id, self.suite_id, self.run_id
        )
        return _parse_run_results(raw)


class TestSuites:
    """Test Suites resource. Provides execution capabilities for test suites."""

    def __init__(self, call_support: CallSupport, recordings: Recordings) -> None:
        self._call_support = call_support
        self._recordings = recordings

    def run(
        self,
        project_id: str,
        suite_id: str,
        environment: Optional[str] = None,
        name: Optional[str] = None,
        prompt_template_version_id: Optional[str] = None,
        page_size: int = 50,
    ) -> TestSuiteRun:
        """Trigger a client-orchestrated run of the given test suite and
        return a TestSuiteRun for iterating over test cases, recording
        results, and retrieving the pass/fail verdict.

        ``environment`` is the deployment environment tag (e.g. "production")
        used to resolve prompt template versions for prompt-type suites.
        Not needed for agent-type suites, or when pinning via
        ``prompt_template_version_id``.

        ``prompt_template_version_id`` optionally pins a specific prompt
        template version instead of resolving via ``environment``.

        ``name`` optionally overrides the auto-generated run name."""
        response = self._call_support.create_test_suite_run(
            project_id=project_id,
            suite_id=suite_id,
            environment=environment,
            name=name,
            prompt_template_version_id=prompt_template_version_id,
        )

        template_prompt = None
        if response.get("prompt_template") is not None:
            template_prompt = _build_template_prompt(response["prompt_template"])

        return TestSuiteRun(
            run_id=str(response["run_id"]),
            suite_id=str(response["suite_id"]),
            project_id=project_id,
            target_type=response["target_type"],
            total_test_cases=response["total_test_cases"],
            template_prompt=template_prompt,
            call_support=self._call_support,
            recordings=self._recordings,
            page_size=page_size,
        )

    def execute(
        self,
        project_id: str,
        suite_id: str,
        environment: Optional[str] = None,
        name: Optional[str] = None,
        prompt_template_version_id: Optional[str] = None,
    ) -> TestSuiteRunResults:
        """Trigger a server-side execution of the given test suite.

        The server runs all test cases, evaluates results, and returns
        the final pass/fail verdict. Use this when you don't need to
        supply your own LLM calls.

        ``environment`` optionally specifies the deployment environment
        tag used to resolve prompt template versions.

        ``prompt_template_version_id`` optionally pins a specific prompt
        template version instead of resolving via ``environment``.

        ``name`` optionally overrides the auto-generated run name."""
        response = self._call_support.execute_test_suite_run(
            project_id=project_id,
            suite_id=suite_id,
            environment=environment,
            name=name,
            prompt_template_version_id=prompt_template_version_id,
        )
        return _parse_run_results(response)


# --- Private helpers ---


def _parse_completion_test_case(tc: Dict[str, Any]) -> CompletionTestCase:
    return CompletionTestCase(
        test_case_id=tc["test_case_id"],
        variables=tc["variables"],
        output=tc.get("output"),
        history=tc.get("history"),
        custom_metadata=tc.get("custom_metadata"),
        media_variables=parse_media_variables(tc.get("media_variables")),
    )


def _parse_trace_test_case(tc: Dict[str, Any]) -> TraceTestCase:
    return TraceTestCase(
        test_case_id=tc["test_case_id"],
        input=tc["input"],
        output=tc.get("output"),
        custom_metadata=tc.get("custom_metadata"),
    )


def _parse_run_results(raw: Dict[str, Any]) -> TestSuiteRunResults:
    summary = raw.get("summary_statistics")
    summary_obj = None
    if summary is not None:
        summary_obj = TestSuiteRunResultsSummary(
            human_evaluation=summary.get("human_evaluation"),
            auto_evaluation=summary.get("auto_evaluation"),
            client_evaluation=summary.get("client_evaluation"),
        )
    return TestSuiteRunResults(
        run_id=str(raw["run_id"]),
        suite_id=str(raw["suite_id"]),
        status=raw.get("status") or "in-progress",
        passed=raw.get("passed"),
        eval_results=raw.get("eval_results"),
        summary_statistics=summary_obj,
    )


def _build_template_prompt(pt_dict: Dict[str, Any]) -> TemplatePrompt:
    metadata = pt_dict.get("metadata", {})
    content = _parse_template_messages(pt_dict.get("content", []))

    tool_schema = None
    if pt_dict.get("tool_schema"):
        tool_schema = [
            ToolSchema(
                name=s["name"],
                description=s["description"],
                parameters=s["parameters"],
            )
            for s in pt_dict["tool_schema"]
        ]

    prompt_info = PromptInfo(
        prompt_template_id=str(pt_dict["prompt_template_id"]),
        prompt_template_version_id=str(pt_dict["prompt_template_version_id"]),
        template_name=pt_dict["prompt_template_name"],
        environment=None,
        model_parameters=cast(LLMParameters, metadata.get("params"))
        or LLMParameters({}),
        provider=metadata.get("provider", ""),
        model=metadata.get("model", ""),
        flavor_name=metadata.get("flavor", ""),
        provider_info=metadata.get("provider_info"),
    )

    return TemplatePrompt(
        prompt_info=prompt_info,
        messages=content,
        tool_schema=tool_schema,
        output_schema=pt_dict.get("output_schema"),
    )


def _parse_template_messages(
    messages: List[Dict[str, Any]],
) -> List[TemplateMessage]:
    result: List[TemplateMessage] = []
    for msg in messages:
        if msg.get("kind") == "history":
            result.append(HistoryTemplateMessage(kind="history"))
        else:
            raw_media_slots = msg.get("media_slots", [])
            media_slots = (
                [
                    MediaSlot(
                        type=cast(MediaType, slot["type"]),
                        placeholder_name=slot["placeholder_name"],
                    )
                    for slot in raw_media_slots
                ]
                if raw_media_slots
                else []
            )
            result.append(
                TemplateChatMessage(
                    role=cast(Role, msg["role"]),
                    content=msg["content"],
                    media_slots=media_slots,
                )
            )
    return result
