import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from freeplay.model import InputVariables, MediaInputBase64, MediaInputUrl, TestRunInfo
from freeplay.support import CallSupport, SummaryStatistics


@dataclass
class CompletionTestCase:
    def __init__(
        self,
        test_case_id: str,
        variables: InputVariables,
        output: Optional[str],
        history: Optional[List[Dict[str, str]]],
        custom_metadata: Optional[Dict[str, str]],
        media_variables: Optional[
            Dict[str, Union[MediaInputBase64, MediaInputUrl]]
        ] = None,
    ):
        self.id = test_case_id
        self.variables = variables
        self.output = output
        self.history = history
        self.custom_metadata = custom_metadata
        self.media_variables = media_variables


class TestCase(CompletionTestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "'TestCase' is deprecated; use 'CompletionTestCase' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class TraceTestCase:
    def __init__(
        self,
        test_case_id: str,
        input: str,
        output: Optional[str],
        custom_metadata: Optional[Dict[str, str]],
    ):
        self.id = test_case_id
        self.input = input
        self.output = output
        self.custom_metadata = custom_metadata

@dataclass
class TestRun:
    def __init__(
        self,
        test_run_id: str,
        test_cases: List[CompletionTestCase] = [],
        trace_test_cases: List[TraceTestCase] = [],
    ):
        self.test_run_id = test_run_id
        self.test_cases = test_cases
        self.trace_test_cases = trace_test_cases

    def __must_not_be_both_trace_and_completion(self) -> None:
        if (
            self.test_cases
            and len(self.test_cases) > 0
            and self.trace_test_cases
            and len(self.trace_test_cases) > 0
        ):
            raise ValueError("Test case and trace test case cannot both be present")

    def get_test_cases(self) -> List[CompletionTestCase]:
        self.__must_not_be_both_trace_and_completion()
        if len(self.trace_test_cases) > 0:
            raise ValueError(
                "Completion test cases are not present. Please use get_trace_test_cases() instead."
            )
        return self.test_cases

    def get_trace_test_cases(self) -> List[TraceTestCase]:
        self.__must_not_be_both_trace_and_completion()
        if len(self.test_cases) > 0:
            raise ValueError(
                "Trace test cases are not present. Please use get_test_cases() instead."
            )
        return self.trace_test_cases

    def get_test_run_info(self, test_case_id: str) -> TestRunInfo:
        return TestRunInfo(self.test_run_id, test_case_id)


@dataclass
class TestRunResults:
    def __init__(
            self,
            name: str,
            description: str,
            test_run_id: str,
            summary_statistics: SummaryStatistics,
    ):
        self.name = name
        self.description = description
        self.test_run_id = test_run_id
        self.summary_statistics = summary_statistics


class TestRuns:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def create(
            self,
            project_id: str,
            testlist: str,
            include_outputs: bool = False,
            name: Optional[str] = None,
            description: Optional[str] = None,
            flavor_name: Optional[str] = None
    ) -> TestRun:
        test_run = self.call_support.create_test_run(
            project_id, testlist, include_outputs, name, description, flavor_name)
        test_cases = [
            CompletionTestCase(
                test_case_id=test_case.id,
                variables=test_case.variables,
                output=test_case.output,
                history=test_case.history,
                custom_metadata=test_case.custom_metadata,
                media_variables=test_case.media_variables,
            )
            for test_case in test_run.test_cases
        ]
        trace_test_cases = [
            TraceTestCase(test_case_id=test_case.id,
                          input=test_case.input,
                          output=test_case.output,
                          custom_metadata=test_case.custom_metadata)
            for test_case in test_run.trace_test_cases
        ]

        return TestRun(test_run.test_run_id, test_cases, trace_test_cases)

    def get(self, project_id: str, test_run_id: str) -> TestRunResults:
        test_run_results = self.call_support.get_test_run_results(project_id, test_run_id)
        return TestRunResults(
            test_run_results.name,
            test_run_results.description,
            test_run_results.test_run_id,
            test_run_results.summary_statistics
        )
