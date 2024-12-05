from dataclasses import dataclass
from typing import List, Optional, Dict

from freeplay.model import InputVariables
from freeplay.resources.recordings import TestRunInfo
from freeplay.support import CallSupport, SummaryStatistics


@dataclass
class TestCase:
    def __init__(
            self,
            test_case_id: str,
            variables: InputVariables,
            output: Optional[str],
            history: Optional[List[Dict[str, str]]]
    ):
        self.id = test_case_id
        self.variables = variables
        self.output = output
        self.history = history


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
            TestCase(test_case_id=test_case.id,
                     variables=test_case.variables,
                     output=test_case.output,
                     history=test_case.history)
            for test_case in test_run.test_cases
        ]

        return TestRun(test_run.test_run_id, test_cases)

    def get(self, project_id: str, test_run_id: str) -> TestRunResults:
        test_run_results = self.call_support.get_test_run_results(project_id, test_run_id)
        return TestRunResults(
            test_run_results.name,
            test_run_results.description,
            test_run_results.test_run_id,
            test_run_results.summary_statistics
        )
