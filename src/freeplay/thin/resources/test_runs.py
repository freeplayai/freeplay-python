from dataclasses import dataclass
from typing import List

from freeplay.model import InputVariables
from freeplay.support import CallSupport
from freeplay.thin.resources.recordings import TestRunInfo


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


class TestRuns:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def create(self, project_id: str, testlist: str) -> TestRun:
        test_run = self.call_support.create_test_run(project_id, testlist)
        test_cases = [
            TestCase(test_case_id=test_case.id, variables=test_case.variables)
            for test_case in test_run.test_cases
        ]

        return TestRun(test_run.test_run_id, test_cases)
