from freeplay.support import CallSupport
from freeplay.thin.model import TestRun, TestCase


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
