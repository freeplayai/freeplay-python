from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from freeplay.model import InputVariables, NormalizedMessage
from freeplay.support import CallSupport, DatasetTestCaseRequest, DatasetTestCasesRetrievalResponse


@dataclass
class DatasetTestCase:
    def __init__(
            self,
            inputs: InputVariables,
            output: Optional[str],
            history: Optional[List[NormalizedMessage]] = None,
            metadata: Optional[Dict[str, str]] = None,
            id: Optional[str] = None, # Only set on retrieval
    ):
        self.inputs = inputs
        self.output = output
        self.history = history
        self.metadata = metadata
        self.id = id



@dataclass
class Dataset:
    def __init__(self, dataset_id: str, test_cases: List[DatasetTestCase]):
        self.dataset_id = dataset_id
        self.test_cases = test_cases


@dataclass
class DatasetResults:
    def __init__(self, dataset_id: str, test_cases: List[DatasetTestCase]) -> None:
        self.dataset_id = dataset_id
        self.test_cases = test_cases

class TestCases:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def create(self, project_id: str, dataset_id: str, test_case: DatasetTestCase) -> Dataset:
        return self.create_many(project_id, dataset_id, [test_case])

    def create_many(self, project_id: str, dataset_id: str, test_cases: List[DatasetTestCase]) -> Dataset:
        dataset_test_cases = [DatasetTestCaseRequest(test_case.history, test_case.inputs, test_case.metadata, test_case.output) for test_case in test_cases]
        self.call_support.create_test_cases(project_id, dataset_id, dataset_test_cases)
        return Dataset(dataset_id, test_cases)

    def get(self, project_id: str, dataset_id: str) -> DatasetResults:
        test_case_results: DatasetTestCasesRetrievalResponse = self.call_support.get_test_cases(project_id, dataset_id)
        dataset_test_cases = test_case_results.test_cases

        return DatasetResults(dataset_id, [DatasetTestCase(id=test_case.id, history=test_case.history, output=test_case.output, inputs=test_case.values, metadata=test_case.metadata) for test_case in dataset_test_cases])
