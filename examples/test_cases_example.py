import os

from freeplay import Freeplay
from freeplay.resources.test_cases import DatasetTestCase, DatasetResults

# logging.basicConfig(level=logging.NOTSET)

fpclient = Freeplay(
    freeplay_api_key=os.environ['FREEPLAY_API_KEY'],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api"
)

# Get Input Keys from your dataset. This example uses 'question' as the input.

# Create 1 Test  Case
test_case_1: DatasetTestCase = DatasetTestCase(history=None,
                                             metadata={"key": "value"},
                                             inputs={"question": "value 1"},
                                             output="Prompt response 1")

fpclient.test_cases.create(project_id=os.environ['FREEPLAY_PROJECT_ID'],
                           dataset_id=os.environ['FREEPLAY_DATASET_ID'],
                           test_case=test_case_1)

# Create Multiple Test Cases
test_case_2: DatasetTestCase = DatasetTestCase(history=None,
                                               metadata={"key 2": "value 2"},
                                               inputs={"question": "value 2"},
                                               output="Prompt response 2")

test_case_3: DatasetTestCase = DatasetTestCase(history=None,
                                               metadata={"key_3": "value_3"},
                                               inputs={"question": "value 3"},
                                               output="Prompt response 3")

fpclient.test_cases.create_many(project_id=os.environ['FREEPLAY_PROJECT_ID'],
                           dataset_id=os.environ['FREEPLAY_DATASET_ID'], test_cases=[test_case_2, test_case_3])

# Get
dataset_results: DatasetResults = fpclient.test_cases.get(
    project_id=os.environ['FREEPLAY_PROJECT_ID'],
    dataset_id=os.environ['FREEPLAY_DATASET_ID']
)

for test_case in dataset_results.test_cases:
    print(f'Test Case Id: {test_case.id}')
    print(f'\tHistory: {test_case.history}')
    print(f'\tInputs: {test_case.inputs}')
    print(f'\tOutput: {test_case.output}')
    print(f'\tMetadata: {test_case.metadata}')