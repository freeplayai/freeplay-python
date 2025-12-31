"""
Example demonstrating how to iterate through a dataset and print output_message fields.
"""

import os
import json

from freeplay import Freeplay
from freeplay.utils import convert_sdk_messages_to_api_messages


fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)

project_id = os.environ["FREEPLAY_PROJECT_ID"]
dataset_id = os.environ["FREEPLAY_DATASET_ID"]

# Maximum length for output_message display
MAX_OUTPUT_MESSAGE_LENGTH = 500

# Retrieve the dataset
print(f"Retrieving test cases from dataset {dataset_id}...")
dataset_results = fp_client.test_cases.get(project_id, dataset_id)

print(
    f"\nRetrieved {len(dataset_results.test_cases)} test cases from dataset {dataset_id}\n"
)

# Iterate through test cases and print output_message
for i, test_case in enumerate(dataset_results.test_cases, start=1):
    print(f"{'=' * 60}")
    print(f"Test Case {i} (ID: {test_case.id})")
    print(f"{'=' * 60}")

    if test_case.output_message:
        # Serialize output_message (converts dataclass objects to dicts)
        serialized_message = convert_sdk_messages_to_api_messages(
            test_case.output_message
        )

        # Convert to a string representation
        output_message_str = json.dumps(serialized_message, indent=2)

        # Truncate if too long
        if len(output_message_str) > MAX_OUTPUT_MESSAGE_LENGTH:
            output_message_str = output_message_str[:MAX_OUTPUT_MESSAGE_LENGTH] + "..."

        print(f"output_message:\n{output_message_str}\n")
    else:
        print("output_message: None\n")
