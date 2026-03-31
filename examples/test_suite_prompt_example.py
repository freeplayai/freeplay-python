"""
Test Suite — Prompt-Type Example (OpenAI)
=========================================

Runs a prompt-type test suite: iterates over test cases, calls OpenAI,
records results, and prints the pass/fail verdict.

Prerequisites
-------------
Set environment variables:

    export FREEPLAY_API_KEY="..."
    export FREEPLAY_API_URL="http://localhost:3000"   # or your Freeplay instance
    export FREEPLAY_PROJECT_ID="..."
    export OPENAI_API_KEY="..."

Create a test suite via the API (one-time setup):

    # 1. Find your project's prompt template ID and dataset ID.
    #    You can grab these from the Freeplay UI or via the API:
    #
    #    curl -s -H "Authorization: Bearer $FREEPLAY_API_KEY" \
    #      "$FREEPLAY_API_URL/api/v2/projects/$FREEPLAY_PROJECT_ID/prompt-templates" | jq '.data[0].id'
    #
    #    curl -s -H "Authorization: Bearer $FREEPLAY_API_KEY" \
    #      "$FREEPLAY_API_URL/api/v2/projects/$FREEPLAY_PROJECT_ID/datasets" | jq '.data[0].id'

    # 2. Create the test suite:
    curl -X POST "$FREEPLAY_API_URL/api/v2/projects/$FREEPLAY_PROJECT_ID/test-suites" \
      -H "Authorization: Bearer $FREEPLAY_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "My Prompt Suite",
        "description": "Example prompt-type suite",
        "test_list_id": "<DATASET_ID>",
        "prompt_template_id": "<PROMPT_TEMPLATE_ID>"
      }'

    # The response includes the suite ID — set it below:
    export FREEPLAY_SUITE_ID="..."
"""

import os
import time

from openai import OpenAI

from customer_utils import get_freeplay_thin_client
from freeplay import CallInfo
from freeplay.resources.test_suites import TestSuites

fp_client = get_freeplay_thin_client()
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

project_id = os.environ["FREEPLAY_PROJECT_ID"]
suite_id = os.environ["FREEPLAY_SUITE_ID"]
environment = os.environ.get("FREEPLAY_ENVIRONMENT", "latest")

# TestSuites is not yet on fp_client — construct directly for now.
# Once released, this becomes: fp_client.test_suites.run(...)
test_suites = TestSuites(fp_client.call_support, fp_client.recordings)

run = test_suites.run(project_id, suite_id, environment)
print(f"Started run {run.run_id} — {run.total_test_cases} test cases")

for test_case in run.test_cases:
    formatted = run.format_prompt(test_case)

    start = time.time()
    completion = openai_client.chat.completions.create(
        messages=formatted.llm_prompt,
        model=formatted.prompt_info.model,
        tools=formatted.tool_schema,
        **formatted.prompt_info.model_parameters,
    )
    end = time.time()

    print(f"  [{test_case.id}] {completion.choices[0].message.content[:80]}...")

    run.record(
        test_case,
        all_messages=formatted.llm_prompt
        + [{"role": "assistant", "content": completion.choices[0].message.content}],
        call_info=CallInfo.from_prompt_info(formatted.prompt_info, start, end),
    )

print("\nWaiting for evaluations to complete...")
time.sleep(10)

results = run.get_results()
print(f"\nStatus:  {results.status}")
print(f"Passed:  {results.passed}")
if results.summary_statistics:
    print(f"Auto:    {results.summary_statistics.auto_evaluation}")
    print(f"Human:   {results.summary_statistics.human_evaluation}")
