"""
Test Suite — Agent/Trace-Type Example
======================================

Runs an agent-type test suite: iterates over trace test cases, runs your
agent logic, records trace results, and prints the pass/fail verdict.

Prerequisites
-------------
Set environment variables:

    export FREEPLAY_API_KEY="..."
    export FREEPLAY_API_URL="http://localhost:3000"   # or your Freeplay instance
    export FREEPLAY_PROJECT_ID="..."

Create an agent-type test suite via the API (one-time setup):

    # 1. Find your project's agent ID and dataset ID:
    #
    #    curl -s -H "Authorization: Bearer $FREEPLAY_API_KEY" \
    #      "$FREEPLAY_API_URL/api/v2/projects/$FREEPLAY_PROJECT_ID/datasets" | jq '.data[0].id'

    # 2. Create the test suite (agent_id instead of prompt_template_id):
    curl -X POST "$FREEPLAY_API_URL/api/v2/projects/$FREEPLAY_PROJECT_ID/test-suites" \
      -H "Authorization: Bearer $FREEPLAY_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "My Agent Suite",
        "description": "Example agent-type suite",
        "test_list_id": "<DATASET_ID>",
        "agent_id": "<AGENT_ID>"
      }'

    # The response includes the suite ID — set it below:
    export FREEPLAY_SUITE_ID="..."
"""

import os
import time

from customer_utils import get_freeplay_thin_client
from freeplay.resources.test_suites import TestSuites

fp_client = get_freeplay_thin_client()

project_id = os.environ["FREEPLAY_PROJECT_ID"]
suite_id = os.environ["FREEPLAY_SUITE_ID"]

# TestSuites is not yet on fp_client — construct directly for now.
# Once released, this becomes: fp_client.test_suites.run(...)
test_suites = TestSuites(fp_client.call_support, fp_client.recordings)

run = test_suites.run(
    project_id, suite_id, name="Trace Example Run"
)  # no environment needed for agent suites
print(f"Started run {run.run_id} — {run.total_test_cases} trace test cases")


def my_agent(input_text: str) -> str:
    """Placeholder for your agent logic."""
    return f"Agent response to: {input_text}"


for test_case in run.trace_test_cases:
    session = fp_client.sessions.create()
    trace = session.create_trace(
        input=test_case.input,
        agent_name="my-agent",
    )

    output = my_agent(test_case.input)
    print(f"  [{test_case.id}] input={test_case.input!r}  output={output!r}")

    run.record_trace(test_case, trace, output)

print("\nWaiting for evaluations to complete...")
time.sleep(10)

results = run.get_results()
print(f"\nStatus:  {results.status}")
print(f"Passed:  {results.passed}")
if results.summary_statistics:
    print(f"Auto:    {results.summary_statistics.auto_evaluation}")
