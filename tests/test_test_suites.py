import json
from typing import Any, Dict, Optional
from unittest import TestCase
from uuid import uuid4

import responses

from freeplay.resources.recordings import Recordings
from freeplay.resources.sessions import SessionInfo
from freeplay.resources.test_runs import CompletionTestCase, TraceTestCase
from freeplay.resources.test_suites import (
    TestSuiteRun,
    TestSuiteRunResults,
    TestSuites,
)
from freeplay.support import CallSupport


class TestTestSuites(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.api_base = "http://localhost:9091/api"
        self.api_key = "test-api-key"
        self.project_id = str(uuid4())
        self.suite_id = str(uuid4())
        self.run_id = str(uuid4())
        self.test_case_id_1 = str(uuid4())
        self.test_case_id_2 = str(uuid4())

        self.call_support = CallSupport(self.api_key, self.api_base)
        self.recordings = Recordings(self.call_support)
        self.test_suites = TestSuites(self.call_support, self.recordings)

        self.prompt_template_id = str(uuid4())
        self.prompt_template_version_id = str(uuid4())

    def _prompt_template_payload(self) -> Dict[str, Any]:
        return {
            "prompt_template_id": self.prompt_template_id,
            "prompt_template_version_id": self.prompt_template_version_id,
            "prompt_template_name": "test-prompt",
            "version_name": "v1",
            "version_description": "Test version",
            "metadata": {
                "provider": "openai",
                "flavor": "openai_chat",
                "model": "gpt-4",
                "params": {"temperature": 0.7},
                "provider_info": {},
            },
            "format_version": 3,
            "project_id": self.project_id,
            "content": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello {{name}}"},
            ],
            "tool_schema": None,
            "output_schema": None,
        }

    def _mock_create_run_prompt(self) -> None:
        responses.post(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs",
            json={
                "run_id": self.run_id,
                "suite_id": self.suite_id,
                "target_type": "prompt",
                "total_test_cases": 2,
                "prompt_template": self._prompt_template_payload(),
            },
            status=201,
        )

    def _mock_create_run_agent(self) -> None:
        responses.post(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs",
            json={
                "run_id": self.run_id,
                "suite_id": self.suite_id,
                "target_type": "agent",
                "total_test_cases": 2,
                "prompt_template": None,
            },
            status=201,
        )

    def _mock_test_cases_completion(self, has_next: bool = False) -> None:
        responses.get(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs/{self.run_id}/test-cases",
            json={
                "data": [
                    {
                        "test_case_id": self.test_case_id_1,
                        "variables": {"name": "Alice"},
                        "output": None,
                        "history": None,
                        "custom_metadata": None,
                        "test_case_type": "completion",
                    },
                    {
                        "test_case_id": self.test_case_id_2,
                        "variables": {"name": "Bob"},
                        "output": "expected output",
                        "history": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ],
                        "custom_metadata": {"key": "value"},
                        "test_case_type": "completion",
                    },
                ],
                "pagination": {"page": 1, "page_size": 50, "has_next": has_next},
            },
            status=200,
        )

    def _mock_test_cases_trace(self) -> None:
        responses.get(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs/{self.run_id}/test-cases",
            json={
                "data": [
                    {
                        "test_case_id": self.test_case_id_1,
                        "input": "What is 2+2?",
                        "output": "4",
                        "custom_metadata": {"difficulty": "easy"},
                        "test_case_type": "trace",
                    },
                ],
                "pagination": {"page": 1, "page_size": 50, "has_next": False},
            },
            status=200,
        )

    def _mock_results(
        self,
        status: str = "complete",
        passed: Optional[bool] = True,
    ) -> None:
        summary: Optional[Dict[str, Any]] = None
        eval_results: Optional[Dict[str, Any]] = None
        if passed is not None:
            eval_results = {"accuracy": 0.95}
            summary = {
                "human_evaluation": {"score": 0.8},
                "auto_evaluation": {"accuracy": 0.95},
                "client_evaluation": None,
            }

        responses.get(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs/{self.run_id}/results",
            json={
                "run_id": self.run_id,
                "suite_id": self.suite_id,
                "status": status,
                "passed": passed,
                "eval_results": eval_results,
                "summary_statistics": summary,
            },
            status=200,
        )

    # --- Run creation ---

    @responses.activate
    def test_run_prompt_suite(self) -> None:
        self._mock_create_run_prompt()
        self._mock_test_cases_completion()

        run = self.test_suites.run(self.project_id, self.suite_id)

        self.assertEqual(self.run_id, run.run_id)
        self.assertEqual(self.suite_id, run.suite_id)
        self.assertEqual("prompt", run.target_type)
        self.assertEqual(2, run.total_test_cases)
        self.assertIsNotNone(run._template_prompt)

        test_cases = list(run.test_cases)
        self.assertEqual(2, len(test_cases))
        self.assertEqual(self.test_case_id_1, test_cases[0].id)
        self.assertEqual({"name": "Alice"}, test_cases[0].variables)
        self.assertIsNone(test_cases[0].output)
        self.assertEqual(self.test_case_id_2, test_cases[1].id)
        self.assertEqual({"name": "Bob"}, test_cases[1].variables)
        self.assertEqual("expected output", test_cases[1].output)
        self.assertIsNotNone(test_cases[1].history)
        self.assertEqual(2, len(test_cases[1].history))
        self.assertEqual({"key": "value"}, test_cases[1].custom_metadata)

    @responses.activate
    def test_run_prompt_suite_paginated(self) -> None:
        self._mock_create_run_prompt()

        responses.get(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs/{self.run_id}/test-cases",
            json={
                "data": [
                    {
                        "test_case_id": self.test_case_id_1,
                        "variables": {"name": "Alice"},
                        "output": None,
                        "test_case_type": "completion",
                    }
                ],
                "pagination": {"page": 1, "page_size": 1, "has_next": True},
            },
            status=200,
        )
        responses.get(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs/{self.run_id}/test-cases",
            json={
                "data": [
                    {
                        "test_case_id": self.test_case_id_2,
                        "variables": {"name": "Bob"},
                        "output": None,
                        "test_case_type": "completion",
                    }
                ],
                "pagination": {"page": 2, "page_size": 1, "has_next": False},
            },
            status=200,
        )

        run = self.test_suites.run(self.project_id, self.suite_id, page_size=1)
        test_cases = list(run.test_cases)

        self.assertEqual(2, len(test_cases))
        self.assertEqual(self.test_case_id_1, test_cases[0].id)
        self.assertEqual(self.test_case_id_2, test_cases[1].id)

    @responses.activate
    def test_run_agent_suite(self) -> None:
        self._mock_create_run_agent()
        self._mock_test_cases_trace()

        run = self.test_suites.run(self.project_id, self.suite_id)

        self.assertEqual("agent", run.target_type)
        self.assertIsNone(run._template_prompt)

        trace_cases = list(run.trace_test_cases)
        self.assertEqual(1, len(trace_cases))
        self.assertEqual(self.test_case_id_1, trace_cases[0].id)
        self.assertEqual("What is 2+2?", trace_cases[0].input)
        self.assertEqual("4", trace_cases[0].output)
        self.assertEqual({"difficulty": "easy"}, trace_cases[0].custom_metadata)

    # --- Type guards ---

    @responses.activate
    def test_test_cases_raises_for_agent_suite(self) -> None:
        self._mock_create_run_agent()
        run = self.test_suites.run(self.project_id, self.suite_id)
        with self.assertRaises(ValueError):
            list(run.test_cases)

    @responses.activate
    def test_trace_test_cases_raises_for_prompt_suite(self) -> None:
        self._mock_create_run_prompt()
        run = self.test_suites.run(self.project_id, self.suite_id)
        with self.assertRaises(ValueError):
            list(run.trace_test_cases)

    # --- format_prompt ---

    @responses.activate
    def test_format_prompt(self) -> None:
        self._mock_create_run_prompt()
        run = self.test_suites.run(self.project_id, self.suite_id)

        tc = CompletionTestCase(
            test_case_id="tc-1",
            variables={"name": "World"},
            output=None,
            history=None,
            custom_metadata=None,
        )
        formatted = run.format_prompt(tc)

        self.assertIsNotNone(formatted)
        self.assertIsNotNone(formatted.prompt_info)
        self.assertEqual("gpt-4", formatted.prompt_info.model)
        self.assertEqual("openai_chat", formatted.prompt_info.flavor_name)
        self.assertEqual(
            self.prompt_template_version_id,
            formatted.prompt_info.prompt_template_version_id,
        )

    @responses.activate
    def test_format_prompt_raises_for_agent_suite(self) -> None:
        self._mock_create_run_agent()
        run = self.test_suites.run(self.project_id, self.suite_id)

        tc = CompletionTestCase(
            test_case_id="tc-1",
            variables={"name": "World"},
            output=None,
            history=None,
            custom_metadata=None,
        )
        with self.assertRaises(ValueError):
            run.format_prompt(tc)

    @responses.activate
    def test_format_prompt_with_history(self) -> None:
        pt = self._prompt_template_payload()
        pt["content"] = [
            {"role": "system", "content": "You are helpful."},
            {"kind": "history"},
            {"role": "user", "content": "Now help with {{name}}"},
        ]
        responses.post(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs",
            json={
                "run_id": self.run_id,
                "suite_id": self.suite_id,
                "target_type": "prompt",
                "total_test_cases": 1,
                "prompt_template": pt,
            },
            status=201,
        )

        run = self.test_suites.run(self.project_id, self.suite_id)
        tc = CompletionTestCase(
            test_case_id="tc-1",
            variables={"name": "World"},
            output=None,
            history=[{"role": "user", "content": "Previous"}],
            custom_metadata=None,
        )
        formatted = run.format_prompt(tc)
        self.assertIsNotNone(formatted)

    # --- record ---

    @responses.activate
    def test_record_completion(self) -> None:
        self._mock_create_run_prompt()

        session_id = str(uuid4())
        completion_id = str(uuid4())

        responses.post(
            url=f"{self.api_base}/v2/projects/{self.project_id}/sessions/{session_id}/completions",
            json={"completion_id": completion_id},
            status=201,
        )

        run = self.test_suites.run(self.project_id, self.suite_id)
        tc = CompletionTestCase(
            test_case_id=self.test_case_id_1,
            variables={"name": "Alice"},
            output=None,
            history=None,
            custom_metadata=None,
        )

        result = run.record(
            tc,
            all_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello Alice"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            session_info=SessionInfo(session_id=session_id, custom_metadata=None),
        )

        self.assertEqual(completion_id, result.completion_id)

        request_body = json.loads(responses.calls[-1].request.body)
        self.assertEqual(self.run_id, request_body["test_run_info"]["test_run_id"])
        self.assertEqual(
            self.test_case_id_1, request_body["test_run_info"]["test_case_id"]
        )
        self.assertIsNotNone(request_body.get("prompt_info"))

    # --- get_results ---

    @responses.activate
    def test_get_results_passed(self) -> None:
        self._mock_create_run_prompt()
        self._mock_results(status="complete", passed=True)

        run = self.test_suites.run(self.project_id, self.suite_id)
        results = run.get_results()

        self.assertEqual(self.run_id, results.run_id)
        self.assertEqual(self.suite_id, results.suite_id)
        self.assertEqual("complete", results.status)
        self.assertTrue(results.passed)
        self.assertIsNotNone(results.eval_results)
        self.assertEqual(0.95, results.eval_results["accuracy"])
        self.assertIsNotNone(results.summary_statistics)
        self.assertIsNotNone(results.summary_statistics.auto_evaluation)
        self.assertIsNotNone(results.summary_statistics.human_evaluation)
        self.assertIsNone(results.summary_statistics.client_evaluation)

    @responses.activate
    def test_get_results_in_progress(self) -> None:
        self._mock_create_run_prompt()
        self._mock_results(status="in-progress", passed=None)

        run = self.test_suites.run(self.project_id, self.suite_id)
        results = run.get_results()

        self.assertEqual("in-progress", results.status)
        self.assertIsNone(results.passed)
        self.assertIsNone(results.eval_results)
        self.assertIsNone(results.summary_statistics)

    @responses.activate
    def test_get_results_failed(self) -> None:
        self._mock_create_run_prompt()
        self._mock_results(status="complete", passed=False)

        run = self.test_suites.run(self.project_id, self.suite_id)
        results = run.get_results()

        self.assertEqual("complete", results.status)
        self.assertFalse(results.passed)

    # --- get_test_run_info ---

    @responses.activate
    def test_get_test_run_info(self) -> None:
        self._mock_create_run_prompt()
        run = self.test_suites.run(self.project_id, self.suite_id)

        info = run.get_test_run_info("some-test-case-id")
        self.assertEqual(self.run_id, info.test_run_id)
        self.assertEqual("some-test-case-id", info.test_case_id)

    # --- media_variables ---

    @responses.activate
    def test_completion_test_case_with_media_variables(self) -> None:
        self._mock_create_run_prompt()
        responses.get(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs/{self.run_id}/test-cases",
            json={
                "data": [
                    {
                        "test_case_id": self.test_case_id_1,
                        "variables": {"name": "Alice"},
                        "output": None,
                        "test_case_type": "completion",
                        "media_variables": {
                            "avatar": {
                                "type": "url",
                                "url": "https://example.com/avatar.png",
                            },
                            "document": {
                                "type": "base64",
                                "data": "aGVsbG8=",
                                "content_type": "application/pdf",
                            },
                        },
                    },
                ],
                "pagination": {"page": 1, "page_size": 50, "has_next": False},
            },
            status=200,
        )

        run = self.test_suites.run(self.project_id, self.suite_id)
        test_cases = list(run.test_cases)

        self.assertEqual(1, len(test_cases))
        tc = test_cases[0]
        self.assertIsNotNone(tc.media_variables)
        self.assertIn("avatar", tc.media_variables)
        self.assertEqual("url", tc.media_variables["avatar"].type)
        self.assertEqual(
            "https://example.com/avatar.png", tc.media_variables["avatar"].url
        )
        self.assertIn("document", tc.media_variables)
        self.assertEqual("base64", tc.media_variables["document"].type)
        self.assertEqual("aGVsbG8=", tc.media_variables["document"].data)
        self.assertEqual(
            "application/pdf", tc.media_variables["document"].content_type
        )

    # --- prompt template with tool schema ---

    @responses.activate
    def test_run_with_tool_schema(self) -> None:
        pt = self._prompt_template_payload()
        pt["tool_schema"] = [
            {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        responses.post(
            url=f"{self.api_base}/v2/projects/{self.project_id}/test-suites/{self.suite_id}/runs",
            json={
                "run_id": self.run_id,
                "suite_id": self.suite_id,
                "target_type": "prompt",
                "total_test_cases": 1,
                "prompt_template": pt,
            },
            status=201,
        )

        run = self.test_suites.run(self.project_id, self.suite_id)
        self.assertIsNotNone(run._template_prompt)
        self.assertIsNotNone(run._template_prompt.tool_schema)
        self.assertEqual(1, len(run._template_prompt.tool_schema))
        self.assertEqual("get_weather", run._template_prompt.tool_schema[0].name)
