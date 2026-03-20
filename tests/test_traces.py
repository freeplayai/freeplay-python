import json
from typing import Any, Dict, Optional
from uuid import uuid4

import responses

from freeplay.errors import FreeplayClientError, FreeplayServerError
from freeplay.model import TestRunInfo
from freeplay.resources.traces import TraceUpdatePayload
from tests.test_base import FreeplayTestBase


class TestTraces(FreeplayTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.session_id = str(uuid4())
        self.trace_id = str(uuid4())

    # ========== Success Tests ==========

    @responses.activate
    def test_update_trace_with_output(self) -> None:
        self._mock_trace_update_endpoint(status=200)

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                output={"result": "updated output"},
            )
        )

        self._assert_patch_method_used()
        body = self._get_request_body()
        self.assertEqual(body["output"], {"result": "updated output"})
        self.assertNotIn("eval_results", body)

    @responses.activate
    def test_update_trace_with_eval_results(self) -> None:
        self._mock_trace_update_endpoint(status=200)

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                eval_results={"accuracy": 0.95, "valid": True},
            )
        )

        self._assert_patch_method_used()
        body = self._get_request_body()
        self.assertEqual(body["eval_results"], {"accuracy": 0.95, "valid": True})
        self.assertNotIn("output", body)

    @responses.activate
    def test_update_trace_with_output_and_eval_results(self) -> None:
        self._mock_trace_update_endpoint(status=200)

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                output="new output text",
                eval_results={"score": 0.8},
            )
        )

        self._assert_patch_method_used()
        body = self._get_request_body()
        self.assertEqual(body["output"], "new output text")
        self.assertEqual(body["eval_results"], {"score": 0.8})

    @responses.activate
    def test_update_trace_with_metadata(self) -> None:
        self._mock_trace_update_endpoint(status=200)

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                metadata={"env": "production", "version": 2},
            )
        )

        self._assert_patch_method_used()
        body = self._get_request_body()
        self.assertEqual(body["metadata"], {"env": "production", "version": 2})
        self.assertNotIn("output", body)
        self.assertNotIn("eval_results", body)

    @responses.activate
    def test_update_trace_with_feedback(self) -> None:
        self._mock_trace_update_endpoint(status=200)

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                feedback={"freeplay_feedback": "positive", "custom_score": 0.9},
            )
        )

        self._assert_patch_method_used()
        body = self._get_request_body()
        self.assertEqual(
            body["feedback"],
            {"freeplay_feedback": "positive", "custom_score": 0.9},
        )
        self.assertNotIn("output", body)

    @responses.activate
    def test_update_trace_with_test_run_info(self) -> None:
        self._mock_trace_update_endpoint(status=200)
        run_id = str(uuid4())
        case_id = str(uuid4())

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                test_run_info=TestRunInfo(test_run_id=run_id, test_case_id=case_id),
            )
        )

        self._assert_patch_method_used()
        body = self._get_request_body()
        self.assertEqual(
            body["test_run_info"],
            {"test_run_id": run_id, "test_case_id": case_id},
        )
        self.assertNotIn("output", body)

    @responses.activate
    def test_update_trace_with_all_fields(self) -> None:
        self._mock_trace_update_endpoint(status=200)
        run_id = str(uuid4())
        case_id = str(uuid4())

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                output="updated",
                metadata={"key": "value"},
                feedback={"freeplay_feedback": "negative"},
                eval_results={"accuracy": 0.99},
                test_run_info=TestRunInfo(test_run_id=run_id, test_case_id=case_id),
            )
        )

        self._assert_patch_method_used()
        body = self._get_request_body()
        self.assertEqual(body["output"], "updated")
        self.assertEqual(body["metadata"], {"key": "value"})
        self.assertEqual(body["feedback"], {"freeplay_feedback": "negative"})
        self.assertEqual(body["eval_results"], {"accuracy": 0.99})
        self.assertEqual(
            body["test_run_info"],
            {"test_run_id": run_id, "test_case_id": case_id},
        )

    # ========== URL Construction ==========

    @responses.activate
    def test_update_trace_url_construction(self) -> None:
        self._mock_trace_update_endpoint(status=200)

        self.client.traces.update(
            TraceUpdatePayload(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                output="test",
            )
        )

        self.assertEqual(len(responses.calls), 1)
        request_url = str(responses.calls[0].request.url)
        self.assertIn(f"/projects/{self.project_id}/", request_url)
        self.assertIn(f"/sessions/{self.session_id}/", request_url)
        self.assertIn(f"/traces/id/{self.trace_id}", request_url)
        self.assertNotIn("/metadata", request_url)

    # ========== Error Handling ==========

    def test_update_trace_requires_at_least_one_field(self) -> None:
        with self.assertRaisesRegex(
            FreeplayClientError,
            r"At least one of 'output', 'metadata', 'feedback', 'eval_results', or 'test_run_info'",
        ):
            self.client.traces.update(
                TraceUpdatePayload(
                    project_id=self.project_id,
                    session_id=self.session_id,
                    trace_id=self.trace_id,
                )
            )

    @responses.activate
    def test_update_trace_not_found(self) -> None:
        self._mock_trace_update_endpoint(
            status=404,
            response_body={"code": "trace_not_found", "message": "Trace not found"},
        )

        with self.assertRaisesRegex(
            FreeplayClientError, r"Error updating trace.*\[404\]"
        ):
            self.client.traces.update(
                TraceUpdatePayload(
                    project_id=self.project_id,
                    session_id=self.session_id,
                    trace_id=self.trace_id,
                    output="test",
                )
            )

    @responses.activate
    def test_update_trace_server_error(self) -> None:
        self._mock_trace_update_endpoint(
            status=500, response_body={"error": "Internal server error"}
        )

        with self.assertRaisesRegex(
            FreeplayServerError, r"Error updating trace.*\[500\]"
        ):
            self.client.traces.update(
                TraceUpdatePayload(
                    project_id=self.project_id,
                    session_id=self.session_id,
                    trace_id=self.trace_id,
                    eval_results={"score": 1.0},
                )
            )

    # ========== Helper Methods ==========

    def _mock_trace_update_endpoint(
        self,
        status: int = 200,
        response_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        if response_body is None:
            response_body = {"message": "Trace updated successfully"}

        responses.patch(
            url=f"{self.api_base}/v2/projects/{self.project_id}/sessions/{self.session_id}/traces/id/{self.trace_id}",
            status=status,
            content_type="application/json",
            json=response_body,
        )

    def _get_request_body(self, call_index: int = 0) -> Dict[str, Any]:
        self.assertGreater(len(responses.calls), call_index)
        raw = responses.calls[call_index].request.body
        self.assertIsNotNone(raw)
        return json.loads(raw)  # type: ignore

    def _assert_patch_method_used(self, call_index: int = 0) -> None:
        self.assertGreater(len(responses.calls), call_index)
        self.assertEqual(responses.calls[call_index].request.method, "PATCH")
