import json
from typing import Any, Dict, Optional
from uuid import uuid4

import responses

from freeplay.errors import FreeplayClientError, FreeplayServerError
from freeplay.model import TestRunInfo
from tests.test_base import FreeplayTestBase


class TestTraces(FreeplayTestBase):

    def setUp(self) -> None:
        super().setUp()
        self.session_id = str(uuid4())
        self.trace_id = str(uuid4())
        self.otel_span_id_hex = "abcdef0123456789"

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

    def _mock_otel_trace_update_endpoint(
        self,
        status: int = 200,
        response_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        if response_body is None:
            response_body = {"message": "Trace updated successfully"}
        responses.patch(
            url=f"{self.api_base}/v2/projects/{self.project_id}/sessions/{self.session_id}/traces/otel-span-id/{self.otel_span_id_hex}",
            status=status,
            content_type="application/json",
            json=response_body,
        )

    def _get_request_body(self, call_index: int = 0) -> Dict[str, Any]:
        request_body_raw = responses.calls[call_index].request.body
        assert request_body_raw is not None
        return json.loads(request_body_raw)

    # ========== Update by trace ID ==========

    @responses.activate
    def test_update_trace_with_eval_results(self) -> None:
        self._mock_trace_update_endpoint()
        eval_results = {"accuracy": 0.95, "is_correct": True}

        result = self.client.traces.update(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            eval_results=eval_results,
        )

        self.assertIsNotNone(result)
        body = self._get_request_body()
        self.assertEqual(body["eval_results"], eval_results)
        self.assertNotIn("custom_metadata", body)
        self.assertNotIn("feedback", body)
        self.assertNotIn("test_run_info", body)

    @responses.activate
    def test_update_trace_with_test_run_info(self) -> None:
        self._mock_trace_update_endpoint()
        test_run_info = TestRunInfo(test_run_id="run-123", test_case_id="case-456")

        result = self.client.traces.update(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            test_run_info=test_run_info,
        )

        self.assertIsNotNone(result)
        body = self._get_request_body()
        self.assertEqual(
            body["test_run_info"],
            {"test_run_id": "run-123", "test_case_id": "case-456"},
        )

    @responses.activate
    def test_update_trace_with_all_fields(self) -> None:
        self._mock_trace_update_endpoint()
        metadata = {"key": "value"}
        feedback = {"thumbs_up": True}
        eval_results = {"score": 0.8}
        test_run_info = TestRunInfo(test_run_id="run-1", test_case_id="case-1")

        result = self.client.traces.update(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            custom_metadata=metadata,
            feedback=feedback,
            eval_results=eval_results,
            test_run_info=test_run_info,
        )

        self.assertIsNotNone(result)
        body = self._get_request_body()
        self.assertEqual(body["custom_metadata"], metadata)
        self.assertEqual(body["feedback"], feedback)
        self.assertEqual(body["eval_results"], eval_results)
        self.assertEqual(
            body["test_run_info"],
            {"test_run_id": "run-1", "test_case_id": "case-1"},
        )

    @responses.activate
    def test_update_trace_only_sends_provided_fields(self) -> None:
        self._mock_trace_update_endpoint()

        self.client.traces.update(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            feedback={"rating": 5},
        )

        body = self._get_request_body()
        self.assertEqual(body, {"feedback": {"rating": 5}})

    @responses.activate
    def test_update_trace_uses_patch_method(self) -> None:
        self._mock_trace_update_endpoint()

        self.client.traces.update(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            eval_results={"score": 1.0},
        )

        self.assertEqual(responses.calls[0].request.method, "PATCH")

    @responses.activate
    def test_update_trace_url_construction(self) -> None:
        self._mock_trace_update_endpoint()

        self.client.traces.update(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            eval_results={"score": 1.0},
        )

        request_url = str(responses.calls[0].request.url)
        self.assertIn(f"/projects/{self.project_id}/", request_url)
        self.assertIn(f"/sessions/{self.session_id}/", request_url)
        self.assertIn(f"/traces/id/{self.trace_id}", request_url)

    @responses.activate
    def test_update_trace_not_found(self) -> None:
        self._mock_trace_update_endpoint(
            status=404,
            response_body={"code": "entity_not_found", "message": "Trace not found"},
        )

        with self.assertRaisesRegex(
            FreeplayClientError, r"Error updating trace.*\[404\]"
        ):
            self.client.traces.update(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                eval_results={"score": 1.0},
            )

    @responses.activate
    def test_update_trace_server_error(self) -> None:
        self._mock_trace_update_endpoint(
            status=500,
            response_body={"error": "Internal server error"},
        )

        with self.assertRaisesRegex(
            FreeplayServerError, r"Error updating trace.*\[500\]"
        ):
            self.client.traces.update(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                eval_results={"score": 1.0},
            )

    # ========== Update by otel span ID ==========

    @responses.activate
    def test_update_by_otel_span_id_with_eval_results(self) -> None:
        self._mock_otel_trace_update_endpoint()
        eval_results = {"accuracy": 0.95, "is_correct": True}

        result = self.client.traces.update_by_otel_span_id(
            project_id=self.project_id,
            session_id=self.session_id,
            otel_span_id_hex=self.otel_span_id_hex,
            eval_results=eval_results,
        )

        self.assertIsNotNone(result)
        body = self._get_request_body()
        self.assertEqual(body["eval_results"], eval_results)

    @responses.activate
    def test_update_by_otel_span_id_with_test_run_info(self) -> None:
        self._mock_otel_trace_update_endpoint()
        test_run_info = TestRunInfo(test_run_id="run-789", test_case_id="case-012")

        result = self.client.traces.update_by_otel_span_id(
            project_id=self.project_id,
            session_id=self.session_id,
            otel_span_id_hex=self.otel_span_id_hex,
            test_run_info=test_run_info,
        )

        self.assertIsNotNone(result)
        body = self._get_request_body()
        self.assertEqual(
            body["test_run_info"],
            {"test_run_id": "run-789", "test_case_id": "case-012"},
        )

    @responses.activate
    def test_update_by_otel_span_id_with_all_fields(self) -> None:
        self._mock_otel_trace_update_endpoint()
        metadata = {"env": "production"}
        feedback = {"helpful": True}
        eval_results = {"relevance": 0.9}
        test_run_info = TestRunInfo(test_run_id="run-x", test_case_id="case-y")

        result = self.client.traces.update_by_otel_span_id(
            project_id=self.project_id,
            session_id=self.session_id,
            otel_span_id_hex=self.otel_span_id_hex,
            custom_metadata=metadata,
            feedback=feedback,
            eval_results=eval_results,
            test_run_info=test_run_info,
        )

        self.assertIsNotNone(result)
        body = self._get_request_body()
        self.assertEqual(body["custom_metadata"], metadata)
        self.assertEqual(body["feedback"], feedback)
        self.assertEqual(body["eval_results"], eval_results)
        self.assertEqual(
            body["test_run_info"],
            {"test_run_id": "run-x", "test_case_id": "case-y"},
        )

    @responses.activate
    def test_update_by_otel_span_id_only_sends_provided_fields(self) -> None:
        self._mock_otel_trace_update_endpoint()

        self.client.traces.update_by_otel_span_id(
            project_id=self.project_id,
            session_id=self.session_id,
            otel_span_id_hex=self.otel_span_id_hex,
            custom_metadata={"key": "val"},
        )

        body = self._get_request_body()
        self.assertEqual(body, {"custom_metadata": {"key": "val"}})

    @responses.activate
    def test_update_by_otel_span_id_uses_patch_method(self) -> None:
        self._mock_otel_trace_update_endpoint()

        self.client.traces.update_by_otel_span_id(
            project_id=self.project_id,
            session_id=self.session_id,
            otel_span_id_hex=self.otel_span_id_hex,
            eval_results={"score": 1.0},
        )

        self.assertEqual(responses.calls[0].request.method, "PATCH")

    @responses.activate
    def test_update_by_otel_span_id_url_construction(self) -> None:
        self._mock_otel_trace_update_endpoint()

        self.client.traces.update_by_otel_span_id(
            project_id=self.project_id,
            session_id=self.session_id,
            otel_span_id_hex=self.otel_span_id_hex,
            eval_results={"score": 1.0},
        )

        request_url = str(responses.calls[0].request.url)
        self.assertIn(f"/projects/{self.project_id}/", request_url)
        self.assertIn(f"/sessions/{self.session_id}/", request_url)
        self.assertIn(f"/traces/otel-span-id/{self.otel_span_id_hex}", request_url)

    @responses.activate
    def test_update_by_otel_span_id_not_found(self) -> None:
        self._mock_otel_trace_update_endpoint(
            status=404,
            response_body={"code": "entity_not_found", "message": "Trace not found"},
        )

        with self.assertRaisesRegex(
            FreeplayClientError, r"Error updating trace.*\[404\]"
        ):
            self.client.traces.update_by_otel_span_id(
                project_id=self.project_id,
                session_id=self.session_id,
                otel_span_id_hex=self.otel_span_id_hex,
                eval_results={"score": 1.0},
            )

    @responses.activate
    def test_update_by_otel_span_id_server_error(self) -> None:
        self._mock_otel_trace_update_endpoint(
            status=500,
            response_body={"error": "Internal server error"},
        )

        with self.assertRaisesRegex(
            FreeplayServerError, r"Error updating trace.*\[500\]"
        ):
            self.client.traces.update_by_otel_span_id(
                project_id=self.project_id,
                session_id=self.session_id,
                otel_span_id_hex=self.otel_span_id_hex,
                eval_results={"score": 1.0},
            )

    @responses.activate
    def test_update_trace_empty_payload(self) -> None:
        self._mock_trace_update_endpoint()

        result = self.client.traces.update(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
        )

        self.assertIsNotNone(result)
        body = self._get_request_body()
        self.assertEqual(body, {})
