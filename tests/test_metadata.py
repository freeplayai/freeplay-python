"""
Tests for Metadata Update Functionality

This module tests the metadata update feature for sessions and traces.

Why this was created:
- Validates that the new Metadata resource class works correctly
- Ensures metadata updates are sent to the correct API endpoints with proper formatting
- Tests error handling for various failure scenarios (404, 500, etc.)
- Verifies HTTP methods (PATCH) and URL construction

What it tests:
- Session metadata updates (success, errors, URL construction, HTTP methods)
- Trace metadata updates (success, errors, RESTful URL hierarchy)
- Edge cases (empty metadata, various data types)
- Request body formatting and validation

Test coverage: 12 tests covering success paths, error handling, and edge cases.
"""

import json
from typing import Any, Dict, Optional
from uuid import uuid4

import responses

from freeplay.errors import FreeplayClientError, FreeplayServerError
from freeplay.support import CustomMetadata
from tests.test_base import FreeplayTestBase


class TestMetadata(FreeplayTestBase):
    """Test cases for metadata update operations."""

    def setUp(self) -> None:
        """Set up metadata-specific test fixtures."""
        super().setUp()

        # Metadata-specific fixtures
        self.session_id = str(uuid4())
        self.trace_id = str(uuid4())

        # Common test metadata
        self.test_metadata = {
            "customer_id": "cust_123",
            "rating": 5,
            "premium": True,
            "score": 4.5,
        }

    # ========== Session Metadata Tests (6 tests) ==========

    @responses.activate
    def test_update_session_metadata_success(self) -> None:
        """Test successful session metadata update."""
        # Arrange
        self._mock_session_metadata_endpoint(status=200)

        # Act
        result = self.client.metadata.update_session(
            project_id=self.project_id,
            session_id=self.session_id,
            metadata=self.test_metadata,
        )

        # Assert
        self.assertIsNotNone(result)
        self._assert_request_made_with_metadata(self.test_metadata)
        self._assert_patch_method_used()

    @responses.activate
    def test_update_session_metadata_not_found(self) -> None:
        """Test session metadata update with 404 error."""
        # Arrange
        self._mock_session_metadata_endpoint(
            status=404,
            response_body={"code": "entity_not_found", "message": "Session not found"},
        )

        # Act & Assert
        with self.assertRaisesRegex(
            FreeplayClientError, r"Error updating session metadata.*\[404\]"
        ):
            self.client.metadata.update_session(
                project_id=self.project_id,
                session_id=self.session_id,
                metadata={"key": "value"},
            )

    @responses.activate
    def test_update_session_metadata_server_error(self) -> None:
        """Test session metadata update with 500 error."""
        # Arrange
        self._mock_session_metadata_endpoint(
            status=500, response_body={"error": "Internal server error"}
        )

        # Act & Assert
        with self.assertRaisesRegex(
            FreeplayServerError, r"Error updating session metadata.*\[500\]"
        ):
            self.client.metadata.update_session(
                project_id=self.project_id,
                session_id=self.session_id,
                metadata={"key": "value"},
            )

    @responses.activate
    def test_update_session_metadata_request_body(self) -> None:
        """Test that request body contains correct metadata."""
        # Arrange
        test_metadata = {
            "string_key": "string_value",
            "int_key": 42,
            "float_key": 3.14,
            "bool_key": True,
        }
        self._mock_session_metadata_endpoint(status=200)

        # Act
        self.client.metadata.update_session(
            project_id=self.project_id,
            session_id=self.session_id,
            metadata=test_metadata,
        )

        # Assert
        self._assert_request_made_with_metadata(test_metadata)

    @responses.activate
    def test_update_session_metadata_url_construction(self) -> None:
        """Test that correct URL is constructed with project and session IDs."""
        # Arrange
        self._mock_session_metadata_endpoint(status=200)

        # Act
        self.client.metadata.update_session(
            project_id=self.project_id,
            session_id=self.session_id,
            metadata={"key": "value"},
        )

        # Assert - verify URL structure
        self.assertEqual(len(responses.calls), 1)
        request_url = str(responses.calls[0].request.url)

        # Check project_id and session_id in URL
        self.assertIn(f"/projects/{self.project_id}/", request_url)
        self.assertIn(f"/sessions/id/{self.session_id}/metadata", request_url)

    @responses.activate
    def test_update_session_metadata_http_method(self) -> None:
        """Test that PATCH method is used (not POST or PUT)."""
        # Arrange
        self._mock_session_metadata_endpoint(status=200)

        # Act
        self.client.metadata.update_session(
            project_id=self.project_id,
            session_id=self.session_id,
            metadata={"key": "value"},
        )

        # Assert - verify HTTP method
        self._assert_patch_method_used()
        self.assertEqual(responses.calls[0].request.method, "PATCH")

    # ========== Trace Metadata Tests (4 tests) ==========

    @responses.activate
    def test_update_trace_metadata_success(self) -> None:
        """Test successful trace metadata update."""
        # Arrange
        self._mock_trace_metadata_endpoint(status=200)

        # Act
        result = self.client.metadata.update_trace(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            metadata=self.test_metadata,
        )

        # Assert
        self.assertIsNotNone(result)
        self._assert_request_made_with_metadata(self.test_metadata)
        self._assert_patch_method_used()

    @responses.activate
    def test_update_trace_metadata_not_found(self) -> None:
        """Test trace metadata update with 404 error."""
        # Arrange
        self._mock_trace_metadata_endpoint(
            status=404,
            response_body={"code": "entity_not_found", "message": "Trace not found"},
        )

        # Act & Assert
        with self.assertRaisesRegex(
            FreeplayClientError, r"Error updating trace metadata.*\[404\]"
        ):
            self.client.metadata.update_trace(
                project_id=self.project_id,
                session_id=self.session_id,
                trace_id=self.trace_id,
                metadata={"key": "value"},
            )

    @responses.activate
    def test_update_trace_metadata_url_with_session_id(self) -> None:
        """Test that trace URL includes session_id (RESTful hierarchy)."""
        # Arrange
        self._mock_trace_metadata_endpoint(status=200)

        # Act
        self.client.metadata.update_trace(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            metadata={"key": "value"},
        )

        # Assert - verify URL structure
        self.assertEqual(len(responses.calls), 1)
        request_url = str(responses.calls[0].request.url)

        # Check all IDs in URL
        self.assertIn(f"/projects/{self.project_id}/", request_url)
        self.assertIn(f"/sessions/{self.session_id}/", request_url)
        self.assertIn(f"/traces/id/{self.trace_id}/metadata", request_url)

    @responses.activate
    def test_update_trace_metadata_request_body(self) -> None:
        """Test that request body contains correct metadata."""
        # Arrange
        test_metadata: CustomMetadata = {
            "trace_key": "trace_value",
            "count": 10,
            "enabled": False,
        }
        self._mock_trace_metadata_endpoint(status=200)

        # Act
        self.client.metadata.update_trace(
            project_id=self.project_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            metadata=test_metadata,
        )

        # Assert
        self._assert_request_made_with_metadata(test_metadata)

    # ========== Edge Cases (2 tests) ==========

    @responses.activate
    def test_update_metadata_empty_dict(self) -> None:
        """Test updating with empty metadata dict (valid no-op)."""
        # Arrange
        self._mock_session_metadata_endpoint(status=200)

        # Act
        result = self.client.metadata.update_session(
            project_id=self.project_id, session_id=self.session_id, metadata={}
        )

        # Assert
        self.assertIsNotNone(result)
        self._assert_request_made_with_metadata({})

    @responses.activate
    def test_update_metadata_various_types(self) -> None:
        """Test metadata with string, int, float, and bool types."""
        # Arrange
        complex_metadata = {
            "string": "test",
            "integer": 123,
            "float": 45.67,
            "boolean_true": True,
            "boolean_false": False,
            "zero": 0,
            "negative": -42,
        }
        self._mock_session_metadata_endpoint(status=200)

        # Act
        self.client.metadata.update_session(
            project_id=self.project_id,
            session_id=self.session_id,
            metadata=complex_metadata,
        )

        # Assert
        self._assert_request_made_with_metadata(complex_metadata)

    # ========== Helper Methods ==========

    def _mock_session_metadata_endpoint(
        self,
        status: int = 200,
        response_body: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Mock the session metadata PATCH endpoint.

        Args:
            status: HTTP status code to return (default: 200)
            response_body: JSON response body (default: success message)
            project_id: Override project ID (default: self.project_id)
            session_id: Override session ID (default: self.session_id)
        """
        if response_body is None:
            response_body = {"message": "Metadata updated successfully"}

        project_id = project_id or self.project_id
        session_id = session_id or self.session_id

        responses.patch(
            url=f"{self.api_base}/v2/projects/{project_id}/sessions/id/{session_id}/metadata",
            status=status,
            content_type="application/json",
            json=response_body,
        )

    def _mock_trace_metadata_endpoint(
        self,
        status: int = 200,
        response_body: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Mock the trace metadata PATCH endpoint.

        Args:
            status: HTTP status code to return (default: 200)
            response_body: JSON response body (default: success message)
            project_id: Override project ID (default: self.project_id)
            session_id: Override session ID (default: self.session_id)
            trace_id: Override trace ID (default: self.trace_id)
        """
        if response_body is None:
            response_body = {"message": "Metadata updated successfully"}

        project_id = project_id or self.project_id
        session_id = session_id or self.session_id
        trace_id = trace_id or self.trace_id

        responses.patch(
            url=f"{self.api_base}/v2/projects/{project_id}/sessions/{session_id}/traces/id/{trace_id}/metadata",
            status=status,
            content_type="application/json",
            json=response_body,
        )

    def _assert_request_made_with_metadata(
        self, expected_metadata: CustomMetadata, call_index: int = 0
    ) -> None:
        """
        Assert that a request was made with the expected metadata.

        Args:
            expected_metadata: Expected metadata in request body
            call_index: Index of the request to check (default: 0 = first)
        """
        self.assertGreater(len(responses.calls), call_index)
        request_body_raw = responses.calls[call_index].request.body
        self.assertIsNotNone(request_body_raw, "Request body should not be None")
        request_body = json.loads(request_body_raw)  # type: ignore
        self.assertEqual(request_body, expected_metadata)

    def _assert_patch_method_used(self, call_index: int = 0) -> None:
        """
        Assert that PATCH HTTP method was used.

        Args:
            call_index: Index of the request to check (default: 0 = first)
        """
        self.assertGreater(len(responses.calls), call_index)
        self.assertEqual(responses.calls[call_index].request.method, "PATCH")
