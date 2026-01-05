"""
Metadata Resource for Freeplay SDK

This module provides the ability to update session and trace metadata after creation.

Why this was created:
- Customers need to associate IDs and metadata with sessions/traces after the conversation ends
  (e.g., ticket IDs, summary IDs, resolution status generated post-conversation)
- Without this, users had to log dummy completions just to update metadata
- Provides a clean API for metadata updates without additional trace/session creation

What it does:
- Exposes `client.metadata.update_session()` to update session metadata
- Exposes `client.metadata.update_trace()` to update trace metadata
- Uses merge semantics: new keys overwrite existing keys, preserving unmentioned keys
- Returns MetadataUpdateResponse indicating successful update

Usage:
    client.metadata.update_session(
        project_id=project_id,
        session_id=session_id,
        metadata={"ticket_id": "TICKET-123", "status": "resolved"}
    )
"""

from dataclasses import dataclass

from freeplay.support import CallSupport, CustomMetadata


@dataclass
class MetadataUpdateResponse:
    pass


class Metadata:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def update_session(
        self, project_id: str, session_id: str, metadata: CustomMetadata
    ) -> MetadataUpdateResponse:
        """
        Update session metadata. New keys overwrite existing keys.

        Args:
            project_id: The project ID
            session_id: The session ID
            metadata: Dictionary of metadata key-value pairs to update

        Returns:
            MetadataUpdateResponse: Empty response indicating success

        Raises:
            FreeplayError: If the session is not found or the update fails
        """
        self.call_support.update_session_metadata(project_id, session_id, metadata)
        return MetadataUpdateResponse()

    def update_trace(
        self,
        project_id: str,
        session_id: str,
        trace_id: str,
        metadata: CustomMetadata,
    ) -> MetadataUpdateResponse:
        """
        Update trace metadata. New keys overwrite existing keys.

        Args:
            project_id: The project ID
            session_id: The session ID (for RESTful hierarchy)
            trace_id: The trace ID
            metadata: Dictionary of metadata key-value pairs to update

        Returns:
            MetadataUpdateResponse: Empty response indicating success

        Raises:
            FreeplayError: If the trace is not found or the update fails
        """
        self.call_support.update_trace_metadata(
            project_id, session_id, trace_id, metadata
        )
        return MetadataUpdateResponse()
