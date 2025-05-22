import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Union

from freeplay.errors import FreeplayClientError
from freeplay.model import TestRunInfo
from freeplay.support import CallSupport, CustomMetadata


@dataclass
class SessionInfo:
    session_id: str
    custom_metadata: CustomMetadata


class TraceInfo:
    session_id: str
    trace_id: str
    input: Optional[str] = None
    agent_name: Optional[str] = None
    custom_metadata: CustomMetadata = None
    _call_support: CallSupport

    def __init__(
            self,
            trace_id: str,
            session_id: str,
            _call_support: CallSupport,
            input: Optional[str] = None,
            agent_name: Optional[str] = None,
            custom_metadata: CustomMetadata = None,
    ):
        self.trace_id = trace_id
        self.session_id = session_id
        self.input = input
        self.agent_name = agent_name
        self.custom_metadata = custom_metadata
        self._call_support = _call_support

    def record_output(
            self,
            project_id: str,
            output: str,
            eval_results: Optional[Dict[str, Union[bool, float]]] = None,
            test_run_info: Optional[TestRunInfo] = None
    ) -> None:
        if self.input is None:
            raise FreeplayClientError("Input must be set before recording output")
        self._call_support.record_trace(
            project_id,
            self.session_id,
            self.trace_id,
            self.input,
            output,
            agent_name=self.agent_name,
            custom_metadata=self.custom_metadata,
            eval_results=eval_results,
            test_run_info=test_run_info
        )


@dataclass
class Session:
    session_id: str
    custom_metadata: CustomMetadata

    def __init__(self, session_id: str, custom_metadata: CustomMetadata, _call_support: CallSupport):
        self.session_id = session_id
        self.custom_metadata = custom_metadata
        self._session_info = SessionInfo(self.session_id, self.custom_metadata)
        self._call_support = _call_support

    @property
    def session_info(self) -> SessionInfo:
        return self._session_info

    def create_trace(
            self,
            input: str,
            agent_name: Optional[str] = None,
            custom_metadata: CustomMetadata = None
    ) -> TraceInfo:
        return TraceInfo(
            trace_id=str(uuid.uuid4()),
            session_id=self.session_id,
            input=input,
            agent_name=agent_name,
            custom_metadata=custom_metadata,
            _call_support=self._call_support
        )

    def restore_trace(
            self,
            trace_id: uuid.UUID,
            input: Optional[str],
            agent_name: Optional[str] = None,
            custom_metadata: CustomMetadata = None
    ) -> TraceInfo:
        return TraceInfo(
            trace_id=str(trace_id),
            session_id=self.session_id,
            input=input,
            agent_name=agent_name,
            custom_metadata=custom_metadata,
            _call_support=self._call_support
        )


class Sessions:
    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def create(self, custom_metadata: CustomMetadata = None) -> Session:
        return Session(
            session_id=str(uuid.uuid4()),
            custom_metadata=custom_metadata,
            _call_support=self.call_support,
        )

    def delete(self, project_id: str, session_id: str) -> None:
        self.call_support.delete_session(project_id, session_id)

    def restore_session(self, session_id: str, custom_metadata: CustomMetadata = None) -> Session:
        return Session(
            session_id=session_id,
            custom_metadata=custom_metadata,
            _call_support=self.call_support,
        )
