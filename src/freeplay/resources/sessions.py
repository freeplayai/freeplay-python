import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Union

from freeplay.errors import FreeplayClientError
from freeplay.support import CallSupport

CustomMetadata = Optional[Dict[str, Union[str, int, float, bool]]]


@dataclass
class SessionInfo:
    session_id: str
    custom_metadata: CustomMetadata


class TraceInfo:
    session_id: str
    trace_id: str
    input: Optional[str] = None
    _call_support: CallSupport

    def __init__(
            self,
            trace_id: str,
            session_id: str,
            _call_support: CallSupport,
            input: Optional[str] = None,
    ):
        self.trace_id = trace_id
        self.session_id = session_id
        self.input = input
        self._call_support = _call_support

    def record_output(self, project_id: str, output: str) -> None:
        if self.input is None:
            raise FreeplayClientError("Input must be set before recording output")
        self._call_support.record_trace(project_id, self.session_id, self.trace_id, self.input, output)


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

    def create_trace(self, input: str) -> TraceInfo:
        return TraceInfo(
            trace_id=str(uuid.uuid4()),
            session_id=self.session_id,
            input=input,
            _call_support=self._call_support
        )

    def restore_trace(self, trace_id: uuid.UUID, input: Optional[str]) -> TraceInfo:
        return TraceInfo(
            trace_id=str(trace_id),
            session_id=self.session_id,
            input=input,
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
