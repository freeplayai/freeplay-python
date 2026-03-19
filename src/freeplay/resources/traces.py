from dataclasses import dataclass
from typing import Dict, Optional, Union

from freeplay.errors import FreeplayClientError
from freeplay.model import JSONValue
from freeplay.support import CallSupport


@dataclass
class TraceUpdatePayload:
    project_id: str
    session_id: str
    trace_id: str
    output: Optional[JSONValue] = None
    eval_results: Optional[Dict[str, Union[bool, float]]] = None


class Traces:
    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def update(self, payload: TraceUpdatePayload) -> None:
        if payload.output is None and payload.eval_results is None:
            raise FreeplayClientError(
                "At least one of 'output' or 'eval_results' must be provided"
            )
        self.call_support.update_trace(
            project_id=payload.project_id,
            session_id=payload.session_id,
            trace_id=payload.trace_id,
            output=payload.output,
            eval_results=payload.eval_results,
        )
