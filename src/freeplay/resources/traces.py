from dataclasses import dataclass
from typing import Dict, Optional, Union

from freeplay.errors import FreeplayClientError
from freeplay.model import JSONValue, TestRunInfo
from freeplay.support import CallSupport, CustomMetadata

FeedbackDict = Dict[str, Union[str, int, float, bool]]


@dataclass
class TraceUpdatePayload:
    project_id: str
    session_id: str
    trace_id: str
    output: Optional[JSONValue] = None
    metadata: Optional[CustomMetadata] = None
    feedback: Optional[FeedbackDict] = None
    eval_results: Optional[Dict[str, Union[bool, float]]] = None
    test_run_info: Optional[TestRunInfo] = None


class Traces:
    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def update(self, payload: TraceUpdatePayload) -> None:
        if (
            payload.output is None
            and payload.metadata is None
            and payload.feedback is None
            and payload.eval_results is None
            and payload.test_run_info is None
        ):
            raise FreeplayClientError(
                "At least one of 'output', 'metadata', 'feedback', 'eval_results', or 'test_run_info' must be provided"
            )
        self.call_support.update_trace(
            project_id=payload.project_id,
            session_id=payload.session_id,
            trace_id=payload.trace_id,
            output=payload.output,
            metadata=payload.metadata,
            feedback=payload.feedback,
            eval_results=payload.eval_results,
            test_run_info=payload.test_run_info,
        )
