from dataclasses import dataclass
from typing import Dict, Optional, Union

from freeplay.model import FeedbackValue, TestRunInfo
from freeplay.support import CallSupport, CustomMetadata


@dataclass
class TraceUpdateResponse:
    pass


class Traces:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def update(
        self,
        project_id: str,
        session_id: str,
        trace_id: str,
        custom_metadata: CustomMetadata = None,
        feedback: Optional[Dict[str, FeedbackValue]] = None,
        eval_results: Optional[Dict[str, Union[bool, float]]] = None,
        test_run_info: Optional[TestRunInfo] = None,
    ) -> TraceUpdateResponse:
        self.call_support.update_trace(
            project_id,
            session_id,
            trace_id,
            custom_metadata=custom_metadata,
            feedback=feedback,
            eval_results=eval_results,
            test_run_info=test_run_info,
        )
        return TraceUpdateResponse()

    def update_by_otel_span_id(
        self,
        project_id: str,
        session_id: str,
        otel_span_id_hex: str,
        custom_metadata: CustomMetadata = None,
        feedback: Optional[Dict[str, FeedbackValue]] = None,
        eval_results: Optional[Dict[str, Union[bool, float]]] = None,
        test_run_info: Optional[TestRunInfo] = None,
    ) -> TraceUpdateResponse:
        self.call_support.update_trace_by_otel_span_id(
            project_id,
            session_id,
            otel_span_id_hex,
            custom_metadata=custom_metadata,
            feedback=feedback,
            eval_results=eval_results,
            test_run_info=test_run_info,
        )
        return TraceUpdateResponse()
