from dataclasses import dataclass
from typing import Dict, Union

from freeplay.model import FeedbackValue
from freeplay.support import CallSupport


@dataclass
class CustomerFeedbackResponse:
    pass


@dataclass
class TraceFeedbackResponse:
    pass


class CustomerFeedback:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def update(self, completion_id: str, feedback: Dict[str, FeedbackValue]) -> CustomerFeedbackResponse:
        self.call_support.update_customer_feedback(completion_id, feedback)
        return CustomerFeedbackResponse()

    def update_trace(
            self,
            project_id: str,
            trace_id: str,
            feedback: Dict[str, FeedbackValue]
    ) -> TraceFeedbackResponse:
        self.call_support.update_trace_feedback(project_id, trace_id, feedback)
        return TraceFeedbackResponse()
