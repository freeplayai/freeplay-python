from dataclasses import dataclass
from typing import Dict, Union

from freeplay.support import CallSupport


@dataclass
class CustomerFeedbackResponse:
    pass


class CustomerFeedback:
    def __init__(self, call_support: CallSupport) -> None:
        self.call_support = call_support

    def update(self, completion_id: str, feedback: Dict[str, Union[bool, str, int, float]]) -> CustomerFeedbackResponse:
        self.call_support.update_customer_feedback(completion_id, feedback)
        return CustomerFeedbackResponse()
