from typing import Optional

from freeplay.errors import FreeplayConfigurationError
from freeplay.resources.customer_feedback import CustomerFeedback
from freeplay.resources.prompts import Prompts, APITemplateResolver, TemplateResolver
from freeplay.resources.recordings import Recordings
from freeplay.resources.sessions import Sessions
from freeplay.resources.test_cases import TestCases
from freeplay.resources.test_runs import TestRuns
from freeplay.support import CallSupport


class Freeplay:
    def __init__(
            self,
            freeplay_api_key: str,
            api_base: str,
            template_resolver: Optional[TemplateResolver] = None
    ) -> None:
        if not freeplay_api_key or not freeplay_api_key.strip():
            raise FreeplayConfigurationError("Freeplay API key not set. It must be set to the Freeplay API.")

        self.call_support = CallSupport(
            freeplay_api_key,
            api_base
        )
        self.freeplay_api_key = freeplay_api_key
        self.api_base = api_base

        resolver: TemplateResolver
        if template_resolver is None:
            resolver = APITemplateResolver(self.call_support)
        else:
            resolver = template_resolver

        # Resources ========
        self.customer_feedback = CustomerFeedback(self.call_support)
        self.prompts = Prompts(self.call_support, resolver)
        self.recordings = Recordings(self.call_support)
        self.sessions = Sessions(self.call_support)
        self.test_runs = TestRuns(self.call_support)
        self.test_cases = TestCases(self.call_support)