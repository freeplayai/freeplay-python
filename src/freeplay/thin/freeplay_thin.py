from typing import Optional

from freeplay.errors import FreeplayConfigurationError
from freeplay.record import DefaultRecordProcessor
from freeplay.support import CallSupport
from freeplay.thin.resources.customer_feedback import CustomerFeedback
from freeplay.thin.resources.prompts import Prompts, APITemplateResolver, TemplateResolver
from freeplay.thin.resources.recordings import Recordings
from freeplay.thin.resources.sessions import Sessions
from freeplay.thin.resources.test_runs import TestRuns


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
            api_base,
            DefaultRecordProcessor(freeplay_api_key, api_base)
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
        self.sessions = Sessions()
        self.test_runs = TestRuns(self.call_support)
