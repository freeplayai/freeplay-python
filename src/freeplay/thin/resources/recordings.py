import json

from freeplay.completions import PromptTemplateWithMetadata
from freeplay.errors import FreeplayClientError
from freeplay.record import RecordCallFields
from freeplay.support import CallSupport
from freeplay.thin.model import RecordPayload


class Recordings:
    def __init__(self, call_support: CallSupport):
        self.call_support = call_support

    def create(self, record_payload: RecordPayload) -> None:
        if len(record_payload.all_messages) < 1:
            raise FreeplayClientError("Messages list must have at least one message. "
                                      "The last message should be the current response.")

        completion = record_payload.all_messages[-1]
        history_as_string = json.dumps(record_payload.all_messages[0:-1])

        template = PromptTemplateWithMetadata(
            prompt_template_id=record_payload.prompt_info.prompt_template_id,
            prompt_template_version_id=record_payload.prompt_info.prompt_template_version_id,
            name=record_payload.prompt_info.template_name,
            content=history_as_string,
            flavor_name=record_payload.prompt_info.flavor_name,
            params=record_payload.prompt_info.model_parameters
        )

        self.call_support.record_processor.record_call(
            RecordCallFields(
                formatted_prompt=history_as_string,
                completion_content=completion['content'],
                completion_is_complete=record_payload.response_info.is_complete,
                start=record_payload.call_info.start_time,
                end=record_payload.call_info.end_time,
                session_id=record_payload.session_id,
                target_template=template,
                variables=record_payload.inputs,
                tag=record_payload.prompt_info.environment,
                test_run_id=record_payload.test_run_info.test_run_id if record_payload.test_run_info else None,
                test_case_id=record_payload.test_run_info.test_case_id if record_payload.test_run_info else None,
                model=record_payload.call_info.model,
                provider=record_payload.prompt_info.provider,
                llm_parameters=record_payload.call_info.model_parameters,
                record_format_type=None  # This is deprecated and unused in the API
            )
        )