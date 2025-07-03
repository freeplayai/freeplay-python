import os
import time
import unittest
from typing import cast

from anthropic import Anthropic, NotGiven

from freeplay import CallInfo, Freeplay, RecordPayload
from freeplay.model import MediaInputBase64, MediaInputMap, TextBlock
from tests.integration.data_support import encode_test_data
from tests.slow_test_support import slow


class TestAnthropicMedia(unittest.TestCase):
    @slow
    def setUp(self) -> None:
        self.freeplay_client = Freeplay(
            freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
            api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
        )
        self.anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.project_id = os.environ["FREEPLAY_PROJECT_ID"]

    @slow
    def test_anthropic_image(self) -> None:
        input_variables = {"query": "Describe what you see"}
        media_inputs: MediaInputMap = {
            'some-image': MediaInputBase64(
                type="base64",
                content_type="image/jpeg",
                data=encode_test_data("whale.jpg")
            )
        }
        formatted_prompt = self.freeplay_client.prompts.get_formatted(
            project_id=self.project_id,
            template_name="media-image",
            environment="latest",
            variables=input_variables,
            media_inputs=media_inputs,
            flavor_name="anthropic_chat"
        )
        formatted_prompt.prompt_info.model = "claude-3-5-haiku-latest"
        formatted_prompt.prompt_info.model_parameters["max_tokens"] = 2000

        completion = self.anthropic_client.messages.create(
            system=formatted_prompt.system_content or NotGiven(),
            messages=formatted_prompt.llm_prompt,
            model=formatted_prompt.prompt_info.model,
            **formatted_prompt.prompt_info.model_parameters,
        )

        response_content = cast(TextBlock, completion.content[0]).text
        self.assertIn("whale", response_content)

        record_response = self.freeplay_client.recordings.create(
            RecordPayload(
                all_messages=[
                    *formatted_prompt.llm_prompt,
                    {"role": "assistant", "content": response_content}
                ],
                session_info=self.freeplay_client.sessions.create().session_info,
                inputs=input_variables,
                media_inputs=media_inputs,
                prompt_info=formatted_prompt.prompt_info,
                call_info=CallInfo.from_prompt_info(formatted_prompt.prompt_info, time.time(), time.time() + 1),
            )
        )

        self.assertIsNotNone(record_response.completion_id)

    @slow
    def test_anthropic_file(self) -> None:
        input_variables = {"query": "Describe this document"}
        media_inputs: MediaInputMap = {
            'some-file': MediaInputBase64(
                type="base64",
                content_type="application/pdf",
                data=encode_test_data("portugal.pdf")
            )
        }
        formatted_prompt = self.freeplay_client.prompts.get_formatted(
            project_id=self.project_id,
            template_name="media-file",
            environment="latest",
            variables=input_variables,
            media_inputs=media_inputs,
            flavor_name="anthropic_chat"
        )
        formatted_prompt.prompt_info.model = "claude-3-5-haiku-latest"
        formatted_prompt.prompt_info.model_parameters["max_tokens"] = 2000

        completion = self.anthropic_client.messages.create(
            system=formatted_prompt.system_content or NotGiven(),
            messages=formatted_prompt.llm_prompt,
            model=formatted_prompt.prompt_info.model,
            **formatted_prompt.prompt_info.model_parameters,
        )

        response_content = cast(TextBlock, completion.content[0]).text
        self.assertIn("Portugal", response_content)

        record_response = self.freeplay_client.recordings.create(
            RecordPayload(
                all_messages=[
                    *formatted_prompt.llm_prompt,
                    {"role": "assistant", "content": response_content}
                ],
                session_info=self.freeplay_client.sessions.create().session_info,
                inputs=input_variables,
                media_inputs=media_inputs,
                prompt_info=formatted_prompt.prompt_info,
                call_info=CallInfo.from_prompt_info(formatted_prompt.prompt_info, time.time(), time.time() + 1),
            )
        )

        self.assertIsNotNone(record_response.completion_id)
