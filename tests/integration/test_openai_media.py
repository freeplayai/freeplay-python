import os
import time
import unittest
from typing import cast

from openai import OpenAI

from freeplay import CallInfo, Freeplay, RecordPayload
from freeplay.model import MediaInputBase64, MediaInputMap
from tests.integration.data_support import encode_test_data
from tests.slow_test_support import slow


class TestOpenAIMedia(unittest.TestCase):
    @slow
    def setUp(self) -> None:
        self.freeplay_client = Freeplay(
            freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
            api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
        )
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.project_id = os.environ["FREEPLAY_PROJECT_ID"]

    @slow
    def test_openai_audio(self) -> None:
        input_variables = {"query": "Describe what you hear"}
        media_inputs: MediaInputMap = {
            'some-audio': MediaInputBase64(
                type="base64",
                content_type="audio/mpeg",
                data=encode_test_data("birds.mp3")
            )
        }
        formatted_prompt = self.freeplay_client.prompts.get_formatted(
            project_id=self.project_id,
            template_name="media-audio",
            environment="latest",
            variables=input_variables,
            media_inputs=media_inputs,
            flavor_name="openai_chat"
        )
        formatted_prompt.prompt_info.model = "gpt-4o-audio-preview"
        formatted_prompt.prompt_info.model_parameters["max_tokens"] = 2000
        completion = self.openai_client.chat.completions.create(
            messages=formatted_prompt.llm_prompt,
            model=formatted_prompt.prompt_info.model,
            **formatted_prompt.prompt_info.model_parameters,
        )

        response_content = cast(str, completion.choices[0].message.content)
        self.assertIn("bird", response_content)

        record_response = self.freeplay_client.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=[
                    *formatted_prompt.llm_prompt,
                    {"role": "assistant", "content": response_content}
                ],
                session_info=self.freeplay_client.sessions.create().session_info,
                inputs=input_variables,
                media_inputs=media_inputs,
                prompt_version_info=formatted_prompt.prompt_info,
                call_info=CallInfo.from_prompt_info(formatted_prompt.prompt_info, time.time(), time.time() + 1),
            )
        )

        self.assertIsNotNone(record_response.completion_id)

    @slow
    def test_openai_image(self) -> None:
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
            flavor_name="openai_chat"
        )
        formatted_prompt.prompt_info.model = "gpt-4o"
        formatted_prompt.prompt_info.model_parameters["max_tokens"] = 2000
        completion = self.openai_client.chat.completions.create(
            messages=formatted_prompt.llm_prompt,
            model=formatted_prompt.prompt_info.model,
            **formatted_prompt.prompt_info.model_parameters,
        )

        response_content = cast(str, completion.choices[0].message.content)
        self.assertIn("whale", response_content)

        record_response = self.freeplay_client.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=[
                    *formatted_prompt.llm_prompt,
                    {"role": "assistant", "content": response_content}
                ],
                session_info=self.freeplay_client.sessions.create().session_info,
                inputs=input_variables,
                media_inputs=media_inputs,
                prompt_version_info=formatted_prompt.prompt_info,
                call_info=CallInfo.from_prompt_info(formatted_prompt.prompt_info, time.time(), time.time() + 1),
            )
        )

        self.assertIsNotNone(record_response.completion_id)

    @slow
    def test_openai_file(self) -> None:
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
            flavor_name="openai_chat"
        )
        formatted_prompt.prompt_info.model = "gpt-4o"
        formatted_prompt.prompt_info.model_parameters["max_tokens"] = 2000
        completion = self.openai_client.chat.completions.create(
            messages=formatted_prompt.llm_prompt,
            model=formatted_prompt.prompt_info.model,
            **formatted_prompt.prompt_info.model_parameters,
        )

        response_content = cast(str, completion.choices[0].message.content)
        self.assertIn("Portugal", response_content)

        record_response = self.freeplay_client.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=[
                    *formatted_prompt.llm_prompt,
                    {"role": "assistant", "content": response_content}
                ],
                session_info=self.freeplay_client.sessions.create().session_info,
                inputs=input_variables,
                media_inputs=media_inputs,
                prompt_version_info=formatted_prompt.prompt_info,
                call_info=CallInfo.from_prompt_info(formatted_prompt.prompt_info, time.time(), time.time() + 1),
            )
        )

        self.assertIsNotNone(record_response.completion_id)
