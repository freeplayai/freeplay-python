import os
import unittest
from time import time

import boto3  # type: ignore

from freeplay import CallInfo, Freeplay, RecordPayload
from tests.slow_test_support import slow


class TestBedrockConverse(unittest.TestCase):
    @slow
    def setUp(self) -> None:
        self.freeplay_client = Freeplay(
            freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
            api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
        )
        self.project_id = os.environ["FREEPLAY_PROJECT_ID"]
        self.converse_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    @slow
    def test_bedrock_converse(self) -> None:
        start_time = time()
        input_variables = {"query": "What is the capital of the moon?"}
        formatted_prompt = self.freeplay_client.prompts.get_formatted(
            project_id=self.project_id,
            template_name="bedrock-converse",
            environment="latest",
            variables=input_variables,
        )
        response = self.converse_client.converse(
            modelId=formatted_prompt.prompt_info.model,
            messages=formatted_prompt.llm_prompt,
            system=[{"text": formatted_prompt.system_content or ""}],
            inferenceConfig=formatted_prompt.prompt_info.model_parameters,
        )
        output_message = response["output"]["message"]
        response_content = output_message["content"][0]["text"]

        self.assertGreater(len(response_content), 0)

        record_response = self.freeplay_client.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=[
                    *formatted_prompt.llm_prompt,
                    output_message,
                ],
                session_info=self.freeplay_client.sessions.create().session_info,
                inputs=input_variables,
                prompt_info=formatted_prompt.prompt_info,
                call_info=CallInfo.from_prompt_info(
                    formatted_prompt.prompt_info, start_time, time()
                ),
            )
        )

        self.assertIsNotNone(record_response.completion_id)

    @slow
    def test_bedrock_converse_with_tool_calls(self) -> None:
        start_time = time()
        input_variables = {"query": "What is the latest news in Boulder, CO?"}
        formatted_prompt = self.freeplay_client.prompts.get_formatted(
            project_id=self.project_id,
            template_name="bedrock-converse-tool",
            environment="latest",
            variables=input_variables,
        )
        response = self.converse_client.converse(
            modelId=formatted_prompt.prompt_info.model,
            messages=formatted_prompt.llm_prompt,
            system=[{"text": formatted_prompt.system_content or ""}],
            inferenceConfig=formatted_prompt.prompt_info.model_parameters,
            toolConfig=formatted_prompt.tool_schema,
        )

        output_message = response["output"]["message"]
        tool_call = output_message["content"][0]["toolUse"]
        self.assertGreater(len(tool_call), 0)

        record_response = self.freeplay_client.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=[
                    *formatted_prompt.llm_prompt,
                    output_message,
                ],
                session_info=self.freeplay_client.sessions.create().session_info,
                inputs=input_variables,
                prompt_info=formatted_prompt.prompt_info,
                call_info=CallInfo.from_prompt_info(
                    formatted_prompt.prompt_info, start_time, time()
                ),
            )
        )

        self.assertIsNotNone(record_response.completion_id)
