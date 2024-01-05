import json
import time
from unittest import TestCase
from uuid import uuid4

import responses

from freeplay.completions import CompletionChunk  # type: ignore
from freeplay.errors import (FreeplayClientError,  # type: ignore
                             FreeplayConfigurationError,
                             LLMServerError)
from freeplay.flavors import OpenAIChat  # type: ignore
from freeplay.freeplay import Freeplay  # type: ignore
from freeplay.freeplay_thin import FreeplayThin, CallInfo, ResponseInfo, RecordPayload  # type: ignore
from freeplay.provider_config import ProviderConfig, OpenAIConfig  # type: ignore
from freeplay.record import no_op_recorder  # type: ignore


class TestFreeplay(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.freeplay_api_key = "freeplay_api_key"
        self.openai_api_key = "openai_api_key"
        self.api_base = "http://localhost:9091/api"
        self.project_id = str(uuid4())
        self.project_version_id = str(uuid4())
        self.prompt_template_version_id = self.project_version_id
        self.prompt_template_id_1 = str(uuid4())
        self.record_url = f'{self.api_base}/v1/record'
        self.tag = 'test-tag'

        self.freeplay_thin = FreeplayThin(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base
        )

    @responses.activate
    def test_single_prompt_get_and_record(self) -> None:
        self.__mock_freeplay_apis()

        prompt_info, messages = self.freeplay_thin.get_prompt(
            project_id=self.project_id,
            template_name="my-chat-prompt",
            environment=self.tag,
            variables={"name": "Sparkles", "question": "Why isn't my door working"}
        )

        ready_for_llm = self.freeplay_thin.format(prompt_info.flavor_name, messages)

        session = self.freeplay_thin.create_session()
        start = time.time()
        end = start + 5
        openai_response = 'This is the response from Anthropic'
        messages.append({'role': 'Assistant', 'content': ('%s' % openai_response)})

        call_info = CallInfo(
            provider=prompt_info.provider,
            model=prompt_info.model,
            start_time=start,
            end_time=end,
            model_parameters=prompt_info.model_parameters
        )
        response_info = ResponseInfo(
            is_complete=True
        )

        self.freeplay_thin.record_call(
            RecordPayload(
                all_messages=messages,
                session_id=session.session_id,
                prompt_info=prompt_info,
                call_info=call_info,
                response_info=response_info
            )
        )

        self.assertTrue(ready_for_llm.startswith("\n\nHuman: System message"))

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual(True, recorded_body_dom['is_complete'])
        self.assertEqual(self.project_version_id, recorded_body_dom['project_version_id'])
        self.assertEqual(self.prompt_template_id_1, recorded_body_dom['prompt_template_id'])
        self.assertIsNotNone(recorded_body_dom['start_time'])
        self.assertIsNotNone(recorded_body_dom['end_time'])
        self.assertEqual(self.tag, recorded_body_dom['tag'])
        self.assertEqual(
            '[{"content": "System message", "role": "system"}, '
            '{"content": "How may I help you, Sparkles?", "role": "Assistant"}, '
            '{"content": "Why isn\'t my door working", "role": "user"}]',
            recorded_body_dom['prompt_content']
        )
        self.assertEqual(openai_response, recorded_body_dom['return_content'])
        self.assertEqual('anthropic', recorded_body_dom['provider'])
        self.assertEqual(0.7, recorded_body_dom['llm_parameters']['temperature'])
        self.assertEqual(50, recorded_body_dom['llm_parameters']['max_tokens_to_sample'])

    @responses.activate
    def test_auth_error(self) -> None:
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/{self.tag}',
            status=401,
            body=self.__get_templates_response()
        )

        freeplay_thin = FreeplayThin(
            freeplay_api_key="not-the-key",
            api_base=self.api_base,
        )

        with self.assertRaisesRegex(FreeplayClientError, "Error getting prompt templates \\[401\\]"):
            freeplay_thin.get_prompt(
                project_id=self.project_id,
                template_name="my-chat-prompt",
                environment=self.tag,
                variables={}
            )

    @responses.activate
    def test_template_not_found(self) -> None:
        self.__mock_freeplay_apis()
        with self.assertRaisesRegex(
                FreeplayConfigurationError,
                'Could not find template with name "invalid-template-id"'
        ):
            self.freeplay_thin.get_prompt(
                project_id=self.project_id,
                template_name="invalid-template-id",
                environment=self.tag,
                variables={}
            )

    def __mock_freeplay_apis(self) -> None:
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/{self.tag}',
            status=200,
            body=self.__get_templates_response()
        )
        self.__mock_record_api()

    def __mock_record_api(self) -> None:
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json'
        )

    def __get_templates_response(self) -> str:
        return json.dumps({
            'templates': [
                {
                    'content': json.dumps([{
                        "role": "system",
                        "content": "System message"
                    }, {
                        "role": "Assistant",
                        "content": "How may I help you, {{name}}?"
                    }, {
                        "role": "user",
                        "content": "{{question}}"
                    }]),
                    'name': 'my-chat-prompt',
                    'project_version_id': self.project_version_id,
                    'prompt_template_version_id': self.prompt_template_version_id,
                    'prompt_template_id': self.prompt_template_id_1,
                    'flavor_name': 'anthropic_chat',
                    'params': {
                        'model': 'claude-2.1',
                        'max_tokens_to_sample': 50,
                        'temperature': 0.7
                    }
                },
            ]
        })
