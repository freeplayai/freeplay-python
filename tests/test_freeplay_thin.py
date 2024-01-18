import json
import time
from unittest import TestCase
from uuid import uuid4

import responses

from freeplay.errors import (FreeplayClientError,
                             FreeplayConfigurationError)
from freeplay.thin import Freeplay, CallInfo, ResponseInfo, RecordPayload


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
        self.test_run_id = str(uuid4())

        self.freeplay_thin = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base
        )

    @responses.activate
    def test_single_prompt_get_and_record(self) -> None:
        self.__mock_freeplay_apis()

        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}

        formatted_prompt = self.freeplay_thin.prompts.get_formatted(
            project_id=self.project_id,
            template_name="my-chat-prompt",
            environment=self.tag,
            variables={"name": "Sparkles", "question": "Why isn't my door working"}
        )

        session = self.freeplay_thin.sessions.create()
        start = time.time()
        end = start + 5
        openai_response = 'This is the response from Anthropic'
        all_messages = formatted_prompt.all_messages({'role': 'Assistant', 'content': ('%s' % openai_response)})

        call_info = CallInfo(
            provider=formatted_prompt.prompt_info.provider,
            model=formatted_prompt.prompt_info.model,
            start_time=start,
            end_time=end,
            model_parameters=formatted_prompt.prompt_info.model_parameters
        )
        response_info = ResponseInfo(
            is_complete=True
        )

        self.freeplay_thin.recordings.create(
            RecordPayload(
                all_messages=all_messages,
                inputs=input_variables,
                session_id=session.session_id,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                response_info=response_info
            )
        )

        self.assertTrue(str(formatted_prompt.llm_prompt).startswith("\n\nHuman: System message"))

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual(True, recorded_body_dom['is_complete'])
        self.assertEqual(self.project_version_id, recorded_body_dom['project_version_id'])
        self.assertEqual(self.prompt_template_id_1, recorded_body_dom['prompt_template_id'])
        self.assertIsNotNone(recorded_body_dom['start_time'])
        self.assertIsNotNone(recorded_body_dom['end_time'])
        self.assertEqual(self.tag, recorded_body_dom['tag'])
        self.assertEqual(
            '[{"role": "system", "content": "System message"}, '
            '{"role": "Assistant", "content": "How may I help you, Sparkles?"}, '
            '{"role": "user", "content": "Why isn\'t my door working"}]',
            recorded_body_dom['prompt_content']
        )
        self.assertEqual(openai_response, recorded_body_dom['return_content'])
        self.assertEqual('anthropic', recorded_body_dom['provider'])
        self.assertEqual(0.7, recorded_body_dom['llm_parameters']['temperature'])
        self.assertEqual(50, recorded_body_dom['llm_parameters']['max_tokens_to_sample'])

    @responses.activate
    def test_get_template_prompt_then_populate(self) -> None:
        self.__mock_freeplay_apis()

        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}

        template_prompt = self.freeplay_thin.prompts.get(
            project_id=self.project_id,
            template_name="my-chat-prompt",
            environment=self.tag
        )

        self.assertTrue("{{question}}" in template_prompt.messages[2]['content'])

        bound_prompt = template_prompt.bind(input_variables)

        self.assertFalse("{{question}}" in bound_prompt.messages[2]['content'])
        self.assertTrue(input_variables.get('name') in bound_prompt.messages[1]['content'])  # type: ignore
        self.assertTrue(input_variables.get('question') in bound_prompt.messages[2]['content'])  # type: ignore

        formatted_prompt = bound_prompt.format()

        self.assertTrue(str(formatted_prompt.llm_prompt).startswith("\n\nHuman: System message"))

        openai_response = 'This is the response from Anthropic'
        all_messages = formatted_prompt.all_messages({'role': 'Assistant', 'content': ('%s' % openai_response)})

        self.assertTrue(input_variables.get('name') in formatted_prompt.llm_prompt)  # type: ignore
        self.assertTrue(input_variables.get('question') in formatted_prompt.llm_prompt)  # type: ignore
        self.assertTrue(openai_response in all_messages[3]['content'])

    @responses.activate
    def test_create_test_run(self) -> None:
        self.__mock_freeplay_apis()

        test_run = self.freeplay_thin.test_runs.create(self.project_id, testlist='good stuff')

        self.assertEqual(2, len(test_run.get_test_cases()))

    @responses.activate
    def test_auth_error(self) -> None:
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/{self.tag}',
            status=401,
            body=self.__get_templates_response()
        )

        freeplay_thin = Freeplay(
            freeplay_api_key="not-the-key",
            api_base=self.api_base,
        )

        with self.assertRaisesRegex(FreeplayClientError, "Error getting prompt templates \\[401\\]"):
            freeplay_thin.prompts.get(
                project_id=self.project_id,
                template_name="my-chat-prompt",
                environment=self.tag
            )

    @responses.activate
    def test_template_not_found(self) -> None:
        self.__mock_freeplay_apis()
        with self.assertRaisesRegex(
                FreeplayConfigurationError,
                'Could not find template with name "invalid-template-id"'
        ):
            self.freeplay_thin.prompts.get(
                project_id=self.project_id,
                template_name="invalid-template-id",
                environment=self.tag
            )

    def __mock_freeplay_apis(self) -> None:
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/{self.tag}',
            status=200,
            body=self.__get_templates_response()
        )
        self.__mock_test_run_api()
        self.__mock_record_api()

    def __mock_record_api(self) -> None:
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json'
        )

    def __mock_test_run_api(self) -> None:
        responses.post(
            url=f'{self.api_base}/projects/{self.project_id}/test-runs-cases',
            status=201,
            body=self.__create_test_run_response(self.test_run_id),
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

    @staticmethod
    def __create_test_run_response(test_run_id: str) -> str:
        return json.dumps({
            'test_run_id': test_run_id,
            'test_cases': [
                {
                    'id': str(uuid4()),
                    'variables': {'question': "Why isn't my internet working?"}
                },
                {
                    'id': str(uuid4()),
                    'variables': {'question': "What does blue look like?"}
                }
            ]
        })
