import json
from unittest import TestCase
from uuid import uuid4

import responses
import respx

from freeplay.provider_config import ProviderConfig, OpenAIConfig
from freeplay.flavors import OpenAIChat
from freeplay.freeplay import Freeplay


class TestFreeplayTestRun(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.freeplay_api_key = "freeplay_api_key"
        self.openai_api_key = "openai_api_key"
        self.api_base = "http://localhost:9091/api"
        self.openai_base_url = "http://localhost:666/v1"
        self.project_id = str(uuid4())
        self.session_id = str(uuid4())
        self.project_version_id = str(uuid4())
        self.prompt_template_version_id = self.project_version_id
        self.openai_chat_prompt_template_id = str(uuid4())
        self.test_run_id = str(uuid4())
        self.flavor = OpenAIChat()
        self.provider_config = ProviderConfig(
            openai=OpenAIConfig(api_key=self.openai_api_key, base_url=self.openai_base_url))
        self.record_url = f'{self.api_base}/v1/record'
        self.tag = 'latest'

    @responses.activate
    @respx.mock
    def test_test_runs(self) -> None:
        respx.post(f'{self.openai_base_url}/chat/completions').respond(
            status_code=200,
            text=self.__openai_chat_response(),
        )
        responses.post(
            url=f'{self.api_base}/projects/{self.project_id}/test-runs-cases',
            status=201,
            body=self.__freeplay_test_runs_response(self.test_run_id),
            content_type='application/json'
        )

        responses.post(
            url=f'{self.api_base}/projects/{self.project_id}/sessions/tag/{self.tag}',
            status=201,
            body=self.__session_create_response(session_id=self.session_id),
            content_type='application/json'
        )

        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/{self.tag}',
            status=200,
            body=self.__get_templates_response(),
            content_type='application/json'
        )

        responses.post(
            url=f'{self.api_base}/v1/record',
            status=201,
            content_type='application/json'
        )

        freeplay = Freeplay(
            flavor=self.flavor,
            freeplay_api_key=self.freeplay_api_key,
            provider_config=self.provider_config,
            api_base=self.api_base)

        test_run = freeplay.create_test_run(self.project_id, testlist='good stuff')

        self.assertEqual(2, len(test_run.get_inputs()))

        for inputs in test_run.get_inputs():
            freeplay_session = test_run.create_session(self.project_id)

            completion = freeplay_session.get_completion(
                template_name="my-prompt",
                variables=inputs
            )

            self.assertEqual(True, completion.is_complete)
            self.assertEqual("I am your assistant", completion.content)

        record_api_request = responses.calls[2].request
        recorded_body_dom = json.loads(record_api_request.body)
        self.assertEqual('Bearer freeplay_api_key', record_api_request.headers['Authorization'])
        self.assertEqual(True, recorded_body_dom['is_complete'])
        self.assertEqual(self.project_version_id, recorded_body_dom['project_version_id'])
        self.assertEqual(self.openai_chat_prompt_template_id, recorded_body_dom['prompt_template_id'])
        self.assertIsNotNone(recorded_body_dom['start_time'])
        self.assertIsNotNone(recorded_body_dom['end_time'])
        self.assertEqual(self.tag, recorded_body_dom['tag'])
        self.assertEqual(
            '[{"content": "Answer this question: Why isn\'t my internet working?", "role": "system"}]',
            recorded_body_dom['prompt_content']
        )
        self.assertEqual("I am your assistant", recorded_body_dom['return_content'])
        self.assertEqual('openai_chat', recorded_body_dom['format_type'])
        self.assertEqual(self.record_url, record_api_request.url)
        self.assertEqual(self.test_run_id, recorded_body_dom['test_run_id'])

        record_api_request_2 = responses.calls[4].request
        recorded_body_dom_2 = json.loads(record_api_request_2.body)

        self.assertEqual(
            '[{"content": "Answer this question: What does blue look like?", "role": "system"}]',
            recorded_body_dom_2['prompt_content']
        )
        self.assertEqual(self.test_run_id, recorded_body_dom_2['test_run_id'])

    @staticmethod
    def __freeplay_test_runs_response(test_run_id: str) -> str:
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

    @staticmethod
    def __openai_chat_response() -> str:
        return json.dumps({
            'id': 'chatcmpl-7R1UPz7UYFVU1K5Wk9b6fS1R3Spsc',
            'object': 'chat.completion',
            'created': 1686674937,
            'model': 'gpt-3.5-turbo-0301',
            'usage': {
                'prompt_tokens': 37,
                'completion_tokens': 178,
                'total_tokens': 215
            },
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'I am your assistant'
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ]
        })

    @staticmethod
    def __session_create_response(session_id: str) -> str:
        return json.dumps({
            'session_id': session_id,
            'prompt_templates': [
                {'name': 'my-prompt', 'content': 'Answer this question: {question}', 'messages': []},
                {'name': 'my-template-2', 'content': 'This is some other prompt', 'messages': []},
                {
                    'name': 'my-chat-template',
                    'content': 'YAML_METADATA [{"role": "system", "content": "YAML_METADATA This is a system message"},'
                               '{"role": "user", "content": "This is a user message"}]',
                    'messages': [
                        {'role': 'system', 'content': 'YAML_METADATA This is a system message'},
                        {'role': 'user', 'content': 'This is a user message about {topic}'}
                    ]},
            ]
        })

    def __get_templates_response(self) -> str:
        return json.dumps({
            'templates': [
                {
                    'content': json.dumps([{
                        "role": "system",
                        "content": "Answer this question: {{question}}"
                    }]),
                    'name': 'my-prompt',
                    'project_version_id': self.project_version_id,
                    'prompt_template_version_id': self.prompt_template_version_id,
                    'prompt_template_id': self.openai_chat_prompt_template_id
                }
            ]
        })
