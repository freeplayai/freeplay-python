import json
import time
import uuid
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional
from unittest import TestCase
from uuid import uuid4

import responses
from requests import PreparedRequest
from responses import matchers

from freeplay import Freeplay
from freeplay.errors import (FreeplayClientError,
                             FreeplayConfigurationError)
from freeplay.llm_parameters import LLMParameters
from freeplay.model import OpenAIFunctionCall
from freeplay.resources.prompts import FormattedPrompt, PromptInfo, TemplatePrompt, FilesystemTemplateResolver, \
    BoundPrompt
from freeplay.resources.recordings import RecordPayload, ResponseInfo, CallInfo
from freeplay.resources.sessions import Session


class PromptInfoMatcher:
    def __init__(self, expected: PromptInfo):
        self.expected = expected

    def __eq__(self, other: PromptInfo) -> bool:  # type: ignore
        return self.expected.prompt_template_id == other.prompt_template_id and \
            self.expected.prompt_template_version_id == other.prompt_template_version_id and \
            self.expected.template_name == other.template_name and \
            self.expected.environment == other.environment and \
            self.expected.model_parameters == other.model_parameters and \
            self.expected.provider == other.provider and \
            self.expected.model == other.model and \
            self.expected.flavor_name == other.flavor_name and \
            self.expected.provider_info == other.provider_info


class TemplatePromptMatcher:
    def __init__(self, expected: TemplatePrompt):
        self.expected = expected

    def __eq__(self, other: TemplatePrompt) -> bool:  # type: ignore
        return PromptInfoMatcher(self.expected.prompt_info) == other.prompt_info and \
            self.expected.messages == other.messages


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
        self.prompt_template_name = "my-prompt-anthropic"
        self.record_url = f'{self.api_base}/v1/record'
        self.tag = 'test-tag'
        self.test_run_id = str(uuid4())

        self.freeplay_thin = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base
        )

        self.bundle_client = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            template_resolver=FilesystemTemplateResolver(Path(__file__).parent / "test_files" / "prompts")
        )
        self.bundle_client_v2 = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            template_resolver=FilesystemTemplateResolver(Path(__file__).parent / "test_files" / "prompts_v2_format")
        )
        self.bundle_project_id = "475516c8-7be4-4d55-9388-535cef042981"
        self.anthropic_prompt_info = PromptInfo(
            prompt_template_id=str(uuid.uuid4()),
            prompt_template_version_id=str(uuid.uuid4()),
            template_name='template-name',
            environment='environment',
            model_parameters=LLMParameters({}),
            provider_info=None,
            provider='anthropic',
            model='model-name',
            flavor_name='anthropic_chat'
        )

    @responses.activate
    def test_single_prompt_get_and_record(self) -> None:
        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}
        llm_response = 'This is the response from the LLM'

        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        all_messages: List[Dict[str, str]]
        all_messages, call_info, formatted_prompt, response_info, session = self.__make_call(
            input_variables,
            llm_response
        )

        self.freeplay_thin.recordings.create(
            RecordPayload(
                all_messages=all_messages,
                inputs=input_variables,
                session_info=session.session_info,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                response_info=response_info
            )
        )

        self.assertEqual(2, len(formatted_prompt.llm_prompt))

        self.assertEqual({"anthropic_endpoint": "https://example.com/anthropic"},
                         formatted_prompt.prompt_info.provider_info)

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
            '{"role": "assistant", "content": "How may I help you, Sparkles?"}, '
            '{"role": "user", "content": "Why isn\'t my door working"}]',
            recorded_body_dom['prompt_content']
        )
        self.assertEqual(llm_response, recorded_body_dom['return_content'])
        self.assertEqual('anthropic', recorded_body_dom['provider'])
        self.assertEqual(0.7, recorded_body_dom['llm_parameters']['temperature'])
        self.assertEqual(50, recorded_body_dom['llm_parameters']['max_tokens_to_sample'])
        self.assertEqual(
            {"anthropic_endpoint": "https://example.com/anthropic"},
            recorded_body_dom['provider_info']
        )

        # Custom metadata recording
        self.assertEqual({'custom_metadata_field': 42}, recorded_body_dom['custom_metadata'])

    @responses.activate
    def test_record_function_call(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}

        all_messages, call_info, formatted_prompt, response_info, session = self.__make_call(
            input_variables=input_variables,
            llm_response='Placeholder--Not Used'
        )

        response_info = ResponseInfo(
            is_complete=True,
            function_call_response=OpenAIFunctionCall(
                name='function_name',
                arguments='{"location": "San Francisco, CA", "format": "celsius"}')
        )
        record_response = self.freeplay_thin.recordings.create(
            RecordPayload(
                # Function call has empty 'content'
                all_messages=formatted_prompt.all_messages({'role': 'assistant'}),
                inputs=input_variables,
                session_info=session.session_info,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                response_info=response_info
            )
        )

        self.assertIsNotNone(record_response.completion_id)

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual(
            '[{"role": "system", "content": "System message"}, '
            '{"role": "assistant", "content": "How may I help you, Sparkles?"}, '
            '{"role": "user", "content": "Why isn\'t my door working"}]',
            recorded_body_dom['prompt_content']
        )
        # Empty body since we have a function call
        self.assertEqual('', recorded_body_dom['return_content'])
        self.assertEqual(
            {
                'arguments': '{"location": "San Francisco, CA", "format": "celsius"}', 'name': 'function_name'
            },
            recorded_body_dom['function_call_response']
        )

    @responses.activate
    def test_customer_feedback(self) -> None:
        completion_id = str(uuid4())

        self.__mock_customer_feedback_api(completion_id)

        self.freeplay_thin.customer_feedback.update(completion_id, {
            'some-feedback': 'it is ok!',
            'float': 1.2,
            'int': 1,
            'bool': True
        })

        customer_feedback_request = responses.calls[0].request
        recorded_body_dom = json.loads(customer_feedback_request.body)
        self.assertEqual('it is ok!', recorded_body_dom['some-feedback'])
        self.assertEqual(1.2, recorded_body_dom['float'])
        self.assertEqual(1, recorded_body_dom['int'])
        self.assertEqual(True, recorded_body_dom['bool'])

    @responses.activate
    def test_customer_feedback__unauthorized(self) -> None:
        completion_id = str(uuid4())
        responses.put(
            url=f'{self.api_base}/v1/completion_feedback/{completion_id}',
            status=401,
            content_type='application/json'
        )

        with self.assertRaisesRegex(FreeplayClientError, "Error updating customer feedback \\[401\\]"):
            self.freeplay_thin.customer_feedback.update(completion_id, {
                'some-feedback': 'it is ok!'
            })

    @responses.activate
    def test_get_template_prompt_then_populate(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}

        template_prompt = self.freeplay_thin.prompts.get(
            project_id=self.project_id,
            template_name=self.prompt_template_name,
            environment=self.tag
        )

        self.assertTrue("{{question}}" in template_prompt.messages[2]['content'])

        bound_prompt = template_prompt.bind(input_variables)

        self.assertFalse("{{question}}" in bound_prompt.messages[2]['content'])
        self.assertTrue(input_variables.get('name') in bound_prompt.messages[1]['content'])  # type: ignore
        self.assertTrue(input_variables.get('question') in bound_prompt.messages[2]['content'])  # type: ignore

        formatted_prompt = bound_prompt.format()

        self.assertEqual([
            {'content': 'How may I help you, Sparkles?', 'role': 'assistant'},
            {'content': "Why isn't my door working", 'role': 'user'}
        ], formatted_prompt.llm_prompt)
        self.assertEqual('System message', formatted_prompt.system_content)

        llm_response = 'This is the response from Anthropic'
        all_messages = formatted_prompt.all_messages({'role': 'assistant', 'content': ('%s' % llm_response)})

        self.assertTrue(input_variables.get('name') in formatted_prompt.llm_prompt[0]['content'])  # type: ignore
        self.assertTrue(input_variables.get('question') in formatted_prompt.llm_prompt[1]['content'])  # type: ignore
        self.assertTrue(llm_response in all_messages[3]['content'])

    @responses.activate
    def test_anthropic_system_prompt_formatting__multiple_system_messages(self) -> None:
        bound_prompt = BoundPrompt(self.anthropic_prompt_info, messages=[{
            'role': 'system',
            'content': 'System message 1',
        }, {
            'role': 'user',
            'content': 'User message 1',
        }, {
            'role': 'system',
            'content': 'System message 2',
        }, {
            'role': 'user',
            'content': 'User message 2',
        }])

        formatted_prompt = bound_prompt.format()

        self.assertEqual([
            {'content': 'User message 1', 'role': 'user'},
            {'content': "User message 2", 'role': 'user'}
        ], formatted_prompt.llm_prompt)
        self.assertEqual('System message 1', formatted_prompt.system_content)

    @responses.activate
    def test_anthropic_system_prompt_formatting__no_system_message(self) -> None:
        bound_prompt = BoundPrompt(self.anthropic_prompt_info, messages=[{
            'role': 'user',
            'content': 'User message 1',
        }, {
            'role': 'user',
            'content': 'User message 2',
        }])

        formatted_prompt = bound_prompt.format()

        self.assertEqual([
            {'content': 'User message 1', 'role': 'user'},
            {'content': 'User message 2', 'role': 'user'}
        ], formatted_prompt.llm_prompt)
        self.assertEqual(None, formatted_prompt.system_content)

    @responses.activate
    def test_create_test_run(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name)

        test_run = self.freeplay_thin.test_runs.create(self.project_id, testlist='good stuff')

        test_cases = test_run.get_test_cases()
        self.assertEqual(2, len(test_cases))
        self.assertTrue(all(test_case.output is None for test_case in test_cases))

    @responses.activate
    def test_create_test_run_with_outputs(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name)

        test_run = self.freeplay_thin.test_runs.create(
            self.project_id,
            testlist='good stuff',
            include_outputs=True,
        )

        test_cases = test_run.get_test_cases()
        self.assertEqual(2, len(test_cases))
        self.assertTrue(all(test_case.output is not None for test_case in test_cases))

    @responses.activate
    def test_auth_error(self) -> None:
        responses.get(
            url=f'{self.api_base}/v2/projects/{self.project_id}/prompt-templates/name/{self.prompt_template_name}',
            status=401,
            body=self.__get_templates_response()
        )

        freeplay_thin = Freeplay(
            freeplay_api_key="not-the-key",
            api_base=self.api_base,
        )

        with self.assertRaisesRegex(
                FreeplayClientError,
                f"Error getting prompt template my-prompt-anthropic in project {self.project_id} and environment test-tag \[401\]"
        ):
            freeplay_thin.prompts.get(
                project_id=self.project_id,
                template_name=self.prompt_template_name,
                environment=self.tag
            )

    @responses.activate
    def test_template_not_found(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name)
        with self.assertRaisesRegex(
                FreeplayClientError,
                f"Error getting prompt template invalid-template-id in project {self.project_id} and environment test-tag \[404\]"
        ):
            self.freeplay_thin.prompts.get(
                project_id=self.project_id,
                template_name="invalid-template-id",
                environment=self.tag
            )

    def test_filesystem_resolver_with_params(self) -> None:
        template_prompt = self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-params", "prod")

        expected = TemplatePrompt(
            prompt_info=PromptInfo(
                prompt_template_id='a8b91d92-e063-4c3e-bb44-0d570793856b',
                prompt_template_version_id='6fe8af2e-defe-41b8-bdf2-7b2ec23592f5',
                template_name='test-prompt-with-params',
                environment='prod',
                model_parameters={'max_tokens': 56, 'temperature': 0.1},  # type: ignore
                provider='openai',
                provider_info=None,
                model='gpt-3.5-turbo-1106',
                flavor_name='openai_chat'
            ),
            messages=[{'content': 'You are a support agent', 'role': 'system'},
                      {'content': 'How can I help you?', 'role': 'assistant'},
                      {'content': '{{question}}', 'role': 'user'}]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_without_params(self) -> None:
        template_prompt = self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-no-params", "prod")

        expected = TemplatePrompt(
            prompt_info=PromptInfo(
                prompt_template_id='5985c6bb-115c-4ca2-99bd-0ffeb917fca4',
                prompt_template_version_id='11e12956-d8d4-448a-af92-66b1dc2155e0',
                template_name='test-prompt-no-params',
                environment='prod',
                model_parameters={},  # type: ignore
                provider='openai',
                provider_info=None,
                model='gpt-3.5-turbo-1106',
                flavor_name='openai_chat'
            ),
            messages=[{'content': 'You are a support agent.', 'role': 'user'},
                      {'content': 'How may I help you?', 'role': 'assistant'},
                      {'content': '{{question}}', 'role': 'user'}]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_other_environment(self) -> None:
        template_prompt = self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-params", "qa")

        # Version ID is different
        expected = TemplatePrompt(
            prompt_info=PromptInfo(
                prompt_template_id='a8b91d92-e063-4c3e-bb44-0d570793856b',
                prompt_template_version_id='188545b0-afdb-4a1c-b99c-9519bb626da2',
                template_name='test-prompt-with-params',
                environment='qa',
                model_parameters={'max_tokens': 56, 'temperature': 0.1},  # type: ignore
                provider='openai',
                provider_info=None,
                model='gpt-3.5-turbo-1106',
                flavor_name='openai_chat'
            ),
            messages=[{'content': 'You are a support agent', 'role': 'system'},
                      {'content': 'How can I help you?', 'role': 'assistant'},
                      {'content': '{{question}}', 'role': 'user'}]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_with_params_v2(self) -> None:
        template_prompt = self.bundle_client_v2.prompts.get(self.bundle_project_id, "test-prompt-with-params", "prod")

        expected = TemplatePrompt(
            prompt_info=PromptInfo(
                prompt_template_id='a8b91d92-e063-4c3e-bb44-0d570793856b',
                prompt_template_version_id='6fe8af2e-defe-41b8-bdf2-7b2ec23592f5',
                template_name='test-prompt-with-params',
                environment='prod',
                model_parameters={'max_tokens': 56, 'temperature': 0.1},  # type: ignore
                provider='openai',
                provider_info={"anthropic_endpoint": "https://example2.com/anthropic"},
                model='gpt-3.5-turbo-1106',
                flavor_name='openai_chat'
            ),
            messages=[{'content': 'You are a support agent', 'role': 'system'},
                      {'content': 'How can I help you?', 'role': 'assistant'},
                      {'content': '{{question}}', 'role': 'user'}]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_without_params_v2(self) -> None:
        template_prompt = self.bundle_client_v2.prompts.get(self.bundle_project_id, "test-prompt-no-params", "prod")

        expected = TemplatePrompt(
            prompt_info=PromptInfo(
                prompt_template_id='a8b91d92-e063-4c3e-bb44-0d570793856b',
                prompt_template_version_id='6fe8af2e-defe-41b8-bdf2-7b2ec23592f5',
                template_name='test-prompt-no-params',
                environment='prod',
                model_parameters={},  # type: ignore
                provider='openai',
                provider_info={"anthropic_endpoint": "https://example2.com/anthropic"},
                model='gpt-3.5-turbo-1106',
                flavor_name='openai_chat'
            ),
            messages=[{'content': 'You are a support agent', 'role': 'system'},
                      {'content': 'How can I help you?', 'role': 'assistant'},
                      {'content': '{{question}}', 'role': 'user'}]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_freeplay_directory_doesnt_exist(self) -> None:
        with self.assertRaisesRegex(FreeplayConfigurationError, "Path for prompt templates is not a valid directory"):
            self.bundle_client = Freeplay(
                freeplay_api_key=self.freeplay_api_key,
                api_base=self.api_base,
                template_resolver=FilesystemTemplateResolver(Path(__file__).parent / "does_not_exist")
            )

    def test_prompt_file_does_not_exist(self) -> None:
        with self.assertRaisesRegex(
                FreeplayClientError,
                f"Could not find prompt with name not-a-prompt for project {self.bundle_project_id} in environment prod"
        ):
            self.bundle_client.prompts.get(self.bundle_project_id, "not-a-prompt", "prod")

    def test_freeplay_directory_is_file(self) -> None:
        with self.assertRaisesRegex(FreeplayConfigurationError, "Path for prompt templates is not a valid directory"):
            self.bundle_client = Freeplay(
                freeplay_api_key=self.freeplay_api_key,
                api_base=self.api_base,
                template_resolver=FilesystemTemplateResolver(Path(__file__))
            )

    def test_freeplay_directory_invalid_environment(self) -> None:
        with self.assertRaisesRegex(FreeplayConfigurationError, "Could not find prompt template directory for project"):
            self.bundle_client = Freeplay(
                freeplay_api_key=self.freeplay_api_key,
                api_base=self.api_base,
                template_resolver=FilesystemTemplateResolver(Path(__file__).parent / "test_files" / "prompts")
            )
            self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-params", "not_real_environment")

    def test_prompt_invalid_flavor(self) -> None:
        with self.assertRaisesRegex(
                FreeplayConfigurationError,
                'Configured flavor \\(not_a_flavor\\) not found in SDK. Please update your SDK version or configure '
                'a different model in the Freeplay UI.'
        ):
            self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-invalid-flavor", "prod")

    def test_prompt_no_model(self) -> None:
        with self.assertRaisesRegex(
                FreeplayConfigurationError,
                'Model must be configured in the Freeplay UI. Unable to fulfill request.'
        ):
            self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-no-model", "prod")

    def __mock_freeplay_apis(self, template_name: str, environment: str = 'latest') -> None:
        responses.get(
            url=f'{self.api_base}/v2/projects/{self.project_id}/prompt-templates/name/'
                f'{template_name}?environment={environment}',
            status=200,
            # Only match if query string on query string to ensure environment is passed.
            match=[matchers.query_param_matcher({'environment': environment})],
            body=self.__get_prompt_response(template_name)
        )
        responses.get(
            url=f'{self.api_base}/v2/projects/{self.project_id}/prompt-templates/name/invalid-template-id',
            status=404,
            body=json.dumps({'message': 'Could not find template with name "invalid-template-id"'})
        )
        self.__mock_test_run_api()
        self.__mock_record_api()

    def __mock_record_api(self) -> None:
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json',
            body=json.dumps({
                'completion_id': str(uuid4())
            })
        )

    def __mock_customer_feedback_api(self, completion_id: str) -> None:
        responses.put(
            url=f'{self.api_base}/v1/completion_feedback/{completion_id}',
            status=201,
            content_type='application/json'
        )

    def __mock_test_run_api(self) -> None:
        def request_callback(request: PreparedRequest) -> Tuple[int, Dict[str, str], str]:
            payload = json.loads(request.body) if request.body else None
            return (
                201,
                {},
                self.__create_test_run_response(
                    self.test_run_id,
                    payload['include_test_case_outputs'] if payload else None
                )
            )

        responses.add_callback(
            responses.POST, f'{self.api_base}/projects/{self.project_id}/test-runs-cases',
            callback=request_callback,
            content_type='application/json',
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

    def __get_prompt_response(self, template_name: str) -> str:
        return json.dumps({
            "content": [
                {
                    "role": "system",
                    "content": "System message"
                },
                {
                    "role": "assistant",
                    "content": "How may I help you, {{name}}?"
                },
                {
                    "role": "user",
                    "content": "{{question}}"
                }
            ],
            "format_version": 2,
            "metadata": {
                "flavor": "anthropic_chat",
                "model": "claude-2.1",
                "params": {
                    "max_tokens_to_sample": 50,
                    "temperature": 0.7
                },
                "provider": "anthropic",
                "provider_info": {
                    "anthropic_endpoint": "https://example.com/anthropic"
                }
            },
            "prompt_template_id": self.prompt_template_id_1,
            "prompt_template_name": template_name,
            "prompt_template_version_id": self.prompt_template_version_id
        })

    @staticmethod
    def __create_test_run_response(test_run_id: str, include_outputs: bool = False) -> str:
        return json.dumps({
            'test_run_id': test_run_id,
            'test_cases': [
                {
                    'id': str(uuid4()),
                    'variables': {'question': "Why isn't my internet working?"},
                    'output': 'It requested PTO this week.' if include_outputs else None,
                },
                {
                    'id': str(uuid4()),
                    'variables': {'question': "What does blue look like?"},
                    'output': 'It\'s a magical synergy between ocean and sky.' if include_outputs else None,
                }
            ]
        })

    def __make_call(
            self,
            input_variables: Dict[str, Any],
            llm_response: str
    ) -> Tuple[List[Dict[str, str]], CallInfo, FormattedPrompt, ResponseInfo, Session]:
        session = self.freeplay_thin.sessions.create(custom_metadata={'custom_metadata_field': 42})
        formatted_prompt = self.freeplay_thin.prompts.get_formatted(
            project_id=self.project_id,
            template_name=self.prompt_template_name,
            environment=self.tag,
            variables=input_variables
        )
        start = time.time()
        end = start + 5
        call_info = CallInfo(
            provider=formatted_prompt.prompt_info.provider,
            model=formatted_prompt.prompt_info.model,
            start_time=start,
            end_time=end,
            model_parameters=formatted_prompt.prompt_info.model_parameters,
            provider_info=formatted_prompt.prompt_info.provider_info,
        )
        all_messages = formatted_prompt.all_messages({'role': 'Assistant', 'content': ('%s' % llm_response)})
        response_info = ResponseInfo(is_complete=True)
        return all_messages, call_info, formatted_prompt, response_info, session
