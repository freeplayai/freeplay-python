import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from unittest import TestCase
from uuid import uuid4

import responses
from anthropic.types import TextBlock, ToolUseBlock
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from requests import PreparedRequest
from responses import matchers

from freeplay import CustomMetadata, Freeplay
from freeplay.errors import (
    FreeplayClientError,
    FreeplayClientWarning,
    FreeplayConfigurationError,
)
from freeplay.llm_parameters import LLMParameters
from freeplay.model import (
    MediaInputBase64,
    MediaInputMap,
    OpenAIFunctionCall,
    TestRunInfo,
)
from freeplay.resources.prompts import (
    BoundPrompt,
    FilesystemTemplateResolver,
    FormattedPrompt,
    PromptInfo,
    TemplatePrompt,
)
from freeplay.resources.recordings import (
    CallInfo,
    RecordPayload,
    RecordUpdatePayload,
    ResponseInfo,
    UsageTokens,
)
from freeplay.resources.sessions import Session, SessionInfo
from freeplay.resources.test_cases import DatasetTestCase
from freeplay.support import (
    HistoryTemplateMessage,
    TemplateChatMessage,
    TemplateMessage,
    ToolSchema,
)


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
        self.maxDiff = None

        self.freeplay_api_key = "freeplay_api_key"
        self.openai_api_key = "openai_api_key"
        self.api_base = "http://localhost:9091/api"
        self.project_id = str(uuid4())
        self.dataset_id = str(uuid4())
        self.project_version_id = str(uuid4())
        self.prompt_template_version_id = self.project_version_id
        self.prompt_template_id_1 = str(uuid4())
        self.prompt_template_name = "my-prompt-anthropic"
        self.session_id = str(uuid4())
        self.completion_id = str(uuid4())
        self.custom_metadata: CustomMetadata = {'custom_metadata_field': 42, 'true': False}
        self.session_info = SessionInfo(session_id=self.session_id, custom_metadata=self.custom_metadata)
        self.record_url = f'{self.api_base}/v2/projects/{self.project_id}/sessions/{self.session_id}/completions'
        self.record_update_url = f'{self.api_base}/v2/projects/{self.project_id}/completions/{self.completion_id}'
        self.tag = 'test-tag'
        self.test_run_id = str(uuid4())

        self.freeplay_thin = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base
        )

        self.legacy_bundle_client = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            template_resolver=FilesystemTemplateResolver(Path(__file__).parent / "test_files" / "legacy_prompt_formats")
        )
        self.bundle_client = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            template_resolver=FilesystemTemplateResolver(Path(__file__).parent / "test_files" / "prompts")
        )
        self.bundle_project_id = "475516c8-7be4-4d55-9388-535cef042981"
        self.openai_api_prompt_info = PromptInfo(
            prompt_template_id=str(uuid.uuid4()),
            prompt_template_version_id=str(uuid.uuid4()),
            template_name='template-name',
            environment='environment',
            model_parameters=LLMParameters({}),
            provider_info=None,
            provider='openai',
            model='model-name',
            flavor_name='openai_chat'
        )
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
        self.sagemaker_llama_3_prompt_info = PromptInfo(
            prompt_template_id=str(uuid.uuid4()),
            prompt_template_version_id=str(uuid.uuid4()),
            template_name='sagemaker-llama-3-template-name',
            environment='environment',
            model_parameters=LLMParameters({}),
            provider_info=None,
            provider='sagemaker',
            model='sagemaker-llama-3-model-name',
            flavor_name='llama_3_chat'
        )
        self.baseten_mistral_prompt_info = PromptInfo(
            prompt_template_id=str(uuid.uuid4()),
            prompt_template_version_id=str(uuid.uuid4()),
            template_name='baseten-mistral-template-name',
            environment='environment',
            model_parameters=LLMParameters({'max_tokens': 512, 'temperature': 0.12}),
            provider_info=None,
            provider='baseten',
            model='baseten-mistral-model-name',
            flavor_name='baseten_mistral_chat'
        )
        self.mistral_prompt_info = PromptInfo(
            prompt_template_id=str(uuid.uuid4()),
            prompt_template_version_id=str(uuid.uuid4()),
            template_name='mistral-template-name',
            environment='environment',
            model_parameters=LLMParameters({'max_tokens': 512, 'temperature': 0.12}),
            provider_info=None,
            provider='bedrock',
            model='mistral-model-name',
            flavor_name='mistral_chat'
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

        test_case_id = str(uuid4())

        self.freeplay_thin.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=all_messages,
                inputs=input_variables,
                session_info=self.session_info,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                tool_schema=formatted_prompt.tool_schema,
                response_info=response_info,
                eval_results={
                    'client_eval_field_bool': True,
                    'client_eval_field_float': 0.23
                },
                test_run_info=TestRunInfo(self.test_run_id, test_case_id)
            )
        )

        self.assertEqual(2, len(cast(List[Dict[str, str]], formatted_prompt.llm_prompt)))

        self.assertEqual({"anthropic_endpoint": "https://example.com/anthropic"},
                         formatted_prompt.prompt_info.provider_info)

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual(self.project_version_id, recorded_body_dom['prompt_info']['prompt_template_version_id'])
        self.assertIsNotNone(recorded_body_dom["call_info"]["start_time"])
        self.assertIsNotNone(recorded_body_dom["call_info"]["end_time"])
        self.assertEqual(recorded_body_dom["call_info"]["usage"]["prompt_tokens"], 123)
        self.assertEqual(recorded_body_dom["call_info"]["usage"]["completion_tokens"], 456)
        self.assertEqual(recorded_body_dom["call_info"]["api_style"], 'batch')
        self.assertEqual(self.tag, recorded_body_dom["prompt_info"]['environment'])
        self.assertEqual(
            all_messages,
            recorded_body_dom['messages']
        )
        self.assertEqual('anthropic', recorded_body_dom["call_info"]['provider'])
        self.assertEqual(0.7, recorded_body_dom["call_info"]['llm_parameters']['temperature'])
        self.assertEqual(50, recorded_body_dom["call_info"]['llm_parameters']['max_tokens_to_sample'])
        self.assertEqual(
            [
                {
                    'name': 'get_album_tracklist',
                    'description': 'Given an album name and genre, return a list of songs.',
                    'input_schema': {
                        'type': 'object',
                        'properties': {
                            'album_name': {'type': 'string',
                                           'description': 'Name of album from which to retrieve tracklist.'},
                            'genre': {'type': 'string', 'description': 'Album genre'}
                        }
                    }
                }
            ],
            recorded_body_dom['tool_schema']
        )
        self.assertEqual(
            {"anthropic_endpoint": "https://example.com/anthropic"},
            recorded_body_dom["call_info"]['provider_info']
        )

        # Custom metadata recording
        self.assertEqual({'custom_metadata_field': 42, 'true': False},
                         recorded_body_dom["session_info"]['custom_metadata'])

        self.assertEqual({"test_run_id": self.test_run_id, "test_case_id": test_case_id},
                         recorded_body_dom['test_run_info']
                         )

        # eval results recording
        self.assertEqual({
            'client_eval_field_bool': True,
            'client_eval_field_float': 0.23
        }, recorded_body_dom['eval_results'])

    @responses.activate
    def test_single_prompt_get_and_record_with_image(self) -> None:
        input_variables = {"query": "Describe this photograph"}
        one_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        media_inputs: MediaInputMap = {
            "subject-image": MediaInputBase64(
                type="base64",
                content_type="image/png",
                data=one_pixel_png
            )
        }
        llm_response = "It looks great to me"

        self.__mock_record_api()
        responses.get(
            url=f'{self.api_base}/v2/projects/{self.project_id}/prompt-templates/name/'
                f'template-with-image?environment=some-tag',
            status=200,
            match=[matchers.query_param_matcher({'environment': 'some-tag'})],
            body=json.dumps({
                "content": [
                    {"role": "system", "content": "Respond to the user's query"},
                    {"role": "user", "content": "{{query}}",
                     "media_slots": [{"type": "image", "placeholder_name": "subject-image"}]}
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
                "project_id": self.project_id,
                "prompt_template_id": str(uuid4()),
                "prompt_template_name": "template-with-image",
                "prompt_template_version_id": str(uuid4())
            }))

        _, call_info, formatted_prompt, response_info, session = self.__make_call(
            input_variables,
            llm_response,
            media_inputs,
            template_name="template-with-image",
            tag="some-tag"
        )

        test_case_id = str(uuid4())

        self.freeplay_thin.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=[*formatted_prompt.llm_prompt, {"role": "assistant", "content": llm_response}],
                inputs=input_variables,
                media_inputs=media_inputs,
                session_info=self.session_info,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                tool_schema=formatted_prompt.tool_schema,
                response_info=response_info,
                eval_results={
                    'client_eval_field_bool': True,
                    'client_eval_field_float': 0.23
                },
                test_run_info=TestRunInfo(self.test_run_id, test_case_id)
            )
        )

        self.assertEqual({"anthropic_endpoint": "https://example.com/anthropic"},
                         formatted_prompt.prompt_info.provider_info)

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)
        self.assertEqual(formatted_prompt.llm_prompt,
                         [{
                             "role": "user",
                             "content": [
                                 {"text": "Describe this photograph", "type": "text"},
                                 {"source": {
                                     "data": one_pixel_png,
                                     "media_type": "image/png",
                                     "type": "base64"},
                                     "type": "image"
                                 }
                             ],
                         }])
        self.assertEqual(recorded_body_dom["media_inputs"], {
            "subject-image": {
                "content_type": "image/png",
                "data": one_pixel_png,
                "type": "base64"
            }
        })

    @responses.activate
    def test_record_trace(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}
        llm_response = 'This is the response from the LLM'

        all_messages, call_info, formatted_prompt, response_info, session = self.__make_call(
            input_variables=input_variables,
            llm_response=llm_response
        )

        trace_metadata: CustomMetadata = {'bool_field': True, 'float_field': 1.2}
        trace_info = session.create_trace(
            input=input_variables['question'],
            agent_name='agent_name',
            custom_metadata=trace_metadata
        )
        completion_id = uuid4()
        self.freeplay_thin.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                completion_id=completion_id,
                all_messages=all_messages,
                inputs=input_variables,
                session_info=self.session_info,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                response_info=response_info,
                trace_info=trace_info,
            )
        )

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual(
            str(completion_id),
            recorded_body_dom['completion_id']
        )
        self.assertEqual(
            trace_info.trace_id,
            recorded_body_dom['trace_info']['trace_id']
        )

        self.__mock_record_trace_api(session.session_id, trace_info.trace_id)

        client_eval_results = {'bool_field': False, 'float_field': 0.23}
        trace_info.record_output(
            project_id=self.project_id,
            output=llm_response,
            eval_results=client_eval_results,
            test_run_info=TestRunInfo(self.test_run_id, "test_case_id")
        )
        record_trace_api_request = responses.calls[2].request
        recorded_trace_body_dom = json.loads(record_trace_api_request.body)
        self.assertEqual(
            input_variables['question'],
            recorded_trace_body_dom['input']
        )
        self.assertEqual(
            llm_response,
            recorded_trace_body_dom['output']
        )
        self.assertEqual(
            'agent_name',
            recorded_trace_body_dom['agent_name']
        )
        self.assertEqual(
            trace_metadata,
            recorded_trace_body_dom['custom_metadata']
        )
        self.assertEqual(
            client_eval_results,
            recorded_trace_body_dom['eval_results']
        )
        self.assertEqual(
            self.test_run_id,
            recorded_trace_body_dom['test_run_info']['test_run_id']
        )
        self.assertEqual(
            "test_case_id",
            recorded_trace_body_dom['test_run_info']['test_case_id']
        )

    @responses.activate
    def test_update_record(self) -> None:
        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}
        llm_response = 'This is the response from the LLM'

        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        all_messages, call_info, formatted_prompt, response_info, session = self.__make_call(
            input_variables,
            llm_response
        )

        # make initial record call
        record_response = self.freeplay_thin.recordings.create(
            RecordPayload(
                project_id=self.project_id,
                all_messages=formatted_prompt.llm_prompt,
                # mimic state where we don't yet have the LLM response like batch api
                inputs=input_variables,
                session_info=self.session_info,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                response_info=response_info,
            )
        )
        completion_id = record_response.completion_id

        # Make the record update call
        new_messages = [{'role': 'assistant', 'content': llm_response}]
        self.freeplay_thin.recordings.update(
            RecordUpdatePayload(
                project_id=self.project_id,
                completion_id=completion_id,
                new_messages=new_messages,
                eval_results={
                    'client_eval_field_bool': True,
                    'client_eval_field_float': 0.23
                }
            )
        )

        # eval results recording
        record_update_api_request = responses.calls[2].request
        recorded_body_dom = json.loads(record_update_api_request.body)
        self.assertEqual(new_messages, recorded_body_dom['new_messages'])
        self.assertEqual({
            'client_eval_field_bool': True,
            'client_eval_field_float': 0.23
        }, recorded_body_dom['eval_results'])

    @responses.activate
    def test_delete_session(self) -> None:
        self.__mock_session_delete_api(self.project_id, self.session_id)
        self.freeplay_thin.sessions.delete(self.project_id, self.session_id)

        record_api_request = responses.calls[0].request

        self.assertIsNone(record_api_request.body)

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
                project_id=self.project_id,
                # Function call has empty 'content'
                all_messages=formatted_prompt.all_messages({'role': 'assistant'}),
                inputs=input_variables,
                session_info=self.session_info,
                prompt_info=formatted_prompt.prompt_info,
                call_info=call_info,
                response_info=response_info
            )
        )

        self.assertIsNotNone(record_response.completion_id)

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual(
            formatted_prompt.all_messages({'role': 'assistant'}),
            recorded_body_dom['messages']
        )
        self.assertEqual(
            {
                'arguments': '{"location": "San Francisco, CA", "format": "celsius"}', 'name': 'function_name'
            },
            recorded_body_dom["response_info"]['function_call_response']
        )

    @responses.activate
    def test_customer_feedback(self) -> None:
        completion_id = str(uuid4())

        self.__mock_customer_feedback_api(completion_id)

        self.freeplay_thin.customer_feedback.update(self.project_id, completion_id, {
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
    def test_trace_feedback(self) -> None:
        trace_id = str(uuid4())

        self.__mock_trace_feedback_api(self.project_id, trace_id)

        self.freeplay_thin.customer_feedback.update_trace(self.project_id, trace_id,
                                                          {'is good': True, 'freeplay_feedback': 'positive', 'int': 1}
                                                          )

        trace_feedback_request = responses.calls[0].request
        recorded_body_dom = json.loads(trace_feedback_request.body)
        self.assertEqual(True, recorded_body_dom['is good'])
        self.assertEqual('positive', recorded_body_dom['freeplay_feedback'])
        self.assertEqual(1, recorded_body_dom['int'])

    @responses.activate
    def test_customer_feedback__unauthorized(self) -> None:
        completion_id = str(uuid4())
        responses.post(
            url=f'{self.api_base}/v2/projects/{self.project_id}/completion-feedback/id/{completion_id}',
            status=401,
            content_type='application/json'
        )

        with self.assertRaisesRegex(FreeplayClientError, "Error updating customer feedback \\[401\\]"):
            self.freeplay_thin.customer_feedback.update(self.project_id, completion_id, {
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

        self.assertTrue("{{question}}" in cast(TemplateChatMessage, template_prompt.messages[2]).content)

        bound_prompt = template_prompt.bind(input_variables)

        self.assertFalse("{{question}}" in bound_prompt.messages[2]['content'])
        self.assertTrue(input_variables.get('name') in bound_prompt.messages[1]['content'])
        self.assertTrue(input_variables.get('question') in bound_prompt.messages[2]['content'])

        formatted_prompt = bound_prompt.format()

        self.assertEqual([
            {'content': 'How may I help you, Sparkles?', 'role': 'assistant'},
            {'content': "Why isn't my door working", 'role': 'user'}
        ], formatted_prompt.llm_prompt)
        self.assertEqual('System message', formatted_prompt.system_content)

        llm_response = 'This is the response from Anthropic'
        all_messages = formatted_prompt.all_messages({'role': 'assistant', 'content': ('%s' % llm_response)})

        self.assertTrue(input_variables.get('name') in formatted_prompt.llm_prompt[0]['content'])
        self.assertTrue(input_variables.get('question') in formatted_prompt.llm_prompt[1]['content'])
        self.assertTrue(llm_response in all_messages[3]['content'])

    @responses.activate
    def test_all_messages(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}

        template_prompt = self.freeplay_thin.prompts.get(
            project_id=self.project_id,
            template_name=self.prompt_template_name,
            environment=self.tag
        )
        bound_prompt = template_prompt.bind(input_variables)
        formatted_prompt = bound_prompt.format()

        # Can append Anthropic messages from Pydantic models
        all_messages = formatted_prompt.all_messages(
            {
                'content': [
                    TextBlock(text='some text', type='text'),
                    ToolUseBlock(
                        id='1',
                        input='{}',
                        name='name',
                        type='tool_use')
                ],
                'role': 'assistant'
            })
        self.assertEqual(4, len(all_messages))
        json.dumps(all_messages)

        # Can append Anthropic messages as raw dict
        all_messages = formatted_prompt.all_messages(
            {
                'content': [{
                    'text': 'some text',
                    'type': 'text'
                }, {
                    'id': '1',
                    'input': '{}',
                    'name': 'name',
                    'type': 'tool_use'
                }],
                'role': 'assistant'
            })
        self.assertEqual(4, len(all_messages))
        json.dumps(all_messages)

        # Can append Freeplay standard format messages
        all_messages = formatted_prompt.all_messages(
            {
                'content': 'text',
                'role': 'assistant'
            })

        self.assertEqual(4, len(all_messages))
        json.dumps(all_messages)

        # Can append OpenAI Pydantic models
        all_messages = formatted_prompt.all_messages(
            ChatCompletionMessage(content='some-content', role='assistant', tool_calls=[
                ChatCompletionMessageToolCall(
                    id='2', function=Function(name='name', arguments='{}'), type='function')
            ]))
        self.assertEqual(4, len(all_messages))
        json.dumps(all_messages)

        # Can append OpenAI as dict models
        all_messages = formatted_prompt.all_messages(
            {'content': 'some-content', 'role': 'assistant', 'tool_calls': [
                {'id': '2', 'function': {'name': 'name', 'arguments': '{}'}, 'type': 'function'}
            ]})

        self.assertEqual(4, len(all_messages))
        json.dumps(all_messages)

    @responses.activate
    def test_get_template_prompt_with_tool_schema(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        template_prompt = self.freeplay_thin.prompts.get(
            project_id=self.project_id,
            template_name=self.prompt_template_name,
            environment=self.tag
        )
        self.assertEqual(template_prompt.tool_schema, [
            ToolSchema(name='get_album_tracklist', description='Given an album name and genre, return a list of songs.',
                       parameters={'type': 'object', 'properties': {'album_name': {'type': 'string',
                                                                                   'description': 'Name of album from which to retrieve tracklist.'},
                                                                    'genre': {'type': 'string',
                                                                              'description': 'Album genre'}}})
        ])

    def test_prompt_format__history_openai(self) -> None:
        messages: List[TemplateMessage] = [
            TemplateChatMessage(role='system', content='System message'),
            HistoryTemplateMessage(kind='history'),
            TemplateChatMessage(role='user', content='User message {{number}}')
        ]
        template_prompt = TemplatePrompt(
            self.openai_api_prompt_info,
            messages=messages
        )
        with self.assertWarnsRegex(FreeplayClientWarning,
                                   "History missing for prompt that expects history"):
            bound_prompt = template_prompt.bind({'number': 1}, history=[])
            formatted_prompt = bound_prompt.format()
            self.assertEqual(formatted_prompt.llm_prompt, [
                {'role': 'system', 'content': 'System message'},
                {'role': 'user', 'content': 'User message 1'}
            ])

        history = [{'role': 'user', 'content': 'User message 1'},
                   {'role': 'assistant', 'content': 'Assistant message 1'}]

        bound_prompt = template_prompt.bind({'number': 2}, history=history)
        formatted_prompt = bound_prompt.format()
        self.assertEqual(formatted_prompt.llm_prompt, [
            {'role': 'system', 'content': 'System message'},
            history[0],
            history[1],
            {'role': 'user', 'content': 'User message 2'}
        ])

    def test_prompt_format__history_anthropic(self) -> None:
        messages: List[TemplateMessage] = [
            TemplateChatMessage(role='system', content='System message'),
            HistoryTemplateMessage(kind='history'),
            TemplateChatMessage(role='user', content='User message {{number}}')
        ]
        template_prompt = TemplatePrompt(
            self.anthropic_prompt_info,
            messages=messages
        )
        with self.assertWarnsRegex(FreeplayClientWarning,
                                   "History missing for prompt that expects history"):
            bound_prompt = template_prompt.bind({'number': 1}, history=[])
            formatted_prompt = bound_prompt.format()
            self.assertEqual(formatted_prompt.llm_prompt, [
                {'role': 'user', 'content': 'User message 1'}
            ])
            self.assertEqual(formatted_prompt.system_content, 'System message')

        history = [{'role': 'user', 'content': 'User message 1'},
                   {'role': 'assistant', 'content': 'Assistant message 1'}]

        bound_prompt = template_prompt.bind({'number': 2}, history=history)
        formatted_prompt = bound_prompt.format()
        self.assertEqual(formatted_prompt.llm_prompt, [
            history[0],
            history[1],
            {'role': 'user', 'content': 'User message 2'}
        ])
        self.assertEqual(formatted_prompt.system_content, 'System message')

    def test_prompt_format__history_llama(self) -> None:
        messages: List[TemplateMessage] = [
            TemplateChatMessage(role='system', content='System message'),
            HistoryTemplateMessage(kind='history'),
            TemplateChatMessage(role='user', content='User message {{number}}')
        ]
        template_prompt = TemplatePrompt(
            self.sagemaker_llama_3_prompt_info,
            messages=messages
        )
        with self.assertWarnsRegex(FreeplayClientWarning,
                                   "History missing for prompt that expects history"):
            bound_prompt = template_prompt.bind({'number': 1}, history=[])
            formatted_prompt = bound_prompt.format()
            self.assertEqual(
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\nSystem message<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\nUser message 1<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>",
                formatted_prompt.llm_prompt_text
            )

        history = [{'role': 'user', 'content': 'User message 1'},
                   {'role': 'assistant', 'content': 'Assistant message 1'}]

        bound_prompt = template_prompt.bind({'number': 2}, history=history)
        formatted_prompt = bound_prompt.format()
        self.assertEqual(
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\nSystem message<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\nUser message 1<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\nAssistant message 1<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\nUser message 2<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>",
            formatted_prompt.llm_prompt_text
        )

    def test_prompt_format__bad_history(self) -> None:
        # send pass history to prompt that doesn't support it
        messages: List[TemplateMessage] = [
            TemplateChatMessage(role='system', content='System message'),
            TemplateChatMessage(role='user', content='User message {{number}}')
        ]
        template_prompt = TemplatePrompt(
            self.openai_api_prompt_info,
            messages
        )
        with self.assertRaisesRegex(FreeplayClientError,
                                    "History provided for prompt that does not expect history"):
            template_prompt.bind({'number': 1}, history=[{'role': 'user', 'content': 'User message 1'}])

        # send no history to prompt that expects it
        messages = [
            TemplateChatMessage(role='system', content='System message'),
            HistoryTemplateMessage(kind='history'),
            TemplateChatMessage(role='user', content='User message {{number}}')
        ]
        template_prompt = TemplatePrompt(
            self.openai_api_prompt_info,
            messages
        )
        with self.assertWarnsRegex(FreeplayClientWarning,
                                   "History missing for prompt that expects history"):
            formatted_prompt = template_prompt.bind({'number': 2}).format()
            self.assertEqual(
                [
                    {'role': 'system', 'content': 'System message'},
                    {'role': 'user', 'content': 'User message 2'}
                ],
                formatted_prompt.llm_prompt
            )

        history = [
            {'role': 'system', 'content': 'System message'},
            {'role': 'user', 'content': 'User message 1'},
            {'role': 'assistant', 'content': 'Assistant message 1'}
        ]
        # send a system message to history
        with self.assertWarnsRegex(FreeplayClientWarning, "System message found"):
            formatted_prompt = template_prompt.bind({'number': 2}, history=history).format()
            self.assertEqual(
                [
                    {'role': 'system', 'content': 'System message'},
                    history[1],
                    history[2],
                    {'role': 'user', 'content': 'User message 2'}
                ],
                formatted_prompt.llm_prompt
            )

    def test_prompt_format_with_tool_schema_anthropic(self) -> None:
        messages: List[TemplateMessage] = [
            TemplateChatMessage(role='system', content='System message'),
            TemplateChatMessage(role='user', content='User message {{number}}')
        ]
        tool_schema = [
            ToolSchema(name='tool_name', description='tool_description',
                       parameters={'name': 'param_name', 'description': 'param_description', 'type': 'string'})
        ]

        template_prompt = TemplatePrompt(
            self.anthropic_prompt_info,
            messages=messages,
            tool_schema=tool_schema
        )

        bound_prompt = template_prompt.bind({'number': 1})
        formatted_prompt = bound_prompt.format()
        self.assertEqual(formatted_prompt.tool_schema, [{
            'name': 'tool_name',
            'description': 'tool_description',
            'input_schema': {'name': 'param_name', 'description': 'param_description', 'type': 'string'}
        }])

    def test_prompt_format_with_tool_schema_openai(self) -> None:
        messages: List[TemplateMessage] = [
            TemplateChatMessage(role='system', content='System message'),
            TemplateChatMessage(role='user', content='User message {{number}}')
        ]
        tool_schema = [ToolSchema(name='tool_name', description='tool_description',
                                  parameters={'name': 'param_name', 'description': 'param_description',
                                              'type': 'string'})]

        template_prompt = TemplatePrompt(
            self.openai_api_prompt_info,
            messages=messages,
            tool_schema=tool_schema
        )

        bound_prompt = template_prompt.bind({'number': 1})
        formatted_prompt = bound_prompt.format()
        self.assertEqual(formatted_prompt.tool_schema, [{
            'function': asdict(tool_schema),
            'type': 'function'
        } for tool_schema in tool_schema])

    def test_prompt_format_with_tool_schema_gemini(self) -> None:
        # Import Vertex AI types for this test
        try:
            from vertexai.generative_models import Tool  # type: ignore[import-untyped]

            messages: List[TemplateMessage] = [
                TemplateChatMessage(role='system', content='System message'),
                TemplateChatMessage(role='user', content='User message {{number}}')
            ]
            tool_schema = [
                ToolSchema(name='get_weather', description='Get weather information',
                           parameters={
                               'type': 'object',
                               'properties': {
                                   'location': {'type': 'string', 'description': 'The city and state'},
                                   'unit': {'type': 'string', 'description': 'Temperature unit',
                                            'enum': ['celsius', 'fahrenheit']}
                               },
                               'required': ['location']
                           })
            ]

            gemini_prompt_info = PromptInfo(
                prompt_template_id=str(uuid.uuid4()),
                prompt_template_version_id=str(uuid.uuid4()),
                template_name='template-name',
                environment='environment',
                model_parameters=LLMParameters({}),
                provider_info=None,
                provider='vertex',
                model='gemini-pro',
                flavor_name='gemini_chat'
            )

            template_prompt = TemplatePrompt(
                gemini_prompt_info,
                messages=messages,
                tool_schema=tool_schema
            )

            bound_prompt = template_prompt.bind({'number': 1})
            formatted_prompt = bound_prompt.format()

            # Verify that the tool_schema is a list with one Tool object
            self.assertIsInstance(formatted_prompt.tool_schema, list)
            self.assertEqual(len(formatted_prompt.tool_schema), 1)
            self.assertIsInstance(formatted_prompt.tool_schema[0], Tool)

            # Verify the function declarations within the Tool
            # Access function declarations through _raw_tool (protobuf representation)
            function_declarations = formatted_prompt.tool_schema[0]._raw_tool.function_declarations
            self.assertEqual(len(function_declarations), 1)

            # Check the function declaration attributes
            fd = function_declarations[0]
            self.assertEqual(fd.name, 'get_weather')
            self.assertEqual(fd.description, 'Get weather information')

            # The parameters are stored as protobuf Schema object
            # We verify the structure exists rather than comparing deeply
            self.assertIsNotNone(fd.parameters)
            self.assertIn('location', str(fd.parameters))
        except ImportError:
            self.skipTest("Vertex AI SDK not installed")

    def test_anthropic_system_prompt_formatting__multiple_system_messages(self) -> None:
        bound_prompt = BoundPrompt(
            self.anthropic_prompt_info,
            messages=[
                {
                    'role': 'system',
                    'content': 'System message 1',
                },
                {
                    'role': 'user',
                    'content': 'User message 1',
                },
                {
                    'role': 'system',
                    'content': 'System message 2',
                },
                {
                    'role': 'user',
                    'content': 'User message 2',
                }
            ]
        )

        formatted_prompt = bound_prompt.format()

        self.assertEqual([
            {'content': 'User message 1', 'role': 'user'},
            {'content': "User message 2", 'role': 'user'}
        ], formatted_prompt.llm_prompt)
        self.assertEqual('System message 1', formatted_prompt.system_content)

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

    def test_sagemaker_llama_3_prompt_formatting__no_system_message(self) -> None:
        bound_prompt = BoundPrompt(self.sagemaker_llama_3_prompt_info, messages=[{
            'role': 'user',
            'content': 'User message 1',
        }, {
            'role': 'user',
            'content': 'User message 2',
        }])

        formatted_prompt = bound_prompt.format()

        self.assertEqual(
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\nUser message 1<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\nUser message 2<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>",
            formatted_prompt.llm_prompt_text
        )
        self.assertEqual(None, formatted_prompt.system_content)

    def test_sagemaker_llama_3_prompt_formatting__with_system_message(self) -> None:
        bound_prompt = BoundPrompt(
            self.sagemaker_llama_3_prompt_info,
            messages=[
                {'role': 'system', 'content': 'System message 1'},
                {'role': 'user', 'content': 'User message 1'},
                {'role': 'user', 'content': 'User message 2'}
            ]
        )

        formatted_prompt = bound_prompt.format()

        self.assertEqual(
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\nSystem message 1<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\nUser message 1<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\nUser message 2<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>",
            formatted_prompt.llm_prompt_text
        )
        self.assertEqual("System message 1", formatted_prompt.system_content)

    def test_baseten_mistral_system_prompt_formatting(self) -> None:
        bound_prompt = BoundPrompt(
            self.baseten_mistral_prompt_info,
            messages=[
                {'role': 'system', 'content': 'System message 1'},
                {'role': 'user', 'content': 'User message 1'},
                {'role': 'user', 'content': 'User message 2'}
            ]
        )

        formatted_prompt = bound_prompt.format()

        self.assertEqual([
            {'role': 'system', 'content': 'System message 1'},
            {'content': 'User message 1', 'role': 'user'},
            {'content': "User message 2", 'role': 'user'}
        ], formatted_prompt.llm_prompt)
        self.assertEqual('System message 1', formatted_prompt.system_content)

    def test_mistral_system_prompt_formatting(self) -> None:
        bound_prompt = BoundPrompt(
            self.mistral_prompt_info,
            messages=[
                {'role': 'system', 'content': 'System message 1'},
                {'role': 'user', 'content': 'User message 1'},
                {'role': 'user', 'content': 'User message 2'}
            ]
        )

        formatted_prompt = bound_prompt.format()

        self.assertEqual([
            {'role': 'system', 'content': 'System message 1'},
            {'content': 'User message 1', 'role': 'user'},
            {'content': "User message 2", 'role': 'user'}
        ], formatted_prompt.llm_prompt)
        self.assertEqual('System message 1', formatted_prompt.system_content)

    @responses.activate
    def test_create_test_run(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name)

        test_run = self.freeplay_thin.test_runs.create(
            self.project_id,
            testlist='good stuff',
            name='test-run-name',
            description='test-run-description',
            flavor_name='openai_chat'
        )

        test_cases = test_run.get_test_cases()
        self.assertEqual(2, len(test_cases))
        self.assertTrue(all(test_case.output is None for test_case in test_cases))
        self.assertIsNone(test_cases[0].history)
        self.assertIsNotNone(test_cases[1].history)

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
        self.assertEqual(1, len(test_cases[0].custom_metadata or {}))
        self.assertTrue(all(test_case.output is not None for test_case in test_cases))

    @responses.activate
    def test_create_test_run_with_trace_test_cases(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name)

        test_run = self.freeplay_thin.test_runs.create(
            self.project_id,
            testlist='agent_dataset',
            include_outputs=True,
        )

        trace_test_cases = test_run.get_trace_test_cases()
        self.assertEqual(1, len(trace_test_cases))
        self.assertEqual(1, len(trace_test_cases[0].custom_metadata or {}))
        self.assertTrue(all(test_case.input is not None for test_case in trace_test_cases))
        self.assertTrue(all(test_case.output is not None for test_case in trace_test_cases))

    @responses.activate
    def test_create_test_run_with_trace_test_cases_and_test_cases(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name)

        with self.assertRaises(ValueError):
            test_run = self.freeplay_thin.test_runs.create(
                self.project_id,
                testlist='agent_dataset',
                include_outputs=True,
            )

            test_run.get_test_cases()

    @responses.activate
    def test_get_test_run_results(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)

        test_run_results = self.freeplay_thin.test_runs.get(self.project_id, self.test_run_id)

        self.assertEqual('Test Test', test_run_results.name)
        self.assertEqual('This is a test!', test_run_results.description)
        self.assertEqual(self.test_run_id, test_run_results.test_run_id)
        self.assertIsNotNone(test_run_results.summary_statistics)
        self.assertIsNotNone(test_run_results.summary_statistics.auto_evaluation)
        self.assertIsNotNone(test_run_results.summary_statistics.human_evaluation)

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
        template_prompt = self.legacy_bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-params",
                                                                "prod")

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
            messages=[
                TemplateChatMessage(role='system', content='You are a support agent'),
                TemplateChatMessage(role='assistant', content='How can I help you?'),
                TemplateChatMessage(role='user', content='{{question}}'),
            ]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_without_params(self) -> None:
        template_prompt = self.legacy_bundle_client.prompts.get(self.bundle_project_id, "test-prompt-no-params", "prod")

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
            messages=[
                TemplateChatMessage(role='user', content='You are a support agent.'),
                TemplateChatMessage(role='assistant', content='How may I help you?'),
                TemplateChatMessage(role='user', content='{{question}}'),
            ]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_other_environment(self) -> None:
        template_prompt = self.legacy_bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-params", "qa")

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
            messages=[
                TemplateChatMessage(role='system', content='You are a support agent'),
                TemplateChatMessage(role='assistant', content='How can I help you?'),
                TemplateChatMessage(role='user', content='{{question}}'),
            ]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_with_params_v2(self) -> None:
        template_prompt = self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-params", "prod")

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
            messages=[
                TemplateChatMessage(role='system', content='You are a support agent'),
                TemplateChatMessage(role='assistant', content='How can I help you?'),
                TemplateChatMessage(role='user', content='{{question}}'),
            ]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_with_history_v2(self) -> None:
        template_prompt = self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-history", "prod")
        variables = {"question": "why?"}
        history = [
            {"role": "user", "content": "User message 1"},
            {"role": "assistant", "content": "Assistant message 1"}
        ]
        bound_prompt = template_prompt.bind(variables, history)

        self.assertEqual(
            [
                {'content': 'You are a support agent', 'role': 'system'},
                {'content': 'User message 1', 'role': 'user'},
                {'content': 'Assistant message 1', 'role': 'assistant'},
                {'content': 'why?', 'role': 'user'}
            ], bound_prompt.messages)

    def test_filesystem_resolver_with_tool_schema(self) -> None:
        template_prompt = self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-tool-schema", "prod")
        self.assertEqual(template_prompt.tool_schema, [ToolSchema(
            name="weather_of_location",
            description="Get weather of a location",
            parameters={
                "additionalProperties": False,
                "properties": {
                    "location": {
                        "description": "Location to get the weather for",
                        "type": "string"
                    }
                },
                "required": ["location"],
                "type": "object"
            }
        )])

    def test_filesystem_resolver_without_params_v2(self) -> None:
        template_prompt = self.bundle_client.prompts.get(self.bundle_project_id, "test-prompt-no-params", "prod")

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
            messages=[
                TemplateChatMessage(role='system', content='You are a support agent'),
                TemplateChatMessage(role='assistant', content='How can I help you?'),
                TemplateChatMessage(role='user', content='{{question}}'),
            ]
        )

        self.assertEqual(TemplatePromptMatcher(expected), template_prompt)

    def test_filesystem_resolver_get_version_id(self) -> None:
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
            messages=[
                TemplateChatMessage(role='system', content='You are a support agent'),
                TemplateChatMessage(role='assistant', content='How can I help you?'),
                TemplateChatMessage(role='user', content='{{question}}'),
            ]
        )
        template_prompt = self.bundle_client.prompts.get_by_version_id(
            self.bundle_project_id,
            expected.prompt_info.prompt_template_id,
            expected.prompt_info.prompt_template_version_id
        )
        self.assertEqual(expected.prompt_info.prompt_template_version_id,
                         template_prompt.prompt_info.prompt_template_version_id)
        self.assertEqual(expected.messages, template_prompt.messages)

    def test_filesystem_resolver_get_formatted_version_id(self) -> None:
        question = "Why isn't my door working"
        expected = FormattedPrompt(
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
                      {'content': f'{question}', 'role': 'user'}],
            formatted_prompt=[{'content': 'You are a support agent', 'role': 'system'},
                              {'content': 'How can I help you?', 'role': 'assistant'},
                              {'content': f'{question}', 'role': 'user'}]
        )
        formatted_prompt = self.bundle_client.prompts.get_formatted_by_version_id(
            self.bundle_project_id,
            expected.prompt_info.prompt_template_id,
            expected.prompt_info.prompt_template_version_id,
            {'question': question}
        )
        self.assertEqual(expected.prompt_info.prompt_template_version_id,
                         formatted_prompt.prompt_info.prompt_template_version_id)
        self.assertEqual(expected.llm_prompt, formatted_prompt.llm_prompt)

    def test_freeplay_directory_doesnt_exist(self) -> None:
        with self.assertRaisesRegex(FreeplayConfigurationError, "Path for prompt templates is not a valid directory"):
            self.legacy_bundle_client = Freeplay(
                freeplay_api_key=self.freeplay_api_key,
                api_base=self.api_base,
                template_resolver=FilesystemTemplateResolver(Path(__file__).parent / "does_not_exist")
            )

    def test_prompt_file_does_not_exist(self) -> None:
        with self.assertRaisesRegex(
                FreeplayClientError,
                f"Could not find prompt with name not-a-prompt for project {self.bundle_project_id} in environment prod"
        ):
            self.legacy_bundle_client.prompts.get(self.bundle_project_id, "not-a-prompt", "prod")

    def test_freeplay_directory_is_file(self) -> None:
        with self.assertRaisesRegex(FreeplayConfigurationError, "Path for prompt templates is not a valid directory"):
            self.legacy_bundle_client = Freeplay(
                freeplay_api_key=self.freeplay_api_key,
                api_base=self.api_base,
                template_resolver=FilesystemTemplateResolver(Path(__file__))
            )

    def test_freeplay_directory_invalid_environment(self) -> None:
        with self.assertRaisesRegex(FreeplayConfigurationError, "Could not find prompt template directory for project"):
            self.legacy_bundle_client = Freeplay(
                freeplay_api_key=self.freeplay_api_key,
                api_base=self.api_base,
                template_resolver=FilesystemTemplateResolver(
                    Path(__file__).parent / "test_files" / "legacy_prompt_formats")
            )
            self.legacy_bundle_client.prompts.get(self.bundle_project_id, "test-prompt-with-params",
                                                  "not_real_environment")

    def test_prompt_invalid_flavor(self) -> None:
        with self.assertRaisesRegex(
                FreeplayConfigurationError,
                'Configured flavor \\(not_a_flavor\\) not found in SDK. Please update your SDK version or configure '
                'a different model in the Freeplay UI.'
        ):
            self.legacy_bundle_client.prompts.get(self.bundle_project_id, "test-prompt-invalid-flavor", "prod")

    def test_prompt_no_model(self) -> None:
        with self.assertRaisesRegex(
                FreeplayConfigurationError,
                'Model must be configured in the Freeplay UI. Unable to fulfill request.'
        ):
            self.legacy_bundle_client.prompts.get(self.bundle_project_id, "test-prompt-no-model", "prod")

    @responses.activate
    def test_get_prompt_version_id(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)
        prompt_template = self.freeplay_thin.prompts.get_by_version_id(
            self.project_id,
            self.prompt_template_id_1,
            self.prompt_template_version_id
        )
        expected = json.loads(self.__get_prompt_response(self.prompt_template_name))
        self.assertEqual([
            TemplateChatMessage(role='system', content='System message'),
            TemplateChatMessage(role='assistant', content='How may I help you, {{name}}?'),
            TemplateChatMessage(role='user', content='{{question}}'),
        ], prompt_template.messages)
        self.assertEqual(expected.get("prompt_template_version_id"),
                         prompt_template.prompt_info.prompt_template_version_id)
        self.assertEqual(expected.get("prompt_template_name"), prompt_template.prompt_info.template_name)
        self.assertEqual(expected.get("metadata").get("provider"), prompt_template.prompt_info.provider)

    @responses.activate
    def test_get_formatted_version_id(self) -> None:
        self.__mock_freeplay_apis(self.prompt_template_name, self.tag)
        input_variables = {"name": "Sparkles", "question": "Why isn't my door working"}
        formatted_prompt = self.freeplay_thin.prompts.get_formatted_by_version_id(
            self.project_id,
            self.prompt_template_id_1,
            self.prompt_template_version_id,
            input_variables
        )
        expected = json.loads(self.__get_prompt_response(self.prompt_template_name))
        self.assertNotEqual(expected.get("content"), formatted_prompt.llm_prompt)
        self.assertEqual(
            expected.get("content")[-1].get("content").replace("{{question}}", "Why isn't my door working"),
            formatted_prompt.llm_prompt[-1].get("content"))
        self.assertEqual(expected.get("prompt_template_version_id"),
                         formatted_prompt.prompt_info.prompt_template_version_id)
        self.assertEqual(expected.get("prompt_template_name"), formatted_prompt.prompt_info.template_name)
        self.assertEqual(expected.get("metadata").get("provider"), formatted_prompt.prompt_info.provider)

    @responses.activate
    def test_insert_and_get_test_cases(self) -> None:
        self.__mock_test_case_insert_api()
        self.__mock_test_case_retrieval_api()

        test_case: DatasetTestCase = DatasetTestCase(history=None,
                                                     metadata={"key": "value"},
                                                     inputs={"question": "value 1"},
                                                     output="Prompt response 1")

        test_case_result = self.freeplay_thin.test_cases.create_many(self.project_id, self.dataset_id, [test_case])
        self.assertEqual(test_case_result.dataset_id, self.dataset_id)
        dataset_result = self.freeplay_thin.test_cases.get(self.project_id, self.dataset_id)
        self.assertEqual(len(dataset_result.test_cases), 1)
        self.assertEqual(dataset_result.test_cases[0].history, test_case.history)
        self.assertEqual(dataset_result.test_cases[0].inputs, test_case.inputs)
        self.assertEqual(dataset_result.test_cases[0].output, test_case.output)
        self.assertEqual(dataset_result.test_cases[0].metadata, test_case.metadata)

    @responses.activate
    def test_insert_test_cases_with_media_inputs(self) -> None:
        """Test creating test cases with media_inputs parameter."""
        one_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

        # Mock the API to capture the request body
        self.__mock_test_case_insert_api()

        # Create test cases with media inputs
        media_inputs: MediaInputMap = {
            "subject_image": MediaInputBase64(
                type="base64",
                content_type="image/png",
                data=one_pixel_png
            )
        }

        test_case: DatasetTestCase = DatasetTestCase(
            history=None,
            metadata={"test_type": "visual"},
            inputs={"query": "What do you see in this image?"},
            output="I see a test image.",
            media_inputs=media_inputs
        )

        test_case_result = self.freeplay_thin.test_cases.create_many(self.project_id, self.dataset_id, [test_case])
        self.assertEqual(test_case_result.dataset_id, self.dataset_id)

        # Verify the request payload includes properly serialized media_inputs
        create_test_cases_request = responses.calls[0].request
        request_body = json.loads(create_test_cases_request.body)
        self.assertIn("examples", request_body)

        examples = request_body["examples"]
        self.assertEqual(len(examples), 1)

        example = examples[0]
        self.assertEqual(example["inputs"], {"query": "What do you see in this image?"})
        self.assertEqual(example["output"], "I see a test image.")
        self.assertEqual(example["metadata"], {"test_type": "visual"})
        self.assertIsNone(example["history"])

        # Verify media_inputs serialization
        self.assertIn("media_inputs", example)
        media_inputs_json = example["media_inputs"]

        expected_media_input = {
            "type": "base64",
            "content_type": "image/png",
            "data": one_pixel_png
        }
        self.assertEqual(media_inputs_json["subject_image"], expected_media_input)

    @responses.activate
    def test_insert_test_cases_without_media_inputs(self) -> None:
        """Test creating test cases without media_inputs parameter (None case)."""
        self.__mock_test_case_insert_api()

        # Create test case without media inputs  
        test_case: DatasetTestCase = DatasetTestCase(
            history=None,
            metadata={"test_type": "text"},
            inputs={"question": "Simple question"},
            output="Simple answer"
        )

        test_case_result = self.freeplay_thin.test_cases.create_many(self.project_id, self.dataset_id, [test_case])
        self.assertEqual(test_case_result.dataset_id, self.dataset_id)

        # Verify the request payload
        create_test_cases_request = responses.calls[0].request
        request_body = json.loads(create_test_cases_request.body)
        examples = request_body["examples"]
        example = examples[0]

        self.assertEqual(example["inputs"], {"question": "Simple question"})
        self.assertEqual(example["output"], "Simple answer")
        self.assertEqual(example["metadata"], {"test_type": "text"})
        self.assertIsNone(example["history"])

        # Verify media_inputs is empty dict when None
        self.assertIn("media_inputs", example)
        self.assertEqual(example["media_inputs"], {})

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
        responses.get(
            url=f'{self.api_base}/v2/projects/{self.project_id}/prompt-templates/id/{self.prompt_template_id_1}/versions/{self.prompt_template_version_id}',
            status=200,
            body=self.__get_prompt_response(template_name)
        )
        self.__mock_test_run_api()
        self.__mock_test_run_retrieval_api()
        self.__mock_record_api()
        self.__mock_record_update_api()

    def __mock_record_api(self) -> None:
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json',
            body=json.dumps({
                'completion_id': self.completion_id
            })
        )

    def __mock_record_update_api(self) -> None:
        responses.post(
            url=self.record_update_url,
            status=201,
            content_type='application/json',
            body=json.dumps({
                'completion_id': self.completion_id
            })
        )

    def __mock_record_trace_api(self, session_id: str, trace_id: str) -> None:
        responses.post(
            url=f'{self.api_base}/v2/projects/{self.project_id}/sessions/{session_id}/traces/id/{trace_id}',
            status=201,
            content_type='application/json',
            body=json.dumps({})
        )

    def __mock_customer_feedback_api(self, completion_id: str) -> None:
        responses.post(
            url=f'{self.api_base}/v2/projects/{self.project_id}/completion-feedback/id/{completion_id}',
            status=201,
            content_type='application/json'
        )

    def __mock_trace_feedback_api(self, project_id: str, trace_id: str) -> None:
        responses.post(
            url=f'{self.api_base}/v2/projects/{project_id}/trace-feedback/id/{trace_id}',
            status=201,
            content_type='application/json'
        )

    def __mock_test_run_api(self) -> None:
        def request_callback(request: PreparedRequest) -> Tuple[int, Dict[str, str], str]:
            payload: Optional[Dict[str, Any]] = None  # Start with None

            loaded_data = json.loads(request.body) if request.body else None
            if isinstance(loaded_data, dict):
                payload = loaded_data

            # Ensure we have a dictionary for the .get() calls.
            # If payload is None (e.g. empty/invalid body, or JSON "null"), use an empty dict.
            final_payload_dict = payload if payload is not None else {}

            include_outputs = final_payload_dict.get('include_outputs', False)
            testlist_name = final_payload_dict.get('dataset_name')

            if testlist_name == 'agent_dataset':
                response_body = self.__create_agent_test_run_response(
                    self.test_run_id,
                    include_outputs
                )
            else:
                response_body = self.__create_test_run_response(
                    self.test_run_id,
                    include_outputs
                )

            return (
                201,
                {},
                response_body
            )

        responses.add_callback(
            responses.POST, f'{self.api_base}/v2/projects/{self.project_id}/test-runs',
            callback=request_callback,
            content_type='application/json',
        )

    def __mock_session_delete_api(self, project_id: str, session_id: str) -> str:
        url = f'{self.api_base}/v2/projects/{project_id}/sessions/{session_id}'
        responses.delete(
            url=url,
            status=201,
            content_type='application/json'
        )
        return url

    def __mock_test_case_retrieval_api(self) -> None:
        body = json.dumps([
            {'history': None, 'id': '995f77f8-eaa2-45f1-8c77-136183fc4a74', 'output': 'Prompt response 1',
             'values': {'question': 'value 1'}, 'metadata': {'key': "value"}}])
        responses.get(
            url=f'{self.api_base}/v2/projects/{self.project_id}/datasets/id/{self.dataset_id}/test-cases',
            status=200,
            body=body
        )

    def __mock_test_case_insert_api(self) -> None:
        examples: List[Dict[str, Any]] = [
            {'history': None, 'output': 'Prompt response 1', 'inputs': {'question': 'value 1'},
             'metadata': {'key': "value"}}]

        payload: Dict[str, Any] = {"examples": examples}
        json_payload = json.dumps(payload)
        responses.post(
            url=f'{self.api_base}/v2/projects/{self.project_id}/datasets/id/{self.dataset_id}/test-cases',
            status=201,
            content_type='application/json',
            body=json_payload
        )

    def __mock_test_run_retrieval_api(self) -> None:
        body = json.dumps({
            "description": "This is a test!",
            "id": self.test_run_id,
            "name": "Test Test",
            "summary_statistics": {
                "auto_evaluation": {
                    "Answer Accuracy": {
                        "4": 2,
                        "5": 6
                    },
                    "Context Relevance": {
                        "4": 4,
                        "5": 4
                    },
                    "Faithfulness": {
                        "no": 1,
                        "yes": 7
                    }
                },
                "human_evaluation": {
                    "Answer Accuracy": {
                        "4": 1,
                        "5": 7
                    },
                    "Context Relevance": {
                        "4": 5,
                        "5": 3
                    },
                    "Faithfulness": {
                        "no": 1,
                        "yes": 7
                    }
                }
            }
        })
        responses.get(
            url=f'{self.api_base}/v2/projects/{self.project_id}/test-runs/id/{self.test_run_id}',
            status=200,
            body=body
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
            "tool_schema": [
                {
                    "name": "get_album_tracklist",
                    "description": "Given an album name and genre, return a list of songs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "album_name": {"type": "string",
                                           "description": "Name of album from which to retrieve tracklist."},
                            "genre": {"type": "string", "description": "Album genre"}
                        }
                    }
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
            "project_id": self.project_id,
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
                    'test_case_id': str(uuid4()),
                    'variables': {'question': "Why isn't my internet working?"},
                    'output': 'It requested PTO this week.' if include_outputs else None,
                    'test_case_type': 'completion',
                    'custom_metadata': {
                        'key': 'value'
                    }
                },
                {
                    'test_case_id': str(uuid4()),
                    'variables': {'question': "What does blue look like?"},
                    'output': 'It\'s a magical synergy between ocean and sky.' if include_outputs else None,
                    'test_case_type': 'completion',
                    'history': [{
                        'role': 'user',
                        'content': [{
                            'type': 'text',
                            'text': 'some text'
                        }]
                    }]
                }
            ],
            'trace_test_cases': None,
        })

    @staticmethod
    def __create_agent_test_run_response(test_run_id: str, include_outputs: bool = False) -> str:
        return json.dumps({
            'test_run_id': test_run_id,
            'test_cases': None,
            'trace_test_cases': [
                {
                    'test_case_id': str(uuid4()),
                    'input': 'some input',
                    'output': 'some output' if include_outputs else None,
                    'custom_metadata': {
                        'key': 'value'
                    },
                    'test_case_type': 'trace'
                }
            ],
        })

    def __make_call(
            self,
            input_variables: Dict[str, Any],
            llm_response: str,
            media_inputs: Optional[MediaInputMap] = None,
            template_name: Optional[str] = None,
            tag: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], CallInfo, FormattedPrompt, ResponseInfo, Session]:
        session = self.freeplay_thin.sessions.create(custom_metadata={'custom_metadata_field': 42})
        formatted_prompt = self.freeplay_thin.prompts.get_formatted(
            project_id=self.project_id,
            template_name=template_name if template_name else self.prompt_template_name,
            environment=tag if tag else self.tag,
            variables=input_variables,
            media_inputs=media_inputs,
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
            usage=UsageTokens(
                prompt_tokens=123,
                completion_tokens=456,
            ),
            api_style='batch'
        )
        all_messages = formatted_prompt.all_messages({'role': 'assistant', 'content': ('%s' % llm_response)})
        response_info = ResponseInfo(is_complete=True)
        return all_messages, call_info, formatted_prompt, response_info, session
