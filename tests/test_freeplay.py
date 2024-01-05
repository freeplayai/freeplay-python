import json
from typing import Any, Collection, Dict, Generator, Iterable, Union
from unittest import TestCase
from unittest.mock import patch, MagicMock
from uuid import uuid4

import openai
import responses
import respx
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta

from freeplay.completions import ChatMessage, CompletionChunk  # type: ignore
from freeplay.errors import (  # type: ignore
    FreeplayClientError,
    FreeplayConfigurationError,
    LLMServerError
)
from freeplay.flavors import OpenAIChat  # type: ignore
from freeplay.freeplay import Freeplay  # type: ignore
from freeplay.provider_config import ProviderConfig, OpenAIConfig  # type: ignore
from freeplay.record import no_op_recorder  # type: ignore
from freeplay.support import JsonDom  # type: ignore


class TestFreeplay(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.freeplay_api_key = "freeplay_api_key"
        self.openai_api_key = "openai_api_key"
        self.api_base = "http://localhost:9091/api"
        self.openai_base = "http://localhost:666/v1"
        self.project_id = str(uuid4())
        self.session_id = str(uuid4())
        self.project_version_id = str(uuid4())
        self.prompt_template_version_id = self.project_version_id
        self.openai_chat_prompt_template_id_1 = str(uuid4())
        self.openai_chat_prompt_template_id_2 = str(uuid4())
        self.openai_chat_prompt_template_id_3 = str(uuid4())
        self.openai_chat_prompt_template_id_4 = str(uuid4())
        self.record_url = f'{self.api_base}/v1/record'
        self.tag = 'test-tag'

        self.open_ai_chat_flavor = OpenAIChat()
        self.provider_config = ProviderConfig(openai=OpenAIConfig(self.openai_api_key, self.openai_base))
        self.freeplay_chat_client = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config)

        responses.start()
        respx.start()

    def tearDown(self) -> None:
        responses.stop()
        responses.reset()
        respx.stop()
        respx.reset()

    def test_completion_parameters_with_overrides(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        client_model_params = {
            'max_tokens': 1,
            'temperature': 0.7,
            'stop': ['and'],
        }

        completion_model_params = {
            'max_tokens': 100,
            'frequency_penalty': -1.0,
        }

        freeplay = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config,
            flavor=self.open_ai_chat_flavor,
            record_processor=None,
            **client_model_params)

        freeplay.get_completion(project_id=str(self.project_id),
                                template_name="my-prompt",
                                variables={"question": "Why isn't my internet working?"},
                                tag=self.tag,
                                flavor=None,
                                metadata=None,
                                **completion_model_params)

        openai_request_dom = json.loads(respx.calls.last.request.content)
        # From model defaults
        self.assertEqual('gpt-3.5-turbo', openai_request_dom['model'])
        # From client config
        self.assertEqual(0.7, openai_request_dom['temperature'])
        self.assertEqual(['and'], openai_request_dom['stop'])
        # From completion config.
        # max_tokens in completion overrides max_tokens value from client.
        self.assertEqual(100, openai_request_dom['max_tokens'])
        self.assertEqual(-1.0, openai_request_dom['frequency_penalty'])

    def test_completion_params_from_freeplay_api(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        freeplay_flavorless_client = Freeplay(
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config)

        freeplay_flavorless_client.get_completion(project_id=str(self.project_id),
                                                  template_name="my-chat-prompt-in-freeplay-db",
                                                  variables={"name": "Sparkles"},
                                                  tag=self.tag)

        self.assertEqual(f'{self.openai_base}/chat/completions', respx.calls.last.request.url)
        openai_request_dom = json.loads(respx.calls.last.request.content)
        # From Prompt response object - request should go to chat_completions endpoint, and include UI configured data.
        self.assertEqual('gpt-2', openai_request_dom['model'])
        self.assertEqual(0.7, openai_request_dom['temperature'])
        self.assertEqual(5, openai_request_dom['max_tokens'])

        # Text completion
        freeplay_flavorless_client.get_completion(project_id=str(self.project_id),
                                                  template_name='my-prompt',
                                                  variables={'question': "Some question"},
                                                  tag=self.tag)

        self.assertEqual(f'{self.openai_base}/chat/completions', respx.calls.last.request.url)
        openai_request_2_dom = json.loads(respx.calls.last.request.content)
        self.assertEqual('gpt-3.5-turbo', openai_request_2_dom['model'])

    def test_openai_chat_mustache(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        self.freeplay_chat_client.get_completion(
            project_id=str(self.project_id),
            template_name="my-mustache-prompt",
            variables={"tolkien": "True"},
            tag=self.tag,
        )

        first_openai_call = json.loads(respx.calls[0].request.content)
        self.assertEqual(first_openai_call['messages'][0]['content'], 'Tell me about John Ronald Reuel Tolkien')

        record_api_request = responses.calls[1].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual(
            '[{"content": "Tell me about John Ronald Reuel Tolkien", "role": "system"}]',
            recorded_body_dom['prompt_content']
        )

    def test_openai_chat_single_prompt_session(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        completion = self.freeplay_chat_client.get_completion(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            variables={"name": "Sparkles"},
            tag=self.tag,
            metadata={
                "customer_id": 123456,
                "gitSHA": "d5afe656acfedad35ef75eb55c8a1b853fcd1cd2",
            })

        session_api_request = responses.calls[1].request
        session_body_dom = json.loads(session_api_request.body)
        self.assertEqual(123456, session_body_dom['custom_metadata']['customer_id'])
        self.assertEqual("d5afe656acfedad35ef75eb55c8a1b853fcd1cd2", session_body_dom['custom_metadata']['gitSHA'])

        record_api_request = responses.calls[-1].request
        recorded_body_dom = json.loads(record_api_request.body)
        self.assertEqual('Bearer freeplay_api_key', record_api_request.headers['Authorization'])
        self.assertEqual(True, recorded_body_dom['is_complete'])
        self.assertEqual(self.project_version_id, recorded_body_dom['project_version_id'])
        self.assertEqual(self.openai_chat_prompt_template_id_3, recorded_body_dom['prompt_template_id'])
        self.assertIsNotNone(recorded_body_dom['start_time'])
        self.assertIsNotNone(recorded_body_dom['end_time'])
        self.assertEqual(self.tag, recorded_body_dom['tag'])
        self.assertEqual('[{"content": "System message", "role": "system"}, {"content": "How may I '
                         'help you, Sparkles?", "role": "assistant"}]', recorded_body_dom['prompt_content'])
        self.assertEqual('I am your assistant',
                         recorded_body_dom['return_content'])
        self.assertEqual('openai_chat', recorded_body_dom['format_type'])
        self.assertEqual('openai', recorded_body_dom['provider'])
        self.assertEqual(0.7, recorded_body_dom['llm_parameters']['temperature'])
        self.assertEqual(5, recorded_body_dom['llm_parameters']['max_tokens'])

        # Completion checks
        self.assertEqual(self.record_url, record_api_request.url)
        self.assertEqual(True, completion.is_complete)
        self.assertEqual('I am your assistant', completion.content)

    def test_openai_chat_single_prompt_session_function_call(self) -> None:
        self.__mock_freeplay_apis()
        function_call_response = {
            'name': 'get_album_tracklist',
            'arguments': {
                'album_name': '24K Magic',
                'genre': 'Pop'
            }
        }
        respx.post(f'{self.openai_base}/chat/completions').respond(
            status_code=200,
            text=self.__openai_chat_response('soy tu asistente', function_call_response),
        )
        functions = [
            {
                "name": "get_album_tracklist",
                "description": "Given an album name and genre, return a list of songs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "album_name": {
                            "type": "string",
                            "description": "Name of album from which to retrieve tracklist."
                        },
                        "genre": {
                            "type": "string",
                            "description": "Album genre"
                        }
                    }
                }
            }
        ]
        completion = self.freeplay_chat_client.get_completion(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            variables={"name": "Sparkles"},
            tag=self.tag,
            functions=functions
        )

        if completion.openai_function_call is not None:
            self.assertEqual(completion.openai_function_call['name'], 'get_album_tracklist')
            self.assertEqual(completion.openai_function_call['arguments'], {'album_name': '24K Magic', 'genre': 'Pop'})
        else:
            self.assertIsNotNone(completion.openai_function_call)

    @patch("freeplay.freeplay.CallSupport.create_session_id")
    def test_openai_continuous_chat_session(self, mock_create_session_id: MagicMock) -> None:
        self.__mock_freeplay_and_openai_http_apis()

        mock_create_session_id.return_value = self.session_id

        chat_session, _ = self.freeplay_chat_client.start_chat(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            variables={"name": "Sparkles"},
            tag=self.tag
        )
        respx.post(f'{self.openai_base}/chat/completions').respond(
            status_code=200,
            text=self.__openai_chat_response('soy tu asistente'),
        )

        second_completion = chat_session.continue_chat(
            new_messages=[{'role': 'user', 'content': 'En espanol por favor'}],
            max_tokens=10)

        self.assertEqual("soy tu asistente", second_completion.content)
        self.assertEqual(True, second_completion.is_complete)

        restored_chat_session = self.freeplay_chat_client.restore_chat_session(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            session_id=chat_session.session_id,
            variables={"name": "Sparkles"},
            tag=self.tag,
            messages=chat_session.message_history,
        )

        restored_chat_session.continue_chat(
            new_messages=[{'role': 'user', 'content': 'Em portugues, por favor'}]
        )

        expected_final_message_history = [{"content": "System message", "role": "system"},
                                          {"content": "How may I help you, Sparkles?", "role": "assistant"},
                                          {"role": "assistant", "content": "I am your assistant"},
                                          {"role": "user", "content": "En espanol por favor"},
                                          {"role": "assistant", "content": "soy tu asistente"},
                                          {'role': 'user', 'content': 'Em portugues, por favor'},
                                          {"role": "assistant", "content": "soy tu asistente"}]

        self.assertEqual(expected_final_message_history, restored_chat_session.message_history)

        self.assertEqual(5, len(responses.calls))
        self.assertEqual(3, len(respx.calls))

        # Ensure all the appropriate data is passed to OpenAI
        first_openai_call = json.loads(respx.calls[0].request.content)
        self.assertEqual(2, len(first_openai_call['messages']))
        second_openai_call = json.loads(respx.calls[1].request.content)
        self.assertEqual(4, len(second_openai_call['messages']))
        self.assertEqual(10, second_openai_call['max_tokens'])

        # Ensure expected data is recorded to Freeplay
        first_record_call = self.__extract_request_body_to_dom(responses.calls[1])
        self.assertEqual(self.session_id, first_record_call['session_id'])
        self.assertEqual(json.dumps(expected_final_message_history[0:2]), first_record_call['prompt_content'])

        second_record_call = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual(self.session_id, second_record_call['session_id'])
        self.assertEqual(expected_final_message_history[0:4], json.loads(second_record_call['prompt_content']))
        self.assertEqual(second_record_call['return_content'], 'soy tu asistente')

        third_record_call = self.__extract_request_body_to_dom(responses.calls[4])
        self.assertEqual(self.session_id, third_record_call['session_id'])

    @patch("freeplay.freeplay.CallSupport.create_session_id")
    @patch("openai.resources.chat.Completions.create")
    def test_openai_continuous_chat_session_streaming(self, mock_completion_create: MagicMock, mock_create_session_id: MagicMock) -> None:
        self.__mock_freeplay_apis()

        mock_completion_create.return_value = self.__mock_openai_completion_stream_response("I am your assistant")
        mock_create_session_id.return_value = self.session_id

        chat_session, first_completion = self.freeplay_chat_client.start_chat_stream(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            variables={"name": "Sparkles"},
            tag=self.tag)

        self.__assert_generator_response(first_completion, 'I am your assistant', True)

        mock_completion_create.return_value = self.__mock_openai_completion_stream_response("soy tu asistente")

        new_message = ChatMessage(role="user", content="A message")
        second_completion = chat_session.continue_chat_stream(
            [new_message],
            max_tokens=10)

        self.__assert_generator_response(second_completion, "soy tu asistente", True)

        mock_completion_create.return_value = self.__mock_openai_completion_stream_response("I am your assistant")

        restored_chat_session = self.freeplay_chat_client.restore_chat_session(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            session_id=chat_session.session_id,
            variables={"name": "Sparkles"},
            messages=chat_session.message_history,
            tag=self.tag
        )

        third_completion = restored_chat_session.continue_chat_stream(
            new_messages=[new_message],
            max_tokens=10
        )
        self.__assert_generator_response(third_completion, "I am your assistant", True)

        self.assertEqual(5, len(responses.calls))
        expected_final_message_history = [{"content": "System message", "role": "system"},
                                          {"content": "How may I help you, Sparkles?", "role": "assistant"},
                                          {"role": "assistant", "content": "I am your assistant"},
                                          {"role": "user", "content": "A message"},
                                          {"role": "assistant", "content": "soy tu asistente"},
                                          {"role": "user", "content": "A message"}]

        # Ensure expected data is recorded to Freeplay
        first_record_call = self.__extract_request_body_to_dom(responses.calls[1])
        self.assertEqual(self.session_id, first_record_call['session_id'])
        self.assertEqual(json.dumps(expected_final_message_history[0:2]), first_record_call['prompt_content'])

        self.assertEqual('openai', first_record_call['provider'])
        self.assertEqual(0.7, first_record_call['llm_parameters']['temperature'])
        self.assertEqual(5, first_record_call['llm_parameters']['max_tokens'])

        second_record_call = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual(self.session_id, second_record_call['session_id'])
        self.assertEqual(expected_final_message_history[0:4], json.loads(second_record_call['prompt_content']))
        self.assertEqual(second_record_call['return_content'], 'soy tu asistente')

        # Third call is a restored session. Session ID must match
        third_record_call = self.__extract_request_body_to_dom(responses.calls[4])
        self.assertEqual(self.session_id, third_record_call['session_id'])
        self.assertEqual(json.dumps(expected_final_message_history[0:6]), third_record_call['prompt_content'])
        self.assertEqual(third_record_call['return_content'], 'I am your assistant')

    def test_openai_mixed_multi_prompt_session(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config,
            model="text-davinci-003",
            max_tokens=1)

        session = freeplay.create_session(self.project_id, self.tag)

        session.get_completion(template_name="my-prompt", variables={"question": "Why is nothing working?"})
        session.get_completion(
            template_name="my-chat-prompt",
            variables={"name": "charlie"},
            flavor=self.open_ai_chat_flavor,
            model="gpt-4")

        freeplay.restore_session(
            project_id=self.project_id,
            session_id=session.session_id,
            template_name="my-prompt",
            variables={"question": "Why is nothing working?"},
            tag=self.tag)

        self.assertEqual(len(responses.calls), 5)
        self.assertEqual(len(respx.calls), 3)
        first_openai_request_dom = json.loads(respx.calls[0].request.content)
        self.assertEqual('text-davinci-003', first_openai_request_dom['model'])
        self.assertEqual(1, first_openai_request_dom['max_tokens'])
        first_recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual(session.session_id, first_recorded_body_dom['session_id'])
        self.assertEqual(self.tag, first_recorded_body_dom['tag'])

        second_openai_request_dom = json.loads(respx.calls[1].request.content)
        self.assertEqual('gpt-4', second_openai_request_dom['model'])
        self.assertEqual(1, second_openai_request_dom['max_tokens'])
        second_recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual(session.session_id, second_recorded_body_dom['session_id'])

        # Restored session test
        third_recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[4])
        self.assertEqual(session.session_id, third_recorded_body_dom['session_id'])

    @patch("openai.resources.chat.Completions.create")
    def test_openai_chat_single_prompt_session_streaming(self, mock_completion_create: MagicMock) -> None:
        # The Responses mocking library does not support sending multiple streamed chunks -- we stream a single response
        # with a full completion response.
        self.__mock_freeplay_apis()

        mock_completion_create.return_value = self.__mock_openai_completion_stream_response("I am your assistant")

        completion_stream = self.freeplay_chat_client.get_completion_stream(project_id=str(self.project_id),
                                                                            template_name="my-chat-prompt",
                                                                            variables={"name": "Charlie"},
                                                                            tag=self.tag,
                                                                            max_tokens=100,
                                                                            frequency_penalty=-1.0)
        self.__assert_generator_response(completion_stream, 'I am your assistant', True)

        self.assertEqual(len(responses.calls), 2)

        recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[1])
        self.assertEqual("I am your assistant", recorded_body_dom['return_content'])

    def test_single_session_uses_default_tag_when_tag_omitted(self) -> None:
        responses.post(
            url=f'{self.api_base}/projects/{self.project_id}/sessions/tag/latest',
            status=201,
            body=self.__session_create_response(self.session_id)
        )
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/latest',
            status=200,
            body=self.__get_templates_response()
        )
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json'
        )
        self.__mock_openai_apis()

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config)

        freeplay.get_completion(project_id=str(self.project_id),
                                template_name="my-prompt",
                                variables={"question": "Why isn't my internet working?"})

    def test_multi_session_uses_default_tag_when_tag_omitted(self) -> None:
        self.__mock_openai_apis()
        responses.post(
            url=f'{self.api_base}/projects/{self.project_id}/sessions/tag/latest',
            status=201,
            body=self.__session_create_response(self.session_id)
        )
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/latest',
            status=200,
            body=self.__get_templates_response()
        )
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json'
        )

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config)

        session = freeplay.create_session(self.project_id)
        session.get_completion(template_name="my-prompt", variables={"question": "Why is nothing working?"})

    def test_requires_openai_and_freeplay_api_keys(self) -> None:
        with self.assertRaisesRegex(FreeplayConfigurationError, "OpenAI API key not set"):
            Freeplay(
                flavor=OpenAIChat(),
                freeplay_api_key=self.freeplay_api_key,
                api_base=self.api_base,
                provider_config=ProviderConfig(openai=OpenAIConfig("")))

        with self.assertRaisesRegex(FreeplayConfigurationError, "Freeplay API key not set"):
            Freeplay(
                flavor=OpenAIChat(),
                freeplay_api_key="",
                api_base=self.api_base,
                provider_config=self.provider_config)

    def test_proceeds_on_record_failure(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        responses.replace(responses.POST, url=self.record_url, body='{}', status=500)

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config)

        completion = freeplay.get_completion(project_id=str(self.project_id),
                                             template_name="my-prompt",
                                             variables={"question": "Why isn't my internet working?"},
                                             tag=self.tag)

        self.assertEqual("I am your assistant", completion.content)

    def test_auth_error(self) -> None:
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/{self.tag}',
            status=401
        )

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key="not-the-key",
            api_base=self.api_base,
            provider_config=self.provider_config)

        with self.assertRaisesRegex(FreeplayClientError, "Error getting prompt templates \\[401\\]"):
            freeplay.get_completion(project_id=self.project_id,
                                    template_name="my-chat-prompt",
                                    tag=self.tag,
                                    variables={})

    def test_template_not_found(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        with self.assertRaisesRegex(
                FreeplayConfigurationError,
                'Could not find template with name "invalid-template-id"'
        ):
            self.freeplay_chat_client.get_completion(project_id=self.project_id,
                                                     template_name="invalid-template-id",
                                                     tag=self.tag,
                                                     variables={})

    def test_internal_error_from_openai(self) -> None:
        self.__mock_freeplay_apis()
        respx.post(f'{self.openai_base}/chat/completions').respond(
            status_code=500,
        )

        with self.assertRaisesRegex(LLMServerError, "Unable to call OpenAI") as context:
            self.freeplay_chat_client.get_completion(project_id=str(self.project_id),
                                                     template_name="my-chat-prompt",
                                                     tag=self.tag,
                                                     variables={"name": "Sparkles"})
        self.assertIsInstance(context.exception.__cause__, openai.APIError)

    def test_no_record_session(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config,
            record_processor=no_op_recorder
        )

        completion = freeplay.get_completion(
            project_id=str(self.project_id),
            template_name="my-prompt",
            variables={"question": "Why isn't my internet working?"},
            tag=self.tag
        )
        # One less call
        self.assertEqual(len(responses.calls), 1)
        self.assertEqual(len(respx.calls), 1)
        self.assertTrue(all([self.record_url not in call.request.url for call in responses.calls]))

        # Completion checks
        self.assertEqual(True, completion.is_complete)
        self.assertEqual("I am your assistant", completion.content)

    def test_no_record_session_2(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config,
            model="text-davinci-003",
            max_tokens=1,
            record_processor=no_op_recorder
        )

        session = freeplay.create_session(self.project_id, self.tag)

        first_completion = session.get_completion(template_name="my-prompt",
                                                  variables={"question": "Why is nothing working?"})
        second_completion = session.get_completion(
            template_name="my-chat-prompt",
            variables={"name": "charlie"},
            flavor=self.open_ai_chat_flavor,
            model="gpt-4")

        self.assertEqual(True, first_completion.is_complete)
        self.assertEqual("I am your assistant", first_completion.content)

        self.assertEqual(True, second_completion.is_complete)
        self.assertEqual("I am your assistant", second_completion.content)

        # Two fewer calls
        self.assertEqual(len(responses.calls), 1)
        self.assertEqual(len(respx.calls), 2)
        self.assertTrue(all([self.record_url not in call.request.url for call in responses.calls]))

    def __mock_freeplay_apis(self) -> None:
        responses.post(
            url=f'{self.api_base}/projects/{self.project_id}/sessions/tag/{self.tag}',
            status=201,
            body=self.__session_create_response(self.session_id)
        )
        responses.get(
            url=f'{self.api_base}/projects/{self.project_id}/templates/all/{self.tag}',
            status=200,
            body=self.__get_templates_response()
        )
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json'
        )

    def __mock_openai_apis(self) -> None:
        respx.post(f'{self.openai_base}/chat/completions').respond(
            status_code=200,
            text=self.__openai_chat_response(),
        )

    def __mock_freeplay_and_openai_http_apis(self) -> None:
        self.__mock_freeplay_apis()
        self.__mock_openai_apis()

    @staticmethod
    def __mock_openai_completion_stream_response(content: str) -> Iterable[ChatCompletionChunk]:
        return iter([
            ChatCompletionChunk(
                id="1",
                choices=[Choice(
                    delta=ChoiceDelta(content=content),
                    finish_reason="stop",
                    index=1
                )],
                created=1703021942,
                model="gpt3.5-turbo-1106",
                object="chat.completion.chunk",
            )
        ])

    # For a given Freeplay SDK completion, there are usually 3 API calls (and response mocks)
    # 1. Retrieve prompts (Freeplay HTTP API)
    # 2. Call LLM Provider (OpenAI HTTP API)
    # 3. Record LLM request to Freeplay (Freeplay HTTP API)
    # We access them by index in order of the requests made by the SDK.
    @staticmethod
    def __extract_request_body_to_dom(responses_call: responses.Call) -> JsonDom:
        return json.loads(responses_call.request.body)

    def __assert_generator_response(self, generator: Generator[CompletionChunk, None, None], expected_text: str,
                                    expected_is_complete: bool) -> None:
        for chunk in generator:
            self.assertEqual(expected_text, chunk.text)
            self.assertEqual(expected_is_complete, chunk.is_complete)

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
                    'prompt_template_id': self.openai_chat_prompt_template_id_1,
                    'flavor_name': 'openai_chat',
                    'params': {
                        'temperature': 'override_this_value',
                    }
                }, {
                    'content': json.dumps([{
                        "role": "system",
                        "content": "Answer this question in a silly way: {{question}}"
                    }]),
                    'name': 'the-second-prompt',
                    'project_version_id': self.project_version_id,
                    'prompt_template_version_id': self.prompt_template_version_id,
                    'prompt_template_id': self.openai_chat_prompt_template_id_2
                },
                {
                    'content': json.dumps([{
                        "role": "system",
                        "content": "System message"
                    }, {
                        "role": "assistant",
                        "content": "How may I help you, {{name}}?"
                    }]),
                    'name': 'my-chat-prompt',
                    'project_version_id': self.project_version_id,
                    'prompt_template_version_id': self.prompt_template_version_id,
                    'prompt_template_id': self.openai_chat_prompt_template_id_3,
                    'flavor_name': 'openai_chat',
                    'params': {
                        'max_tokens': 5,
                        'temperature': 0.7
                    }
                },
                {
                    'content': json.dumps([{
                        "role": "system",
                        "content": "System message"
                    }]),
                    'name': 'my-chat-prompt-in-freeplay-db',
                    'project_version_id': self.project_version_id,
                    'prompt_template_version_id': self.prompt_template_version_id,
                    'prompt_template_id': self.openai_chat_prompt_template_id_3,
                    'flavor_name': 'openai_chat',
                    'params': {
                        'model': 'gpt-2',
                        'max_tokens': 5,
                        'temperature': 0.7
                    }
                },
                {
                    'content': json.dumps([{
                        "role": "system",
                        "content": "{{#tolkien}}Tell me about John Ronald Reuel Tolkien{{/tolkien}}{{#lewis}}"
                                   "Tell me about Clive Staples Lewis{{/lewis}}"
                    }]),
                    'name': 'my-mustache-prompt',
                    'project_version_id': self.project_version_id,
                    'prompt_template_version_id': self.prompt_template_version_id,
                    'prompt_template_id': self.openai_chat_prompt_template_id_4,
                    'flavor_name': 'openai_chat',
                    'params': {
                        'max_tokens': 5,
                        'temperature': 0.7
                    },
                }
            ]
        })

    @staticmethod
    def __openai_chat_response(
            content: str = 'I am your assistant',
            function_call_response: Union[Dict[str, Collection[str]], None] = None,
    ) -> str:
        if function_call_response is None:
            function_call_response = {}
        response: Dict[str, Any] = {
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
                        'content': content
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ]
        }
        if function_call_response:
            response["choices"][0]["message"]["function_call"] = function_call_response

        return json.dumps(response)

    @staticmethod
    def __openai_chat_streaming_response(content: str = 'I am your assistant') -> str:
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
                    'delta': {
                        'role': 'assistant',
                        'content': content
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ]
        })
