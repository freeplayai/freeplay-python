import json
from typing import Generator, Dict, Any, Collection
from unittest import TestCase
from uuid import uuid4

import openai
import responses

from freeplay.completions import CompletionChunk  # type: ignore
from freeplay.errors import (FreeplayClientError,  # type: ignore
                             FreeplayConfigurationError,
                             LLMServerError)
from freeplay.flavors import OpenAIChat  # type: ignore
from freeplay.freeplay import Freeplay, JsonDom  # type: ignore
from freeplay.provider_config import ProviderConfig, OpenAIConfig  # type: ignore
from freeplay.record import no_op_recorder  # type: ignore


class TestFreeplay(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.freeplay_api_key = "freeplay_api_key"
        self.openai_api_key = "openai_api_key"
        self.api_base = "http://localhost:9091/api"
        self.openai_base = "http://localhost:666"
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

    @responses.activate
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
                                **completion_model_params)

        openai_request_dom = self.__extract_request_body_to_dom(responses.calls[2])
        # From Freeplay API
        self.assertEqual(1234, openai_request_dom['secret_param_only_in_freeplay_ui'])
        # From model defaults
        self.assertEqual('gpt-3.5-turbo', openai_request_dom['model'])
        # From client config
        self.assertEqual(0.7, openai_request_dom['temperature'])
        self.assertEqual(['and'], openai_request_dom['stop'])
        # From completion config.
        # max_tokens in completion overrides max_tokens value from client.
        self.assertEqual(100, openai_request_dom['max_tokens'])
        self.assertEqual(-1.0, openai_request_dom['frequency_penalty'])

    @responses.activate
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

        self.assertEqual('http://localhost:666/chat/completions', responses.calls[2].request.url)
        openai_request_dom = self.__extract_request_body_to_dom(responses.calls[2])
        # From Prompt response object - request should go to chat_completions endpoint, and include UI configured data.
        self.assertEqual('gpt-2', openai_request_dom['model'])
        self.assertEqual(0.7, openai_request_dom['temperature'])
        self.assertEqual(5, openai_request_dom['max_tokens'])

        # Text completion
        freeplay_flavorless_client.get_completion(project_id=str(self.project_id),
                                                  template_name='my-prompt',
                                                  variables={'question': "Some question"},
                                                  tag=self.tag)

        self.assertEqual('http://localhost:666/chat/completions', responses.calls[6].request.url)
        openai_request_2_dom = self.__extract_request_body_to_dom(responses.calls[6])
        self.assertEqual('gpt-3.5-turbo', openai_request_2_dom['model'])
        self.assertEqual(1234, openai_request_2_dom['secret_param_only_in_freeplay_ui'])

    @responses.activate
    def test_openai_chat_mustache(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        completion = self.freeplay_chat_client.get_completion(
            project_id=str(self.project_id),
            template_name="my-mustache-prompt",
            variables={"tolkien": True},
            tag=self.tag, )

        record_api_request = responses.calls[2].request
        recorded_body_dom = json.loads(record_api_request.body)
        self.assertEqual(recorded_body_dom['messages'][0]['content'], 'Tell me about John Ronald Reuel Tolkien')

        record_api_request = responses.calls[3].request
        recorded_body_dom = json.loads(record_api_request.body)

        self.assertEqual('[{"content": "Tell me about John Ronald Reuel Tolkien", "role": "system"}]',
                         recorded_body_dom['prompt_content'])

    @responses.activate
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

        record_api_request = responses.calls[0].request
        recorded_body_dom = json.loads(record_api_request.body)
        self.assertEqual(123456, recorded_body_dom['metadata']['customer_id'])
        self.assertEqual("d5afe656acfedad35ef75eb55c8a1b853fcd1cd2", recorded_body_dom['metadata']['gitSHA'])

        record_api_request = responses.calls[3].request
        recorded_body_dom = json.loads(record_api_request.body)
        self.assertEqual('Bearer freeplay_api_key', record_api_request.headers['Authorization'])
        self.assertEqual(True, recorded_body_dom['is_complete'])
        self.assertEqual(self.project_version_id, recorded_body_dom['project_version_id'])
        self.assertEqual(self.openai_chat_prompt_template_id_3, recorded_body_dom['prompt_template_id'])
        self.assertIsNotNone(recorded_body_dom['start_time'])
        self.assertIsNotNone(recorded_body_dom['end_time'])
        self.assertEqual(self.tag, recorded_body_dom['tag'])
        self.assertEqual('[{"content": "System message", "role": "system"}, {"content": "How may I '
                         'help you, Sparkles?", "role": "Assistant"}]', recorded_body_dom['prompt_content'])
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

    @responses.activate
    def test_openai_chat_single_prompt_session_function_call(self) -> None:
        self.__mock_freeplay_apis()
        function_call_response = {
            'name': 'get_album_tracklist',
            'arguments': {
                'album_name': '24K Magic',
                'genre': 'Pop'
            }
        }
        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=200,
            body=self.__openai_chat_response('soy tu asistente', function_call_response),
            content_type='application/json'
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

        self.assertEqual(completion.openai_function_call['name'], 'get_album_tracklist')
        self.assertEqual(completion.openai_function_call['arguments'], {'album_name': '24K Magic', 'genre': 'Pop'})

    @responses.activate
    def test_openai_continuous_chat_session(self) -> None:
        self.__mock_freeplay_and_openai_http_apis()
        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=200,
            body=self.__openai_chat_response('soy tu asistente'),
            content_type='application/json'
        )

        chat_session, _ = self.freeplay_chat_client.start_chat(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            variables={"name": "Sparkles"},
            tag=self.tag
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
                                          {"content": "How may I help you, Sparkles?", "role": "Assistant"},
                                          {"role": "assistant", "content": "I am your assistant"},
                                          {"role": "user", "content": "En espanol por favor"},
                                          {"role": "assistant", "content": "soy tu asistente"},
                                          {'role': 'user', 'content': 'Em portugues, por favor'},
                                          {"role": "assistant", "content": "soy tu asistente"}]

        self.assertEqual(expected_final_message_history, restored_chat_session.message_history)

        self.assertEqual(9, len(responses.calls))

        # Ensure all the appropriate data is passed to OpenAI
        first_openai_call = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual(2, len(first_openai_call['messages']))
        second_openai_call = self.__extract_request_body_to_dom(responses.calls[4])
        self.assertEqual(4, len(second_openai_call['messages']))
        self.assertEqual(10, second_openai_call['max_tokens'])

        # Ensure expected data is recorded to Freeplay
        first_record_call = self.__extract_request_body_to_dom(responses.calls[3])
        self.assertEqual(self.session_id, first_record_call['session_id'])
        self.assertEqual(json.dumps(expected_final_message_history[0:2]), first_record_call['prompt_content'])
        second_record_call = self.__extract_request_body_to_dom(responses.calls[5])
        self.assertEqual(self.session_id, second_record_call['session_id'])
        self.assertEqual(expected_final_message_history[0:4], json.loads(second_record_call['prompt_content']))
        self.assertEqual(second_record_call['return_content'], 'soy tu asistente')

        third_record_call = self.__extract_request_body_to_dom(responses.calls[8])
        self.assertEqual(self.session_id, third_record_call['session_id'])

    @responses.activate
    def test_openai_continuous_chat_session_streaming(self) -> None:
        self.__mock_freeplay_apis()
        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=200,
            body=f"data: {self.__openai_chat_streaming_response('I am your assistant')}",
            auto_calculate_content_length=True,
            content_type='text/event-stream'
        )
        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=200,
            body=f"data: {self.__openai_chat_streaming_response('soy tu asistente')}",
            auto_calculate_content_length=True,
            content_type='text/event-stream'
        )

        chat_session, first_completion = self.freeplay_chat_client.start_chat_stream(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            variables={"name": "Sparkles"},
            tag=self.tag)

        self.__assert_generator_response(first_completion, 'I am your assistant', True)

        new_message = {"role": "user", "content": "A message"}
        second_completion = chat_session.continue_chat_stream(
            [new_message],
            max_tokens=10)

        self.__assert_generator_response(second_completion, "soy tu asistente", True)

        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=200,
            body=f"data: {self.__openai_chat_streaming_response('I am your assistant')}",
            auto_calculate_content_length=True,
            content_type='text/event-stream'
        )

        restored_chat_session = self.freeplay_chat_client.restore_chat_session(
            project_id=str(self.project_id),
            template_name="my-chat-prompt",
            session_id=chat_session.session_id,
            variables={"name": "Sparkles"},
            messages=chat_session.message_history,
            tag=self.tag
        )

        third_completion = restored_chat_session.continue_chat_stream(
            message_history=chat_session.message_history,
            new_messages=[new_message],
            max_tokens=10
        )
        self.__assert_generator_response(third_completion, "I am your assistant", True)

        self.assertEqual(9, len(responses.calls))
        expected_final_message_history = [{"content": "System message", "role": "system"},
                                          {"content": "How may I help you, Sparkles?", "role": "Assistant"},
                                          {"role": "assistant", "content": "I am your assistant"},
                                          {"role": "user", "content": "A message"},
                                          {"role": "assistant", "content": "soy tu asistente"},
                                          {"role": "user", "content": "A message"}]

        # Ensure all the appropriate data is passed to OpenAI
        first_openai_call = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual(2, len(first_openai_call['messages']))
        second_openai_call = self.__extract_request_body_to_dom(responses.calls[4])

        self.assertEqual(4, len(second_openai_call['messages']))
        self.assertEqual(10, second_openai_call['max_tokens'])

        # Ensure expected data is recorded to Freeplay
        first_record_call = self.__extract_request_body_to_dom(responses.calls[3])
        self.assertEqual(self.session_id, first_record_call['session_id'])
        self.assertEqual(json.dumps(expected_final_message_history[0:2]), first_record_call['prompt_content'])

        self.assertEqual('openai', first_record_call['provider'])
        self.assertEqual(0.7, first_record_call['llm_parameters']['temperature'])
        self.assertEqual(5, first_record_call['llm_parameters']['max_tokens'])

        second_record_call = self.__extract_request_body_to_dom(responses.calls[5])
        self.assertEqual(self.session_id, second_record_call['session_id'])
        self.assertEqual(expected_final_message_history[0:4], json.loads(second_record_call['prompt_content']))
        self.assertEqual(second_record_call['return_content'], 'soy tu asistente')

        # Third call is a restored session. Session ID must match
        third_record_call = self.__extract_request_body_to_dom(responses.calls[8])
        self.assertEqual(self.session_id, third_record_call['session_id'])
        self.assertEqual(json.dumps(expected_final_message_history[0:6]), third_record_call['prompt_content'])
        self.assertEqual(third_record_call['return_content'], 'I am your assistant')

    @responses.activate
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

        self.assertEqual(len(responses.calls), 9)
        first_openai_request_dom = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual('text-davinci-003', first_openai_request_dom['model'])
        self.assertEqual(1, first_openai_request_dom['max_tokens'])
        first_recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[3])
        self.assertEqual(session.session_id, first_recorded_body_dom['session_id'])
        self.assertEqual(self.tag, first_recorded_body_dom['tag'])

        second_openai_request_dom = self.__extract_request_body_to_dom(responses.calls[4])
        self.assertEqual('gpt-4', second_openai_request_dom['model'])
        self.assertEqual(1, second_openai_request_dom['max_tokens'])
        second_recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[5])
        self.assertEqual(session.session_id, second_recorded_body_dom['session_id'])

        # Restored session test
        third_recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[8])
        self.assertEqual(session.session_id, third_recorded_body_dom['session_id'])

    @responses.activate
    def test_openai_chat_single_prompt_session_streaming(self) -> None:
        # The Responses mocking library does not support sending multiple streamed chunks -- we stream a single response
        # with a full completion response.
        self.__mock_freeplay_apis()
        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=200,
            body=f"data: {self.__openai_chat_streaming_response()}",
            auto_calculate_content_length=True,
            content_type='text/event-stream'
        )

        completion_stream = self.freeplay_chat_client.get_completion_stream(project_id=str(self.project_id),
                                                                            template_name="my-chat-prompt",
                                                                            variables={"name": "Charlie"},
                                                                            tag=self.tag,
                                                                            max_tokens=100,
                                                                            frequency_penalty=-1.0)
        self.__assert_generator_response(completion_stream, 'I am your assistant', True)

        self.assertEqual(len(responses.calls), 4)

        openai_request_dom = self.__extract_request_body_to_dom(responses.calls[2])
        self.assertEqual('gpt-3.5-turbo', openai_request_dom['model'])
        self.assertEqual(100, openai_request_dom['max_tokens'])
        self.assertEqual(-1.0, openai_request_dom['frequency_penalty'])

        recorded_body_dom = self.__extract_request_body_to_dom(responses.calls[3])
        self.assertEqual("I am your assistant", recorded_body_dom['return_content'])

    @responses.activate
    def test_single_session_uses_default_tag_when_tag_omitted(self) -> None:
        session_url = f'{self.api_base}/projects/{self.project_id}/sessions/tag/latest'
        responses.post(
            url=session_url,
            status=201,
            body=self.__session_create_response(self.session_id)
        )
        templates_url = f'{self.api_base}/projects/{self.project_id}/templates/all/latest'
        responses.get(
            url=templates_url,
            status=200,
            body=self.__get_templates_response()
        )
        self.__mock_openai_apis()
        self.__mock_record_api()

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key=self.freeplay_api_key,
            api_base=self.api_base,
            provider_config=self.provider_config)

        freeplay.get_completion(project_id=str(self.project_id),
                                template_name="my-prompt",
                                variables={"question": "Why isn't my internet working?"})

        responses.assert_call_count(session_url, 1)
        responses.assert_call_count(templates_url, 1)

    @responses.activate
    def test_multi_session_uses_default_tag_when_tag_omitted(self) -> None:
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
        self.__mock_openai_apis()
        self.__mock_record_api()

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

    @responses.activate
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

    @responses.activate
    def test_auth_error(self) -> None:
        responses.post(
            url=f'{self.api_base}/projects/{self.project_id}/sessions/tag/{self.tag}',
            status=401
        )

        freeplay = Freeplay(
            flavor=self.open_ai_chat_flavor,
            freeplay_api_key="not-the-key",
            api_base=self.api_base,
            provider_config=self.provider_config)

        with self.assertRaisesRegex(FreeplayClientError, "Error while creating a session. \\[401\\]"):
            freeplay.get_completion(project_id=self.project_id,
                                    template_name="my-chat-prompt",
                                    tag=self.tag,
                                    variables={})

    @responses.activate
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

    @responses.activate
    def test_internal_error_from_openai(self) -> None:
        self.__mock_freeplay_apis()
        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=500,
            content_type='application/json'
        )

        with self.assertRaisesRegex(LLMServerError, "Unable to call OpenAI") as context:
            self.freeplay_chat_client.get_completion(project_id=str(self.project_id),
                                                     template_name="my-chat-prompt",
                                                     tag=self.tag,
                                                     variables={"name": "Sparkles"})
        self.assertIsInstance(context.exception.__cause__, openai.error.APIError)

    @responses.activate
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
        self.assertEqual(len(responses.calls), 3)
        self.assertTrue(all([self.record_url not in call.request.url for call in responses.calls]))

        # Completion checks
        self.assertEqual(True, completion.is_complete)
        self.assertEqual("I am your assistant", completion.content)

    @responses.activate
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

        # Two less calls
        self.assertEqual(len(responses.calls), 4)
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
        self.__mock_record_api()

    def __mock_record_api(self) -> None:
        responses.post(
            url=self.record_url,
            status=201,
            content_type='application/json'
        )

    def __mock_openai_apis(self) -> None:
        responses.post(
            url=f'{self.openai_base}/chat/completions',
            status=200,
            body=self.__openai_chat_response(),
            content_type='application/json'
        )

    def __mock_freeplay_and_openai_http_apis(self) -> None:
        self.__mock_freeplay_apis()
        self.__mock_openai_apis()

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
                        'secret_param_only_in_freeplay_ui': 1234
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
                        "role": "Assistant",
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
                        "content": "{{#tolkien}}Tell me about John Ronald Reuel Tolkien{{/tolkien}}{{#lewis}}Tell me about Clive Staples Lewis{{/lewis}}"
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
            function_call_response: dict[str, Collection[str]] = {}
    ) -> str:
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
