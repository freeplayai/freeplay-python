import dataclasses
import json
import os
import os.path
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase
from uuid import uuid4

import responses
from click.testing import CliRunner

from freeplay.completions import PromptTemplateWithMetadata, PromptTemplates  # type: ignore
from freeplay.freeplay_cli import cli  # type: ignore


class TestFreeplayCLI(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.project_id = str(uuid4())
        self.prompt_template_version_id_1 = str(uuid4())
        self.prompt_template_version_id_2 = str(uuid4())
        self.prompt_template_id_1 = str(uuid4())
        self.prompt_template_id_2 = str(uuid4())
        self.environment = "prod"
        os.environ["FREEPLAY_API_KEY"] = "freeplay_api_key"
        os.environ["FREEPLAY_SUBDOMAIN"] = "doesnotexist"
        api_base = "http://localhost:9091"
        self.api_url = "%s/api" % api_base
        os.environ["FREEPLAY_API_URL"] = api_base

        self.system_message = "You're a tech support agent"
        self.assistant_message = "How may I help you?"
        self.user_message = "Answer this question: {{question}}"

    @responses.activate
    def test_download_succeeds(self) -> None:
        self.__mock_freeplay_api_success()

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outDir:
            arguments = [
                'download',
                '--project-id=%s' % self.project_id,
                '--environment=%s' % self.environment,
                '--output-dir=%s' % outDir
            ]
            # noinspection PyTypeChecker
            runner.invoke(cli, arguments)

            full_file_path1 = \
                Path(outDir) / "freeplay" / "prompts" / self.project_id / self.environment / "my-prompt.json"
            self.assertTrue(os.path.isfile(full_file_path1))
            with open(full_file_path1, 'r') as file:
                json_dom = json.load(file)
                self.assertEqual(self.prompt_template_id_1, json_dom['prompt_template_id'])
                self.assertEqual(self.prompt_template_version_id_1, json_dom['prompt_template_version_id'])
                self.assertEqual('my-prompt', json_dom['name'])
                self.assertEqual(0, len(json_dom['metadata']['params']))
                self.assertEqual(
                    '[{"role": "system", "content": "Answer this question: {{question}}"}]',
                    json_dom['content']
                )

            full_file_path2 = \
                Path(outDir) / "freeplay" / "prompts" / self.project_id / self.environment / "my-second-prompt.json"
            self.assertTrue(os.path.isfile(full_file_path2))
            with open(full_file_path2, 'r') as file:
                json_dom = json.load(file)
                self.assertEqual(self.prompt_template_id_2, json_dom['prompt_template_id'])
                self.assertEqual(self.prompt_template_version_id_2, json_dom['prompt_template_version_id'])
                self.assertEqual('my-second-prompt', json_dom['name'])
                self.assertEqual('claude-2', json_dom['metadata']['params']['model'])
                self.assertEqual(0.1, json_dom['metadata']['params']['temperature'])
                self.assertEqual(25, json_dom['metadata']['params']['max_tokens_to_sample'])
                self.assertEqual(
                    "[{\"role\": \"system\", \"content\": \"You're a tech support agent\"}, "
                    "{\"role\": \"assistant\", \"content\": \"How may I help you?\"}, "
                    "{\"role\": \"user\", \"content\": \"Answer this question: {{question}}\"}]",
                    json_dom['content'])

    @responses.activate
    def test_download_invalid_project(self) -> None:
        self.__mock_freeplay_api_invalid_project()

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outDir:
            arguments = [
                'download',
                '--project-id=%s' % "not-a-project",
                '--environment=%s' % self.environment,
                '--output-dir=%s' % outDir
            ]
            # noinspection PyTypeChecker
            result = runner.invoke(cli, arguments)

            self.assertEqual(1, result.exit_code)
            # This actually goes to stderr, but Click combines it with stdout
            self.assertTrue("Error getting prompt templates [404]" in result.stdout)

    def __mock_freeplay_api_success(self) -> None:
        responses.get(
            url=f'{self.api_url}/projects/{self.project_id}/templates/all/{self.environment}',
            status=200,
            body=self.__get_templates_response()
        )

    def __mock_freeplay_api_invalid_project(self) -> None:
        responses.get(
            url=f'{self.api_url}/projects/not-a-project/templates/all/{self.environment}',
            status=404,
            body='{"message": "Project not found"}'
        )

    def __get_templates_response(self) -> str:
        return json.dumps(self.__templates_as_dict())

    def __templates_as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(
            PromptTemplates([
                PromptTemplateWithMetadata(
                    prompt_template_id=self.prompt_template_id_1,
                    prompt_template_version_id=self.prompt_template_version_id_1,
                    name='my-prompt',
                    content=json.dumps([{
                        "role": "system",
                        "content": "Answer this question: {{question}}"
                    }]),
                    params={},
                    flavor_name=None
                ),
                PromptTemplateWithMetadata(
                    prompt_template_id=self.prompt_template_id_2,
                    prompt_template_version_id=self.prompt_template_version_id_2,
                    name='my-second-prompt',
                    content=json.dumps([
                        {
                            "role": "system",
                            "content": self.system_message
                        },
                        {
                            "role": "assistant",
                            "content": self.assistant_message
                        },
                        {
                            "role": "user",
                            "content": self.user_message
                        }
                    ]),
                    params={
                        'model': 'claude-2',
                        'max_tokens_to_sample': 25,
                        'temperature': 0.1
                    },
                    flavor_name='anthropic_chat'
                )
            ])
        )
