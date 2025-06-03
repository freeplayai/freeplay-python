import json
import os
import os.path
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import TestCase
from uuid import uuid4

import responses
from click.testing import CliRunner

from freeplay.freeplay_cli import cli


class TestFreeplayCLI(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.project_id_1 = str(uuid4())
        self.project_name_1 = "project-one"
        self.project_id_2 = str(uuid4())
        self.project_name_2 = "project-two"

        self.prompt_name_1 = "my-prompt"
        self.prompt_name_2 = "my-second-prompt"
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
        self.__mock_freeplay_prompts_api_success(self.project_id_1, self.environment)

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outDir:
            arguments = [
                'download',
                '--project-id=%s' % self.project_id_1,
                '--environment=%s' % self.environment,
                '--output-dir=%s' % outDir
            ]
            # noinspection PyTypeChecker
            runner.invoke(cli, arguments)

            full_file_path1 = \
                Path(outDir) / "freeplay" / "prompts" / self.project_id_1 / self.environment / f"{self.prompt_name_1}.json"
            self.assertTrue(os.path.isfile(full_file_path1))
            with open(full_file_path1, 'r') as file:
                json_dom = json.load(file)
                self.assertEqual(self.prompt_template_id_1, json_dom['prompt_template_id'])
                self.assertEqual(self.prompt_template_version_id_1, json_dom['prompt_template_version_id'])
                self.assertEqual(self.prompt_name_1, json_dom['prompt_template_name'])
                self.assertEqual(0, len(json_dom['metadata']['params']))
                self.assertEqual(
                    [{"role": "system", "content": "Answer this question: {{question}}", "media_slots": []}],
                    json_dom['content']
                )

            full_file_path2 = \
                Path(outDir) / "freeplay" / "prompts" / self.project_id_1 / self.environment / f"{self.prompt_name_2}.json"
            self.assertTrue(os.path.isfile(full_file_path2))
            with open(full_file_path2, 'r') as file:
                json_dom = json.load(file)
                self.assertEqual(self.prompt_template_id_2, json_dom['prompt_template_id'])
                self.assertEqual(self.prompt_template_version_id_2, json_dom['prompt_template_version_id'])
                self.assertEqual(self.prompt_name_2, json_dom['prompt_template_name'])
                self.assertEqual('claude-2', json_dom['metadata']['params']['model'])
                self.assertEqual(0.1, json_dom['metadata']['params']['temperature'])
                self.assertEqual(25, json_dom['metadata']['params']['max_tokens_to_sample'])
                self.assertEqual(
                    [
                        {"role": "system", "content": self.system_message, "media_slots": []},
                        {"role": "assistant", "content": self.assistant_message, "media_slots": []},
                        {"role": "user", "content": self.user_message, "media_slots": []}
                    ],
                    json_dom['content']
                )

    @responses.activate
    def test_download_all_succeeds_across_projects(self) -> None:
        self.__mock_freeplay_projects_api_success()
        self.__mock_freeplay_prompts_api_success(self.project_id_1, self.environment)
        self.__mock_freeplay_prompts_api_success(self.project_id_2, self.environment)

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outDir:
            outDir = os.path.abspath(outDir)  # Ensure absolute path
            arguments = [
                'download-all',
                '--environment=%s' % self.environment,
                '--output-dir=%s' % outDir
            ]
            # noinspection PyTypeChecker
            result = runner.invoke(cli, arguments, catch_exceptions=False)
            self.assertEqual(0, result.exit_code)

            self.__assert_prompt_basics(outDir, self.project_id_1, self.environment, self.prompt_name_1)
            self.__assert_prompt_basics(outDir, self.project_id_1, self.environment, self.prompt_name_2)

            self.__assert_prompt_basics(outDir, self.project_id_2, self.environment, self.prompt_name_1)
            self.__assert_prompt_basics(outDir, self.project_id_2, self.environment, self.prompt_name_2)

    @responses.activate
    def test_download_all_defaults_environment_to_latest(self) -> None:
        environment = 'latest'

        self.__mock_freeplay_projects_api_success()
        self.__mock_freeplay_prompts_api_success(self.project_id_1, environment)
        self.__mock_freeplay_prompts_api_success(self.project_id_2, environment)

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as outDir:
            outDir = os.path.abspath(outDir)  # Ensure absolute path
            arguments = [
                'download-all',
                '--output-dir=%s' % outDir
            ]
            # noinspection PyTypeChecker
            result = runner.invoke(cli, arguments, catch_exceptions=False)
            self.assertEqual(0, result.exit_code)

            self.__assert_prompt_basics(outDir, self.project_id_1, environment, self.prompt_name_1)
            self.__assert_prompt_basics(outDir, self.project_id_1, environment, self.prompt_name_2)

            self.__assert_prompt_basics(outDir, self.project_id_2, environment, self.prompt_name_1)
            self.__assert_prompt_basics(outDir, self.project_id_2, environment, self.prompt_name_2)

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

    def __assert_prompt_basics(
        self, out_dir: str, project_id: str, environment: str, prompt_name: str
    ) -> None:
        full_file_path = \
            Path(out_dir) / "freeplay" / "prompts" / project_id / environment / f"{prompt_name}.json"
        self.assertTrue(os.path.isfile(full_file_path))
        with open(full_file_path, 'r') as file:
            json_dom = json.load(file)
            self.assertEqual(prompt_name, json_dom['prompt_template_name'])
            self.assertEqual(project_id, json_dom['project_id'])

    def __mock_freeplay_prompts_api_success(
            self,
            project_id: str,
            environment: str
    ) -> None:
        responses.get(
            url=f'{self.api_url}/v2/projects/{project_id}/prompt-templates/all/{environment}',
            status=200,
            body=self.__get_templates_response(project_id)
        )

    def __mock_freeplay_api_invalid_project(self) -> None:
        responses.get(
            url=f'{self.api_url}/v2/projects/not-a-project/prompt-templates/all/{self.environment}',
            status=404,
            body='{"message": "Project not found"}'
        )

    def __get_templates_response(self, project_id: str) -> str:
        return json.dumps(self.__templates_as_dict(project_id))

    def __templates_as_dict(self, project_id: str) -> Dict[str, Any]:
        return {
            "prompt_templates": [{
                "content": [{
                    "role": "system",
                    "content": "Answer this question: {{question}}"
                }],
                "format_version": 2,
                "metadata": {
                    "flavor": "anthropic_chat",
                    "model": "claude-2.1",
                    "params": {},
                    "provider": "anthropic",
                    "provider_info": {
                        "anthropic_endpoint": "https://example.com/anthropic"
                    }
                },
                "project_id": project_id,
                "prompt_template_id": self.prompt_template_id_1,
                "prompt_template_name": self.prompt_name_1,
                "prompt_template_version_id": self.prompt_template_version_id_1
            }, {
                "content": [
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
                ],
                "format_version": 2,
                "metadata": {
                    "flavor": "anthropic_chat",
                    "model": "claude-2.1",
                    "params": {
                        'model': 'claude-2',
                        'max_tokens_to_sample': 25,
                        'temperature': 0.1,
                    },
                    "provider": "anthropic",
                    "provider_info": {
                        "anthropic_endpoint": "https://example.com/anthropic"
                    }
                },
                "project_id": project_id,
                "prompt_template_id": self.prompt_template_id_2,
                "prompt_template_name": self.prompt_name_2,
                "prompt_template_version_id": self.prompt_template_version_id_2
            }]
        }

    def __mock_freeplay_projects_api_success(self) -> None:
        responses.get(
            url=f'{self.api_url}/v2/projects/all',
            status=200,
            body=self.__get_projects_response()
        )

    def __get_projects_response(self) -> str:
        return json.dumps(self.__projects_as_dict())

    def __projects_as_dict(self) -> Dict[str, Any]:
        return {
            "projects": [
                {"id": str(self.project_id_1), "name": self.project_name_1},
                {"id": str(self.project_id_2), "name": self.project_name_2},
            ]
        }
