import sys
import unittest

from freeplay.errors import FreeplayError
from freeplay.model import InputVariables
from freeplay.utils import bind_template_variables


class MyTestCase(unittest.TestCase):
    template_content = 'Hello, {{name}}, here is a question: {{question_1}} Here is some json: {"is_json": true, ' \
                       '"array": [{}]}'

    def test_format_template_variables(self) -> None:
        variables: InputVariables = {'name': 'Mr. Roboto', 'question_1': 'What is the meaning of life?'}
        self.assertEqual(
            'Hello, Mr. Roboto, here is a question: What is the meaning of life? Here is some json: {"is_json": true, '
            '"array": [{}]}',
            bind_template_variables(self.template_content, variables)
        )

    def test_format_template_variables__no_variables(self) -> None:
        self.assertEqual('Hello world}}', bind_template_variables('Hello world}}', {}))

    def test_format_template_variables__invalid_prompt_template(self) -> None:
        template_content = 'Broken template {{variab}le}}'

        self.assertEqual('Broken template ', bind_template_variables(template_content, {'variable': 'value'}))

    def test_format_template_variables__missing_variables(self) -> None:
        variables: InputVariables = {'name': 'Mr. Roboto'}
        expected = 'Hello, Mr. Roboto, here is a question:  Here is some json: {"is_json": true, "array": [{}]}'

        output = bind_template_variables(self.template_content, variables)
        self.assertEqual(output, expected)

    def test_format_template_variables__extra_variables(self) -> None:
        variables: InputVariables = {'name': 'Mr. Roboto', 'question_1': 'What is the meaning of life?',
                                     'something-else': 'value'}
        self.assertEqual(
            'Hello, Mr. Roboto, here is a question: What is the meaning of life? Here is some json: {"is_json": true, '
            '"array": [{}]}',
            bind_template_variables(self.template_content, variables)
        )

    def test_format_template_variables__bad_variable(self) -> None:
        # We are forcing invalid types with a type ignore here to trigger runtime checks.
        with self.assertRaises(FreeplayError):
            variables: InputVariables = {"foo": None}  # type: ignore
            bind_template_variables('Hello', variables)

        with self.assertRaises(FreeplayError):
            bad_variables: InputVariables = {"foo": lambda s: 1}  # type: ignore
            bind_template_variables('Hello', bad_variables)

    def test_python_version(self) -> None:
        self.assertEqual((3, 8), (sys.version_info[0], sys.version_info[1]), "Tests not running in Python 3.8")
