import sys
import unittest
from typing import Any, Dict

from freeplay.errors import FreeplayError
from freeplay.model import InputVariables
from freeplay.utils import bind_template_variables, all_valid, convert_provider_message_to_dict


class TestUtils(unittest.TestCase):
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

    def test_all_valid(self) -> None:
        self.assertFalse(
            all_valid({'foo': None}))
        self.assertFalse(
            all_valid({1: 2}))
        self.assertFalse(
            all_valid({'foo': [1, None]}))
        self.assertTrue(
            all_valid({'foo': 1}))
        self.assertTrue(
            all_valid({'foo': 'bar'}))
        self.assertTrue(
            all_valid({'foo': [1, 2, 3]}))
        self.assertTrue(
            all_valid({'foo': {'a': False}}))
        self.assertTrue(
            all_valid({'foo': 2.22})
        )

    def test_json(self) -> None:
        template = '{{foo}}'
        variables = {'foo': {'bar': 'baz'}}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, '{"bar":"baz"}')

    def test_number(self) -> None:
        template = '{{foo}}'
        variables = {'foo': 1}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, '1')

    def test_conditional(self) -> None:
        template = '{{#bar}}{{foo}}{{/bar}}'
        self.assertEqual(bind_template_variables(template, {'foo': 1, 'bar': False}), '')
        self.assertEqual(bind_template_variables(template, {'foo': 1, 'bar': True}), '1')

    def test_literal(self) -> None:
        # This is different than JS.
        template = '{{{literal}}}'
        formatted = bind_template_variables(template, {'literal': {'foo': 'bar'}})
        self.assertEqual(formatted, '{"foo":"bar"}')

    def test_undefined_variable(self) -> None:
        template = '{{foo}}'
        variables: Dict[Any, Any] = {}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, '')

    def test_null_variable(self) -> None:
        template = '{{foo}}'
        variables = {'foo': None}
        with self.assertRaises(Exception):
            bind_template_variables(template, variables)  # type: ignore

    def test_array_variable(self) -> None:
        template = '{{#foo}}{{.}}{{/foo}}'
        variables = {'foo': [1, 2, 3]}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, '123')

    def test_nested_object(self) -> None:
        template = '{{foo.bar}}'
        variables = {'foo': {'bar': 'baz'}}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, 'baz')

    def test_unescaped_characters(self) -> None:
        template = '{{{foo}}}'
        variables = {'foo': '<script>alert("xss")</script>'}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, '<script>alert("xss")</script>')

    def test_missing_closing_tag(self) -> None:
        # This does not throw but ideally it would.
        template = '{{#foo}}{{bar}}'
        variables: InputVariables = {'foo': True, 'bar': 'baz'}
        bind_template_variables(template, variables)

    def test_empty_template(self) -> None:
        template = ''
        variables = {'foo': 'bar'}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, '')

    def test_whitespace_handling(self) -> None:
        template = '{{ foo }}'
        variables = {'foo': 'bar'}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, 'bar')

    def test_array_of_numbers_and_strings(self) -> None:
        template = '{{#foo}}{{.}}{{/foo}}'
        variables = {'foo': [1, 'two', 3, 'four']}
        formatted = bind_template_variables(template, variables)
        self.assertEqual(formatted, '1two3four')

    def test_convert_provider_message_to_dict_with_vertex_ai(self) -> None:
        """Test that convert_provider_message_to_dict handles Vertex AI objects with to_dict() method."""

        # Mock a Vertex AI object with to_dict method
        class MockVertexAIMessage:
            def __init__(self, content: str):
                self.content = content
                self.role = "model"

            def to_dict(self) -> Dict[str, Any]:
                return {"content": self.content, "role": self.role}

        # Test with Vertex AI mock object
        vertex_msg = MockVertexAIMessage("Hello from Vertex AI")
        result = convert_provider_message_to_dict(vertex_msg)
        self.assertEqual(result, {"content": "Hello from Vertex AI", "role": "model"})

        # Test with regular dict (should pass through)
        regular_dict = {"content": "Regular message", "role": "user"}
        result = convert_provider_message_to_dict(regular_dict)
        self.assertEqual(result, regular_dict)

        # Test with list of mixed objects
        mixed_list = [
            MockVertexAIMessage("First message"),
            {"content": "Second message", "role": "user"},
            MockVertexAIMessage("Third message")
        ]
        result = convert_provider_message_to_dict(mixed_list)
        self.assertEqual(result, [
            {"content": "First message", "role": "model"},
            {"content": "Second message", "role": "user"},
            {"content": "Third message", "role": "model"}
        ])

    def test_convert_provider_message_to_dict_with_pydantic(self) -> None:
        """Test that convert_provider_message_to_dict handles Pydantic models."""
        try:
            from pydantic import BaseModel

            class PydanticMessage(BaseModel):
                content: str
                role: str

            # Test with Pydantic v2 model
            pydantic_msg = PydanticMessage(content="Pydantic message", role="assistant")
            result = convert_provider_message_to_dict(pydantic_msg)
            self.assertEqual(result, {"content": "Pydantic message", "role": "assistant"})
        except ImportError:
            self.skipTest("Pydantic not installed")

    def test_convert_provider_message_to_dict_with_nested_structures(self) -> None:
        """Test conversion of nested structures with provider objects."""

        class MockProviderObject:
            def __init__(self, data: Dict[str, Any]):
                self.data = data

            def to_dict(self) -> Dict[str, Any]:
                return self.data

        # Test nested structure
        nested = {
            "messages": [
                MockProviderObject({"text": "Hello", "type": "text"}),
                {"text": "World", "type": "text"}
            ],
            "metadata": MockProviderObject({"timestamp": 12345, "source": "test"})
        }

        result = convert_provider_message_to_dict(nested)
        expected = {
            "messages": [
                {"text": "Hello", "type": "text"},
                {"text": "World", "type": "text"}
            ],
            "metadata": {"timestamp": 12345, "source": "test"}
        }
        self.assertEqual(result, expected)
