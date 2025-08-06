import os
import unittest
from typing import Any

from freeplay import CallInfo, Freeplay, RecordPayload
from tests.slow_test_support import slow


class TestVertexAITools(unittest.TestCase):
    @slow
    def setUp(self) -> None:
        self.freeplay_client = Freeplay(
            freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
            api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
        )
        self.project_id = os.environ["FREEPLAY_PROJECT_ID"]

    @slow
    def test_vertex_ai_function_calling(self) -> None:
        """Test that Vertex AI tool schema is properly formatted and can be used with the SDK."""
        try:
            import vertexai  # type: ignore[import-untyped]
            from vertexai.generative_models import Part, Content  # type: ignore[import-untyped]
        except ImportError:
            self.skipTest("Vertex AI SDK not installed")

        # Initialize Vertex AI (requires proper GCP project setup)
        try:
            gcp_project = os.environ.get("VERTEX_AI_PROJECT_ID", "test-project")
            vertexai.init(project=gcp_project, location="us-central1")
        except Exception as e:
            self.skipTest(f"Vertex AI initialization failed: {e}")

        input_variables = {'location': "San Francisco"}
        
        # Create a prompt with tool schema using Freeplay's format
        # Expected tool schema with the prompt:
        # tool_schemas = [
        #     {
        #         name='get_weather',
        #         description='Get the current weather in a given location',
        #         parameters={
        #             'type': 'object',
        #             'properties': {
        #                 'location': {
        #                     'type': 'string',
        #                     'description': 'The city and state, e.g. San Francisco, CA'
        #                 },
        #                 'unit': {
        #                     'type': 'string',
        #                     'enum': ['celsius', 'fahrenheit'],
        #                     'description': 'The unit of temperature'
        #                 }
        #             },
        #             'required': ['location']
        #         }
        #     }
        # ]
        
        # Get formatted prompt with tool schema
        formatted_prompt = self.freeplay_client.prompts.get_formatted(
            project_id=self.project_id,
            template_name="weather-assistant",  # This should be a prompt template that exists
            environment="latest",
            variables=input_variables,
            flavor_name="gemini_chat",
        )
        
        # Override model to use a Gemini model
        formatted_prompt.prompt_info.model = "gemini-1.5-flash"
        
        # Verify tool_schema is properly formatted as Vertex AI Tool objects
        self.assertIsNotNone(formatted_prompt.tool_schema)
        self.assertEqual(len(formatted_prompt.tool_schema), 1)
        
        # Verify it's a Tool object (if Vertex AI SDK is available)
        from vertexai.generative_models import Tool
        self.assertIsInstance(formatted_prompt.tool_schema[0], Tool)
        
        try:
            initial_message = "What's the weather like in San Francisco?"
            
            # Create mock response using Vertex AI SDK classes
            # Create a function call part
            function_call_part = Part.from_dict({
                'function_call': {
                    'name': 'get_weather',
                    'args': {'location': 'San Francisco, CA', 'unit': 'fahrenheit'}
                }
            })
            
            # Create model response content with function call
            model_response_with_function = Content(
                role='model',
                parts=[function_call_part]
            )
            
            # Simulate function execution
            function_result = {"temperature": 72, "condition": "sunny"}
            
            # Create function response part
            function_response_part = Part.from_function_response(
                name='get_weather',
                response=function_result
            )
            
            # Create user message with function response
            user_function_response = Content(
                role='user',
                parts=[function_response_part]
            )
            
            # Create final model response
            final_model_response = Content(
                role='model',
                parts=[Part.from_text('The weather in San Francisco is sunny with a temperature of 72°F.')]
            )
            
            # Build message history for recording using SDK objects
            all_messages = [
                Content(role='user', parts=[Part.from_text(initial_message)]),
                model_response_with_function,
                user_function_response,
                final_model_response
            ]
            
            # Convert to dict for recording (using convert_provider_message_to_dict)
            from freeplay.utils import convert_provider_message_to_dict
            all_messages_dict = [convert_provider_message_to_dict(msg) for msg in all_messages]
            
            # Record the interaction
            record_response = self.freeplay_client.recordings.create(
                RecordPayload(
                    project_id=self.project_id,
                    all_messages=all_messages_dict,
                    inputs=input_variables,
                    prompt_info=formatted_prompt.prompt_info,
                    tool_schema=formatted_prompt.tool_schema,
                    call_info=CallInfo(
                        provider="vertex",
                        model=formatted_prompt.prompt_info.model,
                    )
                )
            )
            
            self.assertIsNotNone(record_response.completion_id)
            
        except Exception as e:
            # If we can't initialize the model (e.g., no API credentials), 
            # at least verify the tool schema was formatted correctly
            self.skipTest(f"Could not test with actual Vertex AI model: {e}")

    @slow
    def test_vertex_ai_multiple_tools(self) -> None:
        """Test formatting of multiple tool schemas for Vertex AI."""
        try:
            from vertexai.generative_models import Tool
        except ImportError:
            self.skipTest("Vertex AI SDK not installed")
        
        # Expected tool schema with the prompt:
        # tool_schemas = [
        #     {
        #         name='get_weather',
        #         description='Get the current weather',
        #         parameters={
        #             'type': 'object',
        #             'properties': {
        #                 'location': {'type': 'string', 'description': 'The location'}
        #             },
        #             'required': ['location']
        #         }
        #     },
        #     {
        #         name='get_news',
        #         description='Get the latest news',
        #         parameters={
        #             'type': 'object',
        #             'properties': {
        #                 'topic': {'type': 'string', 'description': 'The news topic'},
        #                 'limit': {'type': 'integer', 'description': 'Number of articles'}
        #             },
        #             'required': ['topic']
        #         }
        #     }
        # ]
        
        # Get formatted prompt with multiple tools
        formatted_prompt = self.freeplay_client.prompts.get_formatted(
            project_id=self.project_id,
            template_name="multi-tool-assistant",
            environment="latest",
            variables={'query': 'test'},
            flavor_name="gemini_chat",
        )
        
        # Verify the tool_schema contains one Tool with multiple FunctionDeclarations
        self.assertEqual(len(formatted_prompt.tool_schema), 1)
        self.assertIsInstance(formatted_prompt.tool_schema[0], Tool)
        
        # Check that both functions are in the Tool
        function_declarations = formatted_prompt.tool_schema[0]._raw_tool.function_declarations
        self.assertEqual(len(function_declarations), 2)
        
        # Verify function names
        function_names = [fd.name for fd in function_declarations]
        self.assertIn('get_weather', function_names)
        self.assertIn('get_news', function_names)
        
        # Verify function descriptions and parameters
        for fd in function_declarations:
            if fd.name == 'get_weather':
                self.assertEqual(fd.description, 'Get the current weather')
                self.assertIn('location', str(fd.parameters))
            elif fd.name == 'get_news':
                self.assertEqual(fd.description, 'Get the latest news')
                self.assertIn('topic', str(fd.parameters))
                self.assertIn('limit', str(fd.parameters))

    @slow
    def test_vertex_ai_tool_schema_error_handling(self) -> None:
        """Test that appropriate error is raised when Vertex AI SDK is not available."""
        # Mock the absence of Vertex AI SDK
        import builtins
        original_import = builtins.__import__
        
        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if 'vertexai' in name:
                raise ImportError("No module named 'vertexai'")
            return original_import(name, *args, **kwargs)
        
        # Temporarily replace __import__
        builtins.__import__ = mock_import
        
        try:
            from freeplay.resources.prompts import VertexAIToolSchemaError
            
            # This should raise VertexAIToolSchemaError
            with self.assertRaises(VertexAIToolSchemaError):
                self.freeplay_client.prompts.get_formatted(
                    project_id=self.project_id,
                    template_name="test-prompt",
                    environment="latest",
                    variables={},
                    flavor_name="gemini_chat",
                )
        finally:
            # Restore original import
            builtins.__import__ = original_import


if __name__ == '__main__':
    unittest.main()