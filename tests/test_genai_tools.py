"""
Unit tests for GenAI tool schema types.
Tests the GenaiFunction and GenaiTool dataclasses and their serialization.
"""

import unittest
from dataclasses import asdict

from freeplay import GenaiFunction, GenaiTool
from freeplay.utils import convert_provider_message_to_dict


class TestGenaiToolSchema(unittest.TestCase):
    def test_genai_function_creation(self) -> None:
        """Test creating a GenaiFunction with proper structure."""
        function = GenaiFunction(
            name="get_weather",
            description="Get the current weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
                "required": ["location"],
            },
        )

        self.assertEqual(function.name, "get_weather")
        self.assertEqual(
            function.description, "Get the current weather in a given location"
        )
        self.assertIn("location", function.parameters["properties"])
        self.assertEqual(function.parameters["required"], ["location"])

    def test_genai_tool_single_function(self) -> None:
        """Test creating a GenaiTool with a single function declaration."""
        function = GenaiFunction(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )

        tool = GenaiTool(functionDeclarations=[function])

        self.assertEqual(len(tool.functionDeclarations), 1)
        self.assertEqual(tool.functionDeclarations[0].name, "get_weather")

    def test_genai_tool_multiple_functions(self) -> None:
        """Test creating a GenaiTool with multiple function declarations."""
        get_weather = GenaiFunction(
            name="get_weather",
            description="Get the current weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )

        get_news = GenaiFunction(
            name="get_news",
            description="Get the latest news",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The news topic"},
                    "limit": {"type": "integer", "description": "Number of articles"},
                },
                "required": ["topic"],
            },
        )

        tool = GenaiTool(functionDeclarations=[get_weather, get_news])

        self.assertEqual(len(tool.functionDeclarations), 2)
        self.assertEqual(tool.functionDeclarations[0].name, "get_weather")
        self.assertEqual(tool.functionDeclarations[1].name, "get_news")

    def test_genai_tool_serialization_to_dict(self) -> None:
        """Test that GenaiTool can be properly serialized to dict for API calls."""
        function = GenaiFunction(
            name="calculate_sum",
            description="Calculate the sum of two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        )

        tool = GenaiTool(functionDeclarations=[function])

        # Test using asdict
        tool_dict = asdict(tool)
        self.assertIn("functionDeclarations", tool_dict)
        self.assertEqual(len(tool_dict["functionDeclarations"]), 1)
        self.assertEqual(tool_dict["functionDeclarations"][0]["name"], "calculate_sum")

        # Test using convert_provider_message_to_dict
        converted_dict = convert_provider_message_to_dict(tool)
        self.assertIn("functionDeclarations", converted_dict)
        self.assertEqual(
            converted_dict["functionDeclarations"][0]["name"], "calculate_sum"
        )

    def test_genai_tool_format_matches_expected(self) -> None:
        """Test that GenaiTool format matches the expected GenAI API format."""
        function = GenaiFunction(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        )

        tool = GenaiTool(functionDeclarations=[function])
        tool_dict = asdict(tool)

        # Verify the structure matches GenAI API format:
        # {
        #   "functionDeclarations": [
        #     {
        #       "name": "search",
        #       "description": "Search for information",
        #       "parameters": {...}
        #     }
        #   ]
        # }
        self.assertIn("functionDeclarations", tool_dict)
        self.assertIsInstance(tool_dict["functionDeclarations"], list)

        first_function = tool_dict["functionDeclarations"][0]
        self.assertEqual(first_function["name"], "search")
        self.assertEqual(first_function["description"], "Search for information")
        self.assertIn("parameters", first_function)
        self.assertEqual(first_function["parameters"]["type"], "object")
        self.assertIn("query", first_function["parameters"]["properties"])

    def test_empty_function_declarations(self) -> None:
        """Test creating a GenaiTool with empty function declarations."""
        tool = GenaiTool(functionDeclarations=[])
        self.assertEqual(len(tool.functionDeclarations), 0)

    def test_complex_parameter_schema(self) -> None:
        """Test GenaiFunction with complex nested parameter schema."""
        function = GenaiFunction(
            name="book_flight",
            description="Book a flight with passenger and destination details",
            parameters={
                "type": "object",
                "properties": {
                    "passenger": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "passport": {"type": "string"},
                        },
                        "required": ["name", "passport"],
                    },
                    "destination": {
                        "type": "object",
                        "properties": {
                            "airport_code": {"type": "string"},
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["airport_code"],
                    },
                    "dates": {
                        "type": "object",
                        "properties": {
                            "departure": {"type": "string", "format": "date"},
                            "return": {"type": "string", "format": "date"},
                        },
                        "required": ["departure"],
                    },
                },
                "required": ["passenger", "destination", "dates"],
            },
        )

        tool = GenaiTool(functionDeclarations=[function])
        tool_dict = asdict(tool)

        # Verify complex schema is preserved
        first_function = tool_dict["functionDeclarations"][0]
        self.assertIn("passenger", first_function["parameters"]["properties"])
        self.assertIn("destination", first_function["parameters"]["properties"])
        self.assertIn("dates", first_function["parameters"]["properties"])

        # Verify nested properties
        passenger = first_function["parameters"]["properties"]["passenger"]
        self.assertEqual(passenger["type"], "object")
        self.assertIn("name", passenger["properties"])
        self.assertIn("passport", passenger["properties"])


if __name__ == "__main__":
    unittest.main()
