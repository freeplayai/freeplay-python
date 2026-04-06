import unittest

from freeplay.llm_parameters import LLMParameters


class TestLLMParametersForGemini(unittest.TestCase):
    def test_temperature_passed_through(self) -> None:
        params = LLMParameters({"temperature": 0.5})
        result = params.for_gemini()
        self.assertEqual(result["temperature"], 0.5)

    def test_temperature_zero_passed_through(self) -> None:
        params = LLMParameters({"temperature": 0})
        result = params.for_gemini()
        self.assertEqual(result["temperature"], 0)

    def test_max_tokens_renamed(self) -> None:
        params = LLMParameters({"max_tokens": 1024})
        result = params.for_gemini()
        self.assertNotIn("max_tokens", result)
        self.assertEqual(result["max_output_tokens"], 1024)

    def test_thinking_level_low(self) -> None:
        params = LLMParameters({"thinking_level": "low"})
        result = params.for_gemini()
        self.assertNotIn("thinking_level", result)
        self.assertEqual(result["thinking_config"], {"thinking_budget": 1024})

    def test_thinking_level_medium(self) -> None:
        params = LLMParameters({"thinking_level": "medium"})
        result = params.for_gemini()
        self.assertEqual(result["thinking_config"], {"thinking_budget": 8192})

    def test_thinking_level_high(self) -> None:
        params = LLMParameters({"thinking_level": "high"})
        result = params.for_gemini()
        self.assertEqual(result["thinking_config"], {"thinking_budget": 24576})

    def test_thinking_level_off(self) -> None:
        params = LLMParameters({"thinking_level": "off"})
        result = params.for_gemini()
        self.assertEqual(result["thinking_config"], {"thinking_budget": 0})

    def test_thinking_level_none(self) -> None:
        params = LLMParameters({"thinking_level": "none"})
        result = params.for_gemini()
        self.assertEqual(result["thinking_config"], {"thinking_budget": 0})

    def test_thinking_level_case_insensitive(self) -> None:
        params = LLMParameters({"thinking_level": "Low"})
        result = params.for_gemini()
        self.assertEqual(result["thinking_config"], {"thinking_budget": 1024})

    def test_thinking_level_numeric(self) -> None:
        params = LLMParameters({"thinking_level": 4096})
        result = params.for_gemini()
        self.assertEqual(result["thinking_config"], {"thinking_budget": 4096})

    def test_combined_parameters(self) -> None:
        params = LLMParameters({
            "temperature": 0,
            "max_tokens": 256,
            "thinking_level": "low",
            "top_p": 0.9,
        })
        result = params.for_gemini()
        self.assertEqual(result["temperature"], 0)
        self.assertEqual(result["max_output_tokens"], 256)
        self.assertEqual(result["thinking_config"], {"thinking_budget": 1024})
        self.assertEqual(result["top_p"], 0.9)
        self.assertNotIn("max_tokens", result)
        self.assertNotIn("thinking_level", result)

    def test_empty_parameters(self) -> None:
        params = LLMParameters({})
        result = params.for_gemini()
        self.assertEqual(dict(result), {})

    def test_returns_llm_parameters_instance(self) -> None:
        params = LLMParameters({"temperature": 0.5})
        result = params.for_gemini()
        self.assertIsInstance(result, LLMParameters)

    def test_does_not_mutate_original(self) -> None:
        params = LLMParameters({
            "max_tokens": 256,
            "thinking_level": "high",
            "temperature": 0.5,
        })
        _ = params.for_gemini()
        self.assertEqual(params["max_tokens"], 256)
        self.assertEqual(params["thinking_level"], "high")
        self.assertNotIn("max_output_tokens", params)
        self.assertNotIn("thinking_config", params)

    def test_passthrough_unknown_params(self) -> None:
        params = LLMParameters({
            "top_k": 40,
            "stop_sequences": ["END"],
        })
        result = params.for_gemini()
        self.assertEqual(result["top_k"], 40)
        self.assertEqual(result["stop_sequences"], ["END"])

    def test_max_output_tokens_already_present(self) -> None:
        params = LLMParameters({"max_output_tokens": 512})
        result = params.for_gemini()
        self.assertEqual(result["max_output_tokens"], 512)
