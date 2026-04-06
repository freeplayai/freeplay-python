import copy
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_THINKING_LEVEL_BUDGETS: Dict[str, int] = {
    "off": 0,
    "none": 0,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}


class LLMParameters(Dict[str, Any]):
    def __init__(self, members: Dict[str, Any]) -> None:
        super().__init__(members)

    @classmethod
    def empty(cls) -> "LLMParameters":
        return LLMParameters({})

    def merge_and_override(
        self, additional_params: Optional["LLMParameters"]
    ) -> "LLMParameters":
        updated_params = copy.deepcopy(self)

        if additional_params is not None:
            for model_param_key, value in additional_params.items():
                logger.debug(
                    f"Overriding parameter '{model_param_key}' using value '{value}' from get_completion call"
                )
                updated_params[model_param_key] = value

        return updated_params

    def for_gemini(self) -> "LLMParameters":
        """Return a new LLMParameters with keys mapped for Gemini / Vertex AI.

        Transformations applied:
        - ``max_tokens`` is renamed to ``max_output_tokens``
        - ``thinking_level`` is converted to a ``thinking_config`` dict
        - All other keys (including ``temperature``) are passed through unchanged.

        The returned parameters can be used directly as a Gemini
        ``generation_config`` dict (Vertex AI) or as keyword arguments to
        ``types.GenerateContentConfig`` (google-genai SDK).
        """
        result: Dict[str, Any] = {}
        for key, value in self.items():
            if key == "max_tokens":
                result["max_output_tokens"] = value
            elif key == "thinking_level":
                result["thinking_config"] = self._thinking_level_to_config(value)
            else:
                result[key] = copy.deepcopy(value)
        return LLMParameters(result)

    @staticmethod
    def _thinking_level_to_config(level: Any) -> Dict[str, Any]:
        if isinstance(level, str):
            normalized = level.strip().lower()
            budget = _THINKING_LEVEL_BUDGETS.get(normalized)
            if budget is not None:
                return {"thinking_budget": budget}
            logger.warning(
                "Unknown thinking_level '%s'; passing through as thinking_budget",
                level,
            )
            try:
                return {"thinking_budget": int(level)}
            except (ValueError, TypeError):
                return {"thinking_budget": 0}
        if isinstance(level, (int, float)):
            return {"thinking_budget": int(level)}
        return {"thinking_budget": 0}

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        return super().pop(key, default)
