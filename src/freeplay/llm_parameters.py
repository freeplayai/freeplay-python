import copy
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
        - ``thinking_level`` is converted to a ``thinking_config`` dict with
          either ``thinking_level`` (string values like ``"low"``,
          ``"medium"``, ``"high"``) or ``thinking_budget`` (numeric values)
        - All other keys (including ``temperature``) are passed through unchanged.

        The returned parameters can be used directly as a Gemini
        ``generation_config`` dict or as keyword arguments to
        ``types.GenerateContentConfig`` (google-genai SDK).
        """
        result: Dict[str, Any] = {}
        for key, value in self.items():
            if key == "max_tokens":
                result["max_output_tokens"] = value
            elif key == "thinking_level":
                result["thinking_config"] = _thinking_level_to_config(value)
            else:
                result[key] = copy.deepcopy(value)
        return LLMParameters(result)

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        return super().pop(key, default)


def _thinking_level_to_config(level: Any) -> Dict[str, Any]:
    """Convert a thinking_level value to a Gemini thinking_config dict.

    String values (``"low"``, ``"medium"``, ``"high"``, ``"minimal"``, etc.)
    are passed through as ``thinking_level`` for Gemini 3+ models.
    Numeric values are passed as ``thinking_budget`` for Gemini 2.5 models.
    """
    if isinstance(level, str):
        return {"thinking_level": level.strip().lower()}
    if isinstance(level, (int, float)):
        return {"thinking_budget": int(level)}
    return {"thinking_level": str(level)}
