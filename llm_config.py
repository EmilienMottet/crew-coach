"""Centralized LLM configuration - single source of truth.

All model lists, defaults, and blacklists for the meal planning system.
Both llm_provider_rotation.py and test_llm_providers.py import from here.

IMPORTANT: All models listed here must exist on ccproxy.emottet.com/v1/models
Run: curl -s 'https://ccproxy.emottet.com/v1/models' -H 'Authorization: Bearer $KEY' | jq -r '.data[].id' | sort
"""

import os
from typing import Dict, List

# Default endpoint for all models
DEFAULT_BASE_URL = os.getenv("OPENAI_API_BASE", "https://ccproxy.emottet.com/v1")

# Model categories (verified against ccproxy /models endpoint 2025-12-21)
# All models use https://ccproxy.emottet.com/v1 endpoint
COMPLEX_MODELS = (
    # Thinking models (extended reasoning)
    "gemini-claude-sonnet-4-5-thinking",
    "gemini-claude-opus-4-5-thinking",
    # Premium models
    "claude-opus-4-5-20251101",
    "claude-opus-4.5",
    "gemini-3-pro-preview",
)

INTERMEDIATE_MODELS = (
    # Claude Sonnet 4.5 (most reliable)
    "claude-sonnet-4-5-20250929",
    # GPT 5.2 (latest)
    "gpt-5.2",
    "gpt-5.2-codex",
    # Other models
    "glm-4.6",
    "gpt-oss-120b-medium",
)

SIMPLE_MODELS = (
    # Claude Haiku 4.5
    "claude-haiku-4-5-20251001",
    "claude-haiku-4.5",
    # Gemini 3 Flash
    "gemini-3-flash-preview",
    # Qwen 3 Coder
    "qwen3-coder-plus",
    "qwen3-coder-flash",
)

FALLBACK_MODELS = (
    "gpt-5-mini",
    "oswe-vscode-prime",
)

# Category cascade order (from highest to lowest tier)
CATEGORY_CASCADE = ["complex", "intermediate", "simple", "fallback"]

# Thinking models (hallucinate tool calls, must NOT be used with tools)
# These models simulate ReAct reasoning in their content output while returning
# tool_calls=None, causing the agent to fail
THINKING_MODELS = (
    "gemini-claude-sonnet-4-5-thinking",
    "gemini-claude-opus-4-5-thinking",
)

# Default blacklists per agent - centralized to avoid duplication across crews
# These are applied automatically when build_category_cascade() is called
# Note: Can be overridden via {AGENT_NAME}_BLACKLISTED_MODELS env var
DEFAULT_AGENT_BLACKLISTS: Dict[str, List[str]] = {
    # These models don't handle complex Hexis tool calls well
    # gemini-claude-sonnet-4-5-thinking uses ReAct text format instead of native tool_calls
    "HEXIS_ANALYSIS": ["gpt-5", "gpt-5-codex", "gemini-claude-sonnet-4-5-thinking"],
    # gemini-3-pro-preview has meal logging issues
    # gemini-claude-sonnet-4-5-thinking has tool schema format incompatibility
    "MEALY_INTEGRATION": ["gemini-3-pro-preview", "gemini-claude-sonnet-4-5-thinking"],
}

# Hints for models/endpoints that don't support tools
TOOL_FREE_ENDPOINT_HINTS: tuple = ()  # No endpoint blocks tools
TOOL_FREE_MODEL_HINTS: tuple = (
    "deepseek",
    "deepseek-r1",
)


def is_thinking_model(model: str) -> bool:
    """Check if model is a thinking model that shouldn't have tools.

    Thinking models hallucinate tool calls instead of using tool_calls.
    They simulate ReAct in content instead of producing proper tool_calls,
    causing CrewAI to treat their response as a "Final Answer" without
    executing any tools.

    Args:
        model: Model name to check

    Returns:
        True if model is a thinking model
    """
    # Explicit list check
    if model in THINKING_MODELS:
        return True

    model_lower = model.lower()

    # Pattern-based check for any model with "thinking" in the name
    if "thinking" in model_lower:
        return True

    # DeepSeek R1 variants (reasoning models)
    if "deepseek" in model_lower and "r1" in model_lower:
        return True

    return False


def get_models_by_category(category: str) -> tuple:
    """Get models by category name.

    Args:
        category: One of "complex", "intermediate", "simple", "fallback"

    Returns:
        Tuple of model names for that category
    """
    category_map = {
        "complex": COMPLEX_MODELS,
        "intermediate": INTERMEDIATE_MODELS,
        "simple": SIMPLE_MODELS,
        "fallback": FALLBACK_MODELS,
    }
    return category_map.get(category, ())


def get_all_models() -> list:
    """Get all models across all categories.

    Returns:
        List of all model names
    """
    return (
        list(FALLBACK_MODELS)
        + list(SIMPLE_MODELS)
        + list(INTERMEDIATE_MODELS)
        + list(COMPLEX_MODELS)
    )
