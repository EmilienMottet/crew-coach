"""Test LLM providers availability and tool-calling capability.

Imports ALL models from llm_config.py - single source of truth.

Usage:
    # Test ALL models (from llm_config.py)
    pytest test_llm_providers.py -v

    # Test specific model
    pytest test_llm_providers.py -v -k "gpt-5-mini"

    # Quick availability check (standalone)
    python test_llm_providers.py

    # Test only fallback models
    pytest test_llm_providers.py -v -k "gpt-5-mini or raptor-mini"

    # Test by category (filter by model name pattern)
    pytest test_llm_providers.py -v -k "claude"
    pytest test_llm_providers.py -v -k "gpt-5.1"
"""

import os
import pytest
from litellm import completion

# Import ALL model lists from centralized config
from llm_config import (
    COMPLEX_MODELS,
    INTERMEDIATE_MODELS,
    SIMPLE_MODELS,
    FALLBACK_MODELS,
    DEFAULT_BASE_URL,
)

API_KEY = os.getenv("OPENAI_API_KEY", "")

# Combine all models for comprehensive testing
ALL_MODELS = (
    list(FALLBACK_MODELS)
    + list(SIMPLE_MODELS)
    + list(INTERMEDIATE_MODELS)
    + list(COMPLEX_MODELS)
)

# Simple tool definition for testing
TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.parametrize("model", ALL_MODELS)
def test_model_availability(model):
    """Test if model responds to simple prompt."""
    response = completion(
        model=f"openai/{model}",
        api_base=DEFAULT_BASE_URL,
        api_key=API_KEY,
        messages=[{"role": "user", "content": "Say 'OK'"}],
        max_tokens=10,
    )
    assert response.choices[0].message.content is not None
    print(f"✅ {model}: Available")


@pytest.mark.parametrize("model", ALL_MODELS)
def test_model_tool_calling(model):
    """Test if model can use tools correctly."""
    response = completion(
        model=f"openai/{model}",
        api_base=DEFAULT_BASE_URL,
        api_key=API_KEY,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[TEST_TOOL],
        tool_choice="auto",
        max_tokens=100,
    )

    message = response.choices[0].message
    has_tool_call = message.tool_calls is not None and len(message.tool_calls) > 0
    has_content = message.content is not None and len(message.content) > 0

    assert has_tool_call or has_content, f"Model {model} returned empty response"

    if has_tool_call:
        tool_call = message.tool_calls[0]
        assert tool_call.function.name == "get_weather"
        print(f"✅ {model}: Tool calling works")
    else:
        print(f"⚠️ {model}: No tool call, responded with text")


def test_all_models_summary():
    """Summary of all model tests (run standalone)."""
    results = {}
    for model in ALL_MODELS:
        try:
            response = completion(
                model=f"openai/{model}",
                api_base=DEFAULT_BASE_URL,
                api_key=API_KEY,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10,
            )
            results[model] = "✅ Available"
        except Exception as e:
            results[model] = f"❌ {str(e)[:50]}"

    print("\n=== LLM Provider Status ===")
    print(f"Endpoint: {DEFAULT_BASE_URL}")
    for model, status in results.items():
        print(f"  {model}: {status}")


if __name__ == "__main__":
    test_all_models_summary()
