#!/usr/bin/env python3
"""Test provider rotation with model normalization."""

import os
import sys

# Set environment variables for testing
os.environ["OPENAI_API_BASE"] = "https://ccproxy.emottet.com/copilot/v1"
os.environ["OPENAI_MODEL_NAME"] = "claude-sonnet-4.5"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ENABLE_LLM_PROVIDER_ROTATION"] = "true"
os.environ["LLM_PROVIDER_ROTATION"] = (
    "claude-sonnet|claude-sonnet-4-5-20250929|https://ccproxy.emottet.com/claude/v1|ENV:OPENAI_API_KEY;"
    "codex-gpt5|gpt-5|https://ccproxy.emottet.com/codex/v1|ENV:OPENAI_API_KEY"
)

from llm_provider_rotation import _build_provider_chain, _normalize_model_name


def test_provider_chain():
    """Test that provider chain is built with correct model names."""
    print("=" * 80)
    print("Provider Chain Test with Model Normalization")
    print("=" * 80)

    # Build provider chain
    provider_chain = _build_provider_chain(
        agent_name="TestAgent",
        normalized_model=_normalize_model_name(
            os.environ["OPENAI_MODEL_NAME"],
            os.environ["OPENAI_API_BASE"]
        ),
        api_base=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    print(f"\n✅ Provider chain built with {len(provider_chain)} providers:\n")

    for i, provider in enumerate(provider_chain, 1):
        print(f"{i}. {provider.label}")
        print(f"   Model: {provider.model}")
        print(f"   API Base: {provider.api_base}")
        print(f"   Disable System Prompt: {provider.disable_system_prompt}")
        print(f"   Tool Free Only: {provider.tool_free_only}")
        print()

    # Verify model normalization
    print("=" * 80)
    print("Model Normalization Verification")
    print("=" * 80)

    expected_models = {
        "copilot/v1": "claude-sonnet-4.5",
        "claude/v1": "claude-sonnet-4-5-20250929",
        "codex/v1": "gpt-5",
    }

    all_correct = True
    for provider in provider_chain:
        for endpoint_hint, expected_model in expected_models.items():
            if endpoint_hint in provider.api_base:
                if provider.model == expected_model:
                    print(f"✅ {provider.label}: {provider.model} (correct)")
                else:
                    print(f"❌ {provider.label}: {provider.model} (expected: {expected_model})")
                    all_correct = False
                break

    print("\n" + "=" * 80)
    if all_correct:
        print("✅ All providers have correct model names!")
    else:
        print("❌ Some providers have incorrect model names!")
    print("=" * 80)


if __name__ == "__main__":
    test_provider_chain()
