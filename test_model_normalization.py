#!/usr/bin/env python3
"""Test model normalization for different endpoints."""

from llm_provider_rotation import _normalize_model_name


def test_copilot_endpoint():
    """Test model normalization for copilot endpoint."""
    print("🧪 Testing Copilot Endpoint (/copilot/v1)")

    test_cases = [
        ("claude-sonnet-4.5", "https://ccproxy.emottet.com/copilot/v1", "claude-sonnet-4.5"),
        ("claude-sonnet-4-5", "https://ccproxy.emottet.com/copilot/v1", "claude-sonnet-4.5"),
        ("claude-haiku-4.5", "https://ccproxy.emottet.com/copilot/v1", "claude-haiku-4.5"),
        ("gpt-5", "https://ccproxy.emottet.com/copilot/v1", "gpt-5"),
        ("gpt-5-mini", "https://ccproxy.emottet.com/copilot/v1", "gpt-5-mini"),
        ("gpt-5-codex", "https://ccproxy.emottet.com/copilot/v1", "gpt-5-codex"),
        ("gpt-4.1", "https://ccproxy.emottet.com/copilot/v1", "gpt-4.1"),
        ("gemini-2.5-pro", "https://ccproxy.emottet.com/copilot/v1", "gemini-2.5-pro"),
    ]

    for model_input, api_base, expected_output in test_cases:
        result = _normalize_model_name(model_input, api_base)
        status = "✅" if result == expected_output else "❌"
        print(f"  {status} {model_input:25} → {result:30} (expected: {expected_output})")
        if result != expected_output:
            print(f"     ERROR: Got '{result}' but expected '{expected_output}'")


def test_codex_endpoint():
    """Test model normalization for codex endpoint."""
    print("\n🧪 Testing Codex Endpoint (/codex/v1)")

    test_cases = [
        ("gpt-5", "https://ccproxy.emottet.com/codex/v1", "gpt-5"),
        ("gpt-5-codex", "https://ccproxy.emottet.com/codex/v1", "gpt-5-codex"),
        ("gpt-4.1", "https://ccproxy.emottet.com/codex/v1", "gpt-5"),  # Falls back to gpt-5
        ("claude-sonnet-4.5", "https://ccproxy.emottet.com/codex/v1", "gpt-5"),  # Falls back to gpt-5
    ]

    for model_input, api_base, expected_output in test_cases:
        result = _normalize_model_name(model_input, api_base)
        status = "✅" if result == expected_output else "❌"
        print(f"  {status} {model_input:25} → {result:30} (expected: {expected_output})")
        if result != expected_output:
            print(f"     ERROR: Got '{result}' but expected '{expected_output}'")


def test_claude_endpoint():
    """Test model normalization for claude endpoint."""
    print("\n🧪 Testing Claude Endpoint (/claude/v1)")

    test_cases = [
        ("claude-sonnet-4.5", "https://ccproxy.emottet.com/claude/v1", "claude-sonnet-4-5-20250929"),
        ("claude-sonnet-4-5", "https://ccproxy.emottet.com/claude/v1", "claude-sonnet-4-5-20250929"),
        ("claude-haiku-4.5", "https://ccproxy.emottet.com/claude/v1", "claude-haiku-4-5-20251001"),
        ("claude-opus-4.1", "https://ccproxy.emottet.com/claude/v1", "claude-opus-4-1-20250805"),
        ("claude-3.5-sonnet", "https://ccproxy.emottet.com/claude/v1", "claude-3-5-sonnet-20241022"),
        ("claude-3-opus", "https://ccproxy.emottet.com/claude/v1", "claude-3-opus-20240229"),
        # Already full version
        ("claude-sonnet-4-5-20250929", "https://ccproxy.emottet.com/claude/v1", "claude-sonnet-4-5-20250929"),
    ]

    for model_input, api_base, expected_output in test_cases:
        result = _normalize_model_name(model_input, api_base)
        status = "✅" if result == expected_output else "❌"
        print(f"  {status} {model_input:30} → {result:35} (expected: {expected_output})")
        if result != expected_output:
            print(f"     ERROR: Got '{result}' but expected '{expected_output}'")


def test_generic_endpoint():
    """Test model normalization for generic endpoints."""
    print("\n🧪 Testing Generic Endpoint (non-ccproxy)")

    test_cases = [
        ("gpt-4", "https://api.openai.com/v1", "openai/gpt-4"),
        ("claude-3", "https://api.anthropic.com/v1", "openai/claude-3"),
    ]

    for model_input, api_base, expected_output in test_cases:
        result = _normalize_model_name(model_input, api_base)
        status = "✅" if result == expected_output else "❌"
        print(f"  {status} {model_input:25} → {result:30} (expected: {expected_output})")
        if result != expected_output:
            print(f"     ERROR: Got '{result}' but expected '{expected_output}'")


if __name__ == "__main__":
    print("=" * 80)
    print("Model Normalization Test Suite")
    print("=" * 80)

    test_copilot_endpoint()
    test_codex_endpoint()
    test_claude_endpoint()
    test_generic_endpoint()

    print("\n" + "=" * 80)
    print("✅ Test suite complete!")
    print("=" * 80)
