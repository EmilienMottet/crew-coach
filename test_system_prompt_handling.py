#!/usr/bin/env python3
"""Test system prompt handling for different endpoints."""

from llm_provider_rotation import _requires_promptless_mode


def test_system_prompt_handling():
    """Test that system prompt is disabled for codex but enabled for others."""
    print("=" * 80)
    print("System Prompt Handling Test")
    print("=" * 80)

    test_cases = [
        # Codex endpoint should disable system prompt
        ("https://ccproxy.emottet.com/codex/v1", "gpt-5", True, "❌ System prompt (codex requires no system prompt)"),
        ("https://ccproxy.emottet.com/codex/v1", "gpt-5-codex", True, "❌ System prompt (codex requires no system prompt)"),

        # Copilot endpoint should allow system prompt
        ("https://ccproxy.emottet.com/copilot/v1", "gpt-5", False, "✅ System prompt"),
        ("https://ccproxy.emottet.com/copilot/v1", "claude-sonnet-4.5", False, "✅ System prompt"),
        ("https://ccproxy.emottet.com/copilot/v1", "gpt-5-mini", False, "✅ System prompt"),

        # Claude endpoint should allow system prompt
        ("https://ccproxy.emottet.com/claude/v1", "claude-sonnet-4-5-20250929", False, "✅ System prompt"),

        # Generic endpoints should allow system prompt
        ("https://api.openai.com/v1", "gpt-4", False, "✅ System prompt"),
    ]

    print("\n🧪 Testing system prompt handling for different endpoints:\n")

    all_pass = True
    for api_base, model_name, expected_disable, description in test_cases:
        result = _requires_promptless_mode(api_base, model_name)
        status = "✅" if result == expected_disable else "❌"

        if result != expected_disable:
            all_pass = False
            print(f"  {status} FAIL: {api_base}")
            print(f"     Model: {model_name}")
            print(f"     Expected: {description}")
            print(f"     Got: {'❌ System prompt' if result else '✅ System prompt'}")
        else:
            endpoint_name = api_base.split("/")[-2] if "/" in api_base else "generic"
            print(f"  {status} {endpoint_name:12} + {model_name:30} → {description}")

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ All system prompt handling tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 80)


if __name__ == "__main__":
    test_system_prompt_handling()
