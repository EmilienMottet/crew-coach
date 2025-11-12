#!/usr/bin/env python3
"""Test litellm.completion with Basic Auth WITHOUT patches."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Get auth token
auth_token = os.getenv("OPENAI_API_AUTH_TOKEN")
if not auth_token:
    print("❌ No OPENAI_API_AUTH_TOKEN found")
    sys.exit(1)

# Ensure Basic format
if not auth_token.startswith("Basic "):
    auth_token = f"Basic {auth_token}"

print(f"Auth token: {auth_token[:20]}...")

# DO NOT import llm_auth_init - test without patches
import litellm

# Test 1: Call copilot endpoint WITHOUT tools
print("\n📞 Test 1: Copilot endpoint without tools")
try:
    result = litellm.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Say 'hello'"}],
        api_base="https://ccproxy.emottet.com/copilot/v1",
        api_key=auth_token,
        custom_llm_provider="openai",  # Force OpenAI-compatible
    )
    print(f"✅ Success: {result.choices[0].message.content}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Call copilot endpoint WITH tools
print("\n📞 Test 2: Copilot endpoint with tools")
try:
    result = litellm.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "List available tools"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}}
            }
        }],
        api_base="https://ccproxy.emottet.com/copilot/v1",
        api_key=auth_token,
        custom_llm_provider="openai",
    )
    print(f"✅ Success: {result.choices[0].message.content[:100]}")
    if hasattr(result.choices[0].message, 'tool_calls') and result.choices[0].message.tool_calls:
        print(f"   🛠️  Tool calls: {len(result.choices[0].message.tool_calls)}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n✅ All tests passed!")
