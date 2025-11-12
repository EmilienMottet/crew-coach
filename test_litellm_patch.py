#!/usr/bin/env python3
"""Test if litellm.completion patch works."""
import sys
from dotenv import load_dotenv
load_dotenv()

# Import patch
import llm_auth_init  # noqa: F401

import litellm

print(f"✅ litellm.completion function: {litellm.completion}")
print(f"   Has patch marker: {getattr(litellm.completion, '_tools_logging_patched', False)}\n")

# Try calling it
print("Calling litellm.completion with tools...")
try:
    result = litellm.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}}
            }
        }],
        api_base="https://ccproxy.emottet.com/copilot/v1",
        drop_params=True,
    )
    print(f"✅ Success: {result.choices[0].message.content[:100]}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\nCalling litellm.completion WITHOUT tools...")
try:
    result = litellm.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://ccproxy.emottet.com/copilot/v1",
        drop_params=True,
    )
    print(f"✅ Success: {result.choices[0].message.content[:100]}")
except Exception as e:
    print(f"❌ Error: {e}")
