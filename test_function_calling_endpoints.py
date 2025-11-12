#!/usr/bin/env python3
"""Test function calling compatibility for /claude/v1 and /copilot/v1 endpoints.

This script verifies whether the endpoints support OpenAI's function calling format,
which is required for CrewAI tools and MCP integration.
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def test_function_calling(endpoint_url: str, model: str, api_key: str) -> dict:
    """
    Test if an endpoint supports OpenAI-style function calling.
    
    Args:
        endpoint_url: Base URL of the endpoint (e.g., https://ccproxy.emottet.com/copilot/v1)
        model: Model name to use
        api_key: Authentication key (may be Basic Auth token)
        
    Returns:
        Dictionary with test results
    """
    # Define a simple test function/tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. Paris"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Prepare request
    headers = {
        "Content-Type": "application/json"
    }
    
    # Handle Basic Auth vs Bearer token
    if api_key.startswith("Basic "):
        headers["Authorization"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in Paris?"
            }
        ],
        "tools": tools,
        "tool_choice": "auto"  # Let the model decide whether to call the function
    }
    
    url = f"{endpoint_url.rstrip('/')}/chat/completions"
    
    try:
        print(f"🔍 Testing endpoint: {url}")
        print(f"   Model: {model}")
        print(f"   Tools defined: {len(tools)}")
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:500]}",
                "supports_function_calling": False
            }
        
        result = response.json()
        
        # Check if the response contains tool calls
        choices = result.get("choices", [])
        if not choices:
            return {
                "success": False,
                "error": "No choices in response",
                "supports_function_calling": False,
                "response": result
            }
        
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")
        
        # Function calling is supported if:
        # 1. Request succeeded (already verified)
        # 2. Response has expected structure
        # 3. Tool calls are present OR finish_reason indicates compatibility
        
        finish_reason = choices[0].get("finish_reason")
        
        if tool_calls:
            # Perfect! The model actually called the function
            return {
                "success": True,
                "supports_function_calling": True,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "message": "✅ Endpoint supports function calling (tool call detected)"
            }
        elif finish_reason == "tool_calls":
            # The model indicated it wants to call a tool
            return {
                "success": True,
                "supports_function_calling": True,
                "finish_reason": finish_reason,
                "message": "✅ Endpoint supports function calling (finish_reason=tool_calls)"
            }
        elif finish_reason in ["stop", "length"]:
            # The model responded normally without calling the tool
            # This doesn't mean it doesn't support function calling,
            # just that it chose not to use it for this query
            content = message.get("content", "")
            return {
                "success": True,
                "supports_function_calling": "uncertain",
                "finish_reason": finish_reason,
                "content": content[:200],
                "message": "⚠️ Endpoint accepted function definitions but didn't call them (may still be compatible)"
            }
        else:
            return {
                "success": True,
                "supports_function_calling": False,
                "finish_reason": finish_reason,
                "response": result,
                "message": "❌ Endpoint doesn't support function calling (unexpected finish_reason)"
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout (>60s)",
            "supports_function_calling": False
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "supports_function_calling": False
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "supports_function_calling": False
        }


def main():
    """Test both /claude/v1 and /copilot/v1 endpoints."""
    
    # Get authentication
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_AUTH_TOKEN", "")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY or OPENAI_API_AUTH_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    
    # Ensure Basic Auth format if token is provided
    if api_key and not api_key.startswith("Basic ") and not api_key.startswith("sk-"):
        api_key = f"Basic {api_key}"
    
    # Test configurations
    test_cases = [
        {
            "name": "Copilot endpoint (GPT-5-mini)",
            "endpoint": "https://ccproxy.emottet.com/copilot/v1",
            "model": "gpt-5-mini"
        },
        {
            "name": "Copilot endpoint (Claude Sonnet 4.5)",
            "endpoint": "https://ccproxy.emottet.com/copilot/v1",
            "model": "claude-sonnet-4.5"
        },
        {
            "name": "Copilot endpoint (Claude Haiku 4.5)",
            "endpoint": "https://ccproxy.emottet.com/copilot/v1",
            "model": "claude-haiku-4.5"
        },
        {
            "name": "Claude endpoint (Claude Sonnet 4.5)",
            "endpoint": "https://ccproxy.emottet.com/claude/v1",
            "model": "claude-sonnet-4-5-20250929"
        },
        {
            "name": "Codex endpoint (GPT-5-mini)",
            "endpoint": "https://ccproxy.emottet.com/codex/v1",
            "model": "gpt-5-mini"
        },
        {
            "name": "Codex endpoint (Claude Sonnet 4.5)",
            "endpoint": "https://ccproxy.emottet.com/codex/v1",
            "model": "claude-sonnet-4.5"
        },
    ]
    
    results = []
    
    print("=" * 80)
    print("Function Calling Compatibility Test for CrewAI Endpoints")
    print("=" * 80)
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}\n")
        
        result = test_function_calling(
            endpoint_url=test_case["endpoint"],
            model=test_case["model"],
            api_key=api_key
        )
        
        result["test_case"] = test_case["name"]
        results.append(result)
        
        # Print result
        print(f"\n📊 Result: {result.get('message', 'No message')}")
        
        if result.get("tool_calls"):
            print(f"\n🎯 Tool calls detected:")
            for tool_call in result["tool_calls"]:
                print(f"   - Function: {tool_call.get('function', {}).get('name')}")
                print(f"     Arguments: {tool_call.get('function', {}).get('arguments', 'N/A')}")
        
        if result.get("error"):
            print(f"\n⚠️ Error: {result['error']}")
        
        if result.get("content"):
            print(f"\n💬 Response content (first 200 chars):")
            print(f"   {result['content']}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    compatible = []
    incompatible = []
    uncertain = []
    
    for result in results:
        name = result["test_case"]
        support = result.get("supports_function_calling")
        
        if support is True:
            compatible.append(name)
            status = "✅ COMPATIBLE"
        elif support == "uncertain":
            uncertain.append(name)
            status = "⚠️ UNCERTAIN (may be compatible)"
        else:
            incompatible.append(name)
            status = "❌ INCOMPATIBLE"
        
        print(f"{status}: {name}")
    
    print(f"\n{'='*80}")
    print(f"Compatible: {len(compatible)}/{len(results)}")
    print(f"Uncertain: {len(uncertain)}/{len(results)}")
    print(f"Incompatible: {len(incompatible)}/{len(results)}")
    print(f"{'='*80}\n")
    
    if incompatible:
        print("⚠️ WARNING: The following endpoints DO NOT support function calling:")
        for name in incompatible:
            print(f"   - {name}")
        print("\n❌ These endpoints CANNOT be used with CrewAI tools or MCP integration!")
        print("   Use /copilot/v1 or /claude/v1 endpoints that support function calling.\n")
    
    if compatible:
        print("✅ RECOMMENDATION: Use these endpoints for CrewAI agents with tools:")
        for name in compatible:
            print(f"   - {name}")
    
    if uncertain:
        print("\n⚠️ These endpoints need further testing:")
        for name in uncertain:
            print(f"   - {name}")
        print("   They may support function calling but didn't use it for the test query.")
    
    # Exit code
    if not compatible and not uncertain:
        print("\n❌ CRITICAL: No compatible endpoints found!")
        sys.exit(1)
    elif incompatible:
        print("\n⚠️ Some endpoints are incompatible. Review your configuration.")
        sys.exit(0)
    else:
        print("\n✅ All tested endpoints appear compatible with function calling.")
        sys.exit(0)


if __name__ == "__main__":
    main()
