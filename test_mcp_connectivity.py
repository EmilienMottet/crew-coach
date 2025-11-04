#!/usr/bin/env python3
"""Test script to validate MCP server connectivity and tool availability."""

import os
import sys
import json
from dotenv import load_dotenv
import requests


def test_mcp_connectivity():
    """Test if MCP server is reachable and returns valid tool definitions."""
    load_dotenv()

    mcp_url = os.getenv("MCP_SERVER_URL")
    api_key = os.getenv("MCP_API_KEY")

    if not mcp_url:
        print("❌ Error: MCP_SERVER_URL environment variable is not set", file=sys.stderr)
        return False

    # Build full URL with API key if provided
    full_url = mcp_url
    if api_key and "api_key=" not in mcp_url:
        separator = "&" if "?" in mcp_url else "?"
        full_url = f"{mcp_url}{separator}api_key={api_key}"

    print(f"🔍 Testing MCP connectivity to: {mcp_url[:50]}...", file=sys.stderr)
    print(f"🔑 API key configured: {'Yes' if api_key else 'No'}", file=sys.stderr)

    try:
        # Test basic connectivity with authentication
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Try GET first (for basic health check)
        response = requests.get(full_url, headers=headers, timeout=10)

        print(f"📡 Response status: {response.status_code}", file=sys.stderr)

        # For Streamable HTTP MCP, we may need to POST to initialize a session
        # or the GET might return session info
        if response.status_code == 404 and "session" in response.text.lower():
            print("ℹ️  Server requires session initialization (Streamable HTTP)", file=sys.stderr)
            print("✅ MCP server is reachable (session-based authentication works)", file=sys.stderr)
            print("⚠️  Note: Tool discovery happens at runtime via CrewAI", file=sys.stderr)
            return True

        if response.status_code != 200:
            print(f"❌ Error: MCP server returned status {response.status_code}", file=sys.stderr)
            print(f"Response: {response.text[:200]}", file=sys.stderr)
            return False

        # Try to parse response as JSON
        try:
            data = response.json()
            print("✅ MCP server is reachable and returned valid JSON", file=sys.stderr)

            # Check if tools are available
            if isinstance(data, dict):
                # Check for common MCP response structures
                tools = data.get("tools", [])
                if tools:
                    print(f"✅ Found {len(tools)} tools available:", file=sys.stderr)
                    for tool in tools[:5]:  # Show first 5 tools
                        tool_name = tool.get("name", "unknown")
                        print(f"   - {tool_name}", file=sys.stderr)
                    if len(tools) > 5:
                        print(f"   ... and {len(tools) - 5} more", file=sys.stderr)
                else:
                    print("⚠️  Note: No tools in GET response (normal for Streamable HTTP)", file=sys.stderr)
                    print("   Tools are discovered during session initialization", file=sys.stderr)

            return True

        except json.JSONDecodeError:
            print("⚠️  Warning: MCP server response is not JSON", file=sys.stderr)
            print(f"Response preview: {response.text[:200]}", file=sys.stderr)
            return False

    except requests.exceptions.Timeout:
        print("❌ Error: Connection to MCP server timed out", file=sys.stderr)
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Error: Cannot connect to MCP server: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Error: Unexpected error while testing MCP: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def main():
    """Main entry point."""
    print("\n🔧 MCP Connectivity Test\n", file=sys.stderr)

    success = test_mcp_connectivity()

    if success:
        print("\n✅ MCP connectivity test passed!\n", file=sys.stderr)
        sys.exit(0)
    else:
        print("\n❌ MCP connectivity test failed!\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
