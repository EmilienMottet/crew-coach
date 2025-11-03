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

    if not mcp_url:
        print("❌ Error: MCP_SERVER_URL environment variable is not set", file=sys.stderr)
        return False

    print(f"🔍 Testing MCP connectivity to: {mcp_url[:50]}...", file=sys.stderr)

    try:
        # Test basic connectivity
        response = requests.get(mcp_url, timeout=10)

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
                    print("⚠️  Warning: No tools found in MCP response", file=sys.stderr)
                    print(f"Response structure: {json.dumps(data, indent=2)[:300]}", file=sys.stderr)

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
