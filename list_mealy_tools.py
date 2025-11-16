#!/usr/bin/env python3
"""List available Mealy MCP tools from the Food server."""

import os
import sys
from dotenv import load_dotenv
from mcp_auth_wrapper import MetaMCPAdapter


def main():
    """List Mealy tools."""
    load_dotenv()

    mcp_api_key = os.getenv("MCP_API_KEY", "")
    food_server_url = os.getenv("FOOD_MCP_SERVER_URL", "")

    if not mcp_api_key or not food_server_url:
        print("❌ Missing MCP_API_KEY or FOOD_MCP_SERVER_URL", file=sys.stderr)
        sys.exit(1)

    print(f"🔗 Connecting to Food MCP server...\n", file=sys.stderr)

    try:
        adapter = MetaMCPAdapter(food_server_url, mcp_api_key, connect_timeout=30)
        adapter.start()

        print(f"✅ Connected! Total tools: {len(adapter.tools)}\n", file=sys.stderr)

        # Filter Mealy tools
        mealy_tools = [t for t in adapter.tools if "mealy" in t.name.lower()]

        if mealy_tools:
            print(f"🍽️  Found {len(mealy_tools)} Mealy tools:\n", file=sys.stderr)
            for tool in mealy_tools:
                print(f"   📋 {tool.name}", file=sys.stderr)
                if hasattr(tool, 'description') and tool.description:
                    desc = tool.description.split('\n')[0][:100]  # First line, max 100 chars
                    print(f"      {desc}", file=sys.stderr)
                print("", file=sys.stderr)
        else:
            print("⚠️  No Mealy tools found", file=sys.stderr)

        adapter.stop()

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
