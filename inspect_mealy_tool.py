#!/usr/bin/env python3
"""Inspect a specific Mealy MCP tool."""

import os
import sys
import json
from dotenv import load_dotenv
from mcp_auth_wrapper import MetaMCPAdapter


def main():
    """Inspect tool details."""
    load_dotenv()

    mcp_api_key = os.getenv("MCP_API_KEY", "")
    food_server_url = os.getenv("FOOD_MCP_SERVER_URL", "")

    if not mcp_api_key or not food_server_url:
        print("❌ Missing MCP_API_KEY or FOOD_MCP_SERVER_URL", file=sys.stderr)
        sys.exit(1)

    tool_name = "mealy__create_mealplan_bulk"

    print(f"🔍 Inspecting tool: {tool_name}\n", file=sys.stderr)

    try:
        adapter = MetaMCPAdapter(food_server_url, mcp_api_key, connect_timeout=30)
        adapter.start()

        # Find the tool
        tool = next((t for t in adapter.tools if t.name == tool_name), None)

        if not tool:
            print(f"❌ Tool '{tool_name}' not found", file=sys.stderr)
            sys.exit(1)

        print(f"✅ Found tool: {tool.name}\n", file=sys.stderr)

        if hasattr(tool, 'description'):
            print(f"Description:\n{tool.description}\n", file=sys.stderr)

        # Try to get the input schema
        if hasattr(tool, 'inputSchema') or hasattr(tool, 'input_schema'):
            schema = getattr(tool, 'inputSchema', None) or getattr(tool, 'input_schema', None)
            print(f"Input Schema:\n{json.dumps(schema, indent=2)}\n", file=sys.stderr)
        elif hasattr(tool, 'args_schema'):
            print(f"Args Schema:\n{json.dumps(tool.args_schema, indent=2, default=str)}\n", file=sys.stderr)

        adapter.stop()

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
