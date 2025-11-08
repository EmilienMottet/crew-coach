#!/usr/bin/env python3
"""Test script to validate MCP server connectivity and tool availability."""

import os
import sys
from dotenv import load_dotenv
from mcp_auth_wrapper import MetaMCPAdapter


def test_mcp_connectivity():
    """Test if MCP servers are reachable and return valid tool definitions."""
    load_dotenv()

    mcp_api_key = os.getenv("MCP_API_KEY", "")

    if not mcp_api_key:
        print("❌ Error: MCP_API_KEY environment variable is not set", file=sys.stderr)
        return False

    # Define MCP server URLs (matching crew.py configuration)
    mcp_servers = {
        "Strava": os.getenv("STRAVA_MCP_SERVER_URL", ""),
        "Intervals.icu": os.getenv("INTERVALS_MCP_SERVER_URL", ""),
        "Music": os.getenv("MUSIC_MCP_SERVER_URL", ""),
        "Meteo": os.getenv("METEO_MCP_SERVER_URL", ""),
        "Toolbox": os.getenv("TOOLBOX_MCP_SERVER_URL", ""),
    }

    # Filter out empty URLs
    active_servers = {name: url for name, url in mcp_servers.items() if url}

    if not active_servers:
        print("❌ Error: No MCP server URLs configured", file=sys.stderr)
        print("   Please set at least one of: STRAVA_MCP_SERVER_URL, INTERVALS_MCP_SERVER_URL, etc.", file=sys.stderr)
        return False

    print(f"🔗 Connecting to {len(active_servers)} MCP servers...\n", file=sys.stderr)

    mcp_adapters = []
    total_tools = 0

    for server_name, server_url in active_servers.items():
        try:
            print(f"   Connecting to {server_name}...", file=sys.stderr)
            adapter = MetaMCPAdapter(server_url, mcp_api_key, connect_timeout=30)
            adapter.start()
            mcp_adapters.append(adapter)

            tool_count = len(adapter.tools)
            total_tools += tool_count

            print(f"   ✅ {server_name}: {tool_count} tools discovered", file=sys.stderr)

            # Show first few tools for this server
            if tool_count > 0:
                print(f"      Tools:", file=sys.stderr)
                for i, tool in enumerate(adapter.tools[:5]):
                    print(f"      - {tool.name}", file=sys.stderr)
                if tool_count > 5:
                    print(f"      ... and {tool_count - 5} more", file=sys.stderr)
            print("", file=sys.stderr)

        except Exception as e:
            print(f"   ❌ {server_name}: Connection failed - {e}\n", file=sys.stderr)
            # Continue testing other servers

    # Cleanup
    for adapter in mcp_adapters:
        try:
            adapter.stop()
        except Exception:
            pass

    print(f"✅ MCP connection complete! Total tools discovered: {total_tools}\n", file=sys.stderr)

    return total_tools > 0


def main():
    """Main entry point."""
    print("\n🔧 MCP Connectivity Test\n", file=sys.stderr)

    success = test_mcp_connectivity()

    if success:
        print("✅ MCP connectivity test passed!\n", file=sys.stderr)
        sys.exit(0)
    else:
        print("❌ MCP connectivity test failed!\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
