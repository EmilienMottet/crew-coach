#!/usr/bin/env python3
"""Test script to understand Intervals.icu tool parameters and ID format."""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from mcp_auth_wrapper import MetaMCPAdapter


def test_intervals_tools():
    """Test Intervals.icu MCP tools to understand expected parameters."""
    load_dotenv()

    mcp_api_key = os.getenv("MCP_API_KEY", "")
    intervals_url = os.getenv("INTERVALS_MCP_SERVER_URL", "")

    if not intervals_url or not mcp_api_key:
        print("❌ Error: INTERVALS_MCP_SERVER_URL or MCP_API_KEY not configured", file=sys.stderr)
        return False

    print("🔗 Connecting to Intervals.icu MCP server...\n", file=sys.stderr)

    try:
        adapter = MetaMCPAdapter(intervals_url, mcp_api_key, connect_timeout=30)
        adapter.start()

        print(f"✅ Connected! {len(adapter.tools)} tools discovered\n", file=sys.stderr)

        # Find the tools we're interested in
        get_activities_tool = None
        get_activity_details_tool = None
        get_activity_intervals_tool = None

        for tool in adapter.tools:
            if tool.name == "IntervalsIcu__get_activities":
                get_activities_tool = tool
            elif tool.name == "IntervalsIcu__get_activity_details":
                get_activity_details_tool = tool
            elif tool.name == "IntervalsIcu__get_activity_intervals":
                get_activity_intervals_tool = tool

        # Print tool descriptions to understand expected parameters
        print("=" * 70)
        print("📋 Tool Information:")
        print("=" * 70)

        if get_activities_tool:
            print("\n1️⃣  IntervalsIcu__get_activities")
            print("-" * 70)
            print(f"{get_activities_tool.description}")

        if get_activity_details_tool:
            print("\n2️⃣  IntervalsIcu__get_activity_details")
            print("-" * 70)
            print(f"{get_activity_details_tool.description}")

        if get_activity_intervals_tool:
            print("\n3️⃣  IntervalsIcu__get_activity_intervals")
            print("-" * 70)
            print(f"{get_activity_intervals_tool.description}")

        print("\n" + "=" * 70)

        # Try calling get_activities to see what IDs are returned
        if get_activities_tool:
            print("\n🔍 Testing get_activities to see recent activities...\n")

            # Get activities from today
            today = datetime.now().strftime("%Y-%m-%d")
            try:
                result = get_activities_tool.func(
                    start_date=today,
                    end_date=today,
                    limit=5
                )

                # Parse result if it's a string
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        pass

                print(f"Result type: {type(result)}")
                print(f"\nResult preview:")
                print(json.dumps(result, indent=2)[:2000])

                # Try to extract ID format if activities are returned
                if isinstance(result, list) and len(result) > 0:
                    print(f"\n✅ Found {len(result)} activities")
                    print(f"First activity ID format: {result[0].get('id', 'N/A')}")
                elif isinstance(result, dict) and "activities" in result:
                    activities = result["activities"]
                    if len(activities) > 0:
                        print(f"\n✅ Found {len(activities)} activities")
                        print(f"First activity ID format: {activities[0].get('id', 'N/A')}")

            except Exception as e:
                print(f"Error calling get_activities: {e}")
                import traceback
                traceback.print_exc()

        # Cleanup
        adapter.stop()
        return True

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


if __name__ == "__main__":
    test_intervals_tools()
