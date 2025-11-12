"""
Test real tool calling with a prompt that requires fetching data.

This simulates the Music Agent receiving activity data and needing to
call Spotify MCP tools to get recently played tracks.
"""
import sys
import json
from datetime import datetime, timedelta
from agents.music_agent import create_music_agent

# Initialize auth patches
import llm_auth_init

# Connect to MCP servers and get tools
from mcp_utils import build_mcp_references, load_catalog_tool_names
from mcp_tool_wrapper import wrap_mcp_tool, reset_tool_call_counter, get_tool_call_summary
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

print("🔗 Connecting to MCP servers...\n")

# Connect to Spotify MCP
mcp_server_url = os.getenv("MCP_SERVER_URL")
if not mcp_server_url:
    print("❌ MCP_SERVER_URL not set", file=sys.stderr)
    sys.exit(1)

spotify_tool_names = load_catalog_tool_names(["spotify"])
print(f"🎵 Found {len(spotify_tool_names)} Spotify tools")

# Create the Music Agent
music_agent = create_music_agent()

print(f"\n🎵 Music Agent created:")
print(f"   tools parameter: {len(music_agent.tools)}")
print(f"   agent.tools: {len(music_agent.tools)}")
print(f"   Agent LLM type: {type(music_agent.llm)}")

# Create a test prompt that REQUIRES calling the spotify__getRecentlyPlayed tool
# Activity happened on 2024-01-15 at 11:30 AM CET
activity_start = "2024-01-15T11:30:00+01:00"
activity_end = "2024-01-15T12:45:00+01:00"  # 75 minutes run

test_prompt = f"""You are analyzing a running activity to add music information.

Activity details:
- Start time: {activity_start}
- End time: {activity_end}
- Duration: 75 minutes
- Type: Running

Your task:
1. Call the spotify__getRecentlyPlayed tool to get tracks played during this time window
2. Extract up to 5 tracks from the results
3. Return them in JSON format:

{{
  "tracks": ["Artist - Song", "Artist2 - Song2", ...]
}}

IMPORTANT: You MUST call the spotify__getRecentlyPlayed tool to get real data.
Do NOT make up or hallucinate track names.
"""

print(f"\n📝 Test prompt:\n{test_prompt}\n")

# Call the LLM directly with the Music Agent's LLM
print("🚀 Calling Music Agent LLM with tool-requiring prompt...\n")

try:
    # Reset tool call counter
    reset_tool_call_counter()
    
    # Call the LLM
    response = music_agent.llm.call(
        messages=test_prompt,
        tools=None,  # Will be extracted from agent
        from_agent=music_agent,
    )
    
    print(f"\n✅ LLM Response:\n{response}\n")
    
    # Check if tools were called
    tool_summary = get_tool_call_summary()
    if tool_summary:
        print(f"\n🎉 SUCCESS! Tools were called:")
        for tool_name, count in tool_summary.items():
            print(f"   - {tool_name}: {count} calls")
    else:
        print(f"\n⚠️  WARNING: No tools were called. LLM may have hallucinated the response.")
        print(f"   This might be due to:")
        print(f"   1. LLM ignored the tool calling capability")
        print(f"   2. Tool execution failed")
        print(f"   3. Fallback to CrewAI LLM.call() which doesn't support tools")

except Exception as e:
    print(f"\n❌ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
