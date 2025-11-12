#!/usr/bin/env python3
"""Test actual MCP tool structure."""

import sys
import json
from dotenv import load_dotenv

load_dotenv()

# Get real MCP tools
from mcp_tool_wrapper import create_mcp_client, discover_tools_from_server, WrappedBaseTool

def test_mcp_tool_structure():
    """Test the structure of actual MCP tools."""
    print("🔗 Connecting to Spotify MCP server...\n")
    
    # Connect to Spotify
    music_server_url = "https://mcp.emottet.com/metamcp/spotify/spotify/mcp?api_key=..."
    client = create_mcp_client(music_server_url)
    
    raw_tools = discover_tools_from_server(client, "Spotify")
    print(f"✅ Found {len(raw_tools)} Spotify tools\n")
    
    # Wrap first tool
    if raw_tools:
        first_raw = raw_tools[0]
        print(f"🔍 First raw tool: {first_raw.name}\n")
        print(f"   Input schema: {json.dumps(first_raw.inputSchema, indent=2)[:300]}...\n")
        
        # Wrap it
        wrapped = WrappedBaseTool(raw_tool=first_raw)
        print(f"🎁 Wrapped tool: {wrapped.name}\n")
        print(f"   Attributes: {[a for a in dir(wrapped) if not a.startswith('_')]}\n")
        
        # Check for args_schema
        if hasattr(wrapped, 'args_schema'):
            schema = wrapped.args_schema
            print(f"✅ Has args_schema: {type(schema)}\n")
            if schema:
                try:
                    json_schema = schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema
                    print(f"   Schema: {json.dumps(json_schema, indent=2)[:500]}...\n")
                except Exception as e:
                    print(f"   ⚠️  Failed to convert schema: {e}\n")
        
        # Try to_structured_tool
        if hasattr(wrapped, 'to_structured_tool'):
            try:
                structured = wrapped.to_structured_tool()
                print(f"✅ to_structured_tool() works!\n")
                print(f"   Type: {type(structured)}\n")
                print(f"   Attributes: {[a for a in dir(structured) if not a.startswith('_')][:20]}\n")
            except Exception as e:
                print(f"❌ to_structured_tool() failed: {e}\n")
                import traceback
                traceback.print_exc()
        
        return True
    
    return False

if __name__ == "__main__":
    success = test_mcp_tool_structure()
    sys.exit(0 if success else 1)
