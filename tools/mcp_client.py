"""MCP Server client for accessing Intervals.icu and other data sources."""
import os
import requests
from typing import Dict, Any, Optional
import json


class MCPClient:
    """Client for interacting with the MCP server."""
    
    def __init__(self):
        self.server_url = os.getenv("MCP_SERVER_URL")
        if not self.server_url:
            raise ValueError("MCP_SERVER_URL environment variable not set")
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server using the JSON-RPC 2.0 protocol.
        
        Args:
            tool_name: Name of the tool to call (e.g., "IntervalsIcu__get_activity_details")
            arguments: Dictionary of arguments for the tool
            
        Returns:
            Result from the tool call
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 1
        }
        
        try:
            response = requests.post(
                self.server_url,
                json=payload,
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                return {"error": result["error"]}
            
            return result.get("result", {})
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


# Singleton instance
_mcp_client = None

def get_mcp_client() -> MCPClient:
    """Get or create the MCP client singleton."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client
