"""Custom MCP adapter for CrewAI that properly handles MetaMCP API key authentication.

This module provides a custom MCP adapter that ensures the API key is included in ALL
HTTP requests (both initial connection and subsequent session messages), solving the
401 Unauthorized error on /message endpoints.

The solution uses a custom httpx.AsyncClient that intercepts all requests and adds
the api_key query parameter automatically.
"""

import os
import asyncio
import threading
from contextlib import AsyncExitStack
from typing import Any, List
from functools import partial
import httpx

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcpadapt.crewai_adapter import CrewAIAdapter
from crewai.tools import BaseTool


class APIKeyHTTPClient(httpx.AsyncClient):
    """Custom HTTP client that adds API key to ALL requests."""

    def __init__(self, api_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._api_key = api_key

    async def request(self, method: str, url, **kwargs) -> httpx.Response:
        """Override request to add api_key to ALL URLs."""
        url_str = str(url)
        if "api_key=" not in url_str:
            separator = "&" if "?" in url_str else "?"
            url_str = f"{url_str}{separator}api_key={self._api_key}"
        return await super().request(method, url_str, **kwargs)


class MetaMCPAdapter:
    """
    Custom MCP adapter for CrewAI that handles MetaMCP authentication correctly.

    This adapter creates a custom HTTP client that adds the API key to all requests,
    solving the authentication propagation issue with MetaMCP servers.

    Usage:
        adapter = MetaMCPAdapter(mcp_url, api_key)
        adapter.start()
        tools = adapter.tools  # Get CrewAI-compatible tools
        # ... use tools with CrewAI agents ...
        adapter.stop()
    """

    def __init__(self, mcp_url: str, api_key: str, connect_timeout: int = 30):
        """
        Initialize the MetaMCP adapter.

        Args:
            mcp_url: Base MCP server URL (without api_key parameter)
            api_key: MetaMCP API key
            connect_timeout: Connection timeout in seconds
        """
        # Clean URL but preserve the path (e.g., /sse or /mcp)
        if "?" in mcp_url:
            self.mcp_url = mcp_url.split("?")[0]
        else:
            self.mcp_url = mcp_url.rstrip("/")

        self.api_key = api_key
        self.connect_timeout = connect_timeout

        # Async resources
        self.loop = None
        self.thread = None
        self.session: ClientSession | None = None
        self.mcp_tools: List[Any] = []
        self.ready = threading.Event()
        self.crewai_adapter = CrewAIAdapter()

    def _create_auth_client(self, **kwargs) -> APIKeyHTTPClient:
        """Create HTTP client factory for MCP."""
        return APIKeyHTTPClient(self.api_key, **kwargs)

    def _run_loop(self):
        """Run the async event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def setup():
            async with AsyncExitStack() as stack:
                # Connect to MCP server with custom HTTP client using streamable HTTP
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(
                        url=f"{self.mcp_url}?api_key={self.api_key}",
                        httpx_client_factory=self._create_auth_client,
                        timeout=float(self.connect_timeout),
                        terminate_on_close=True
                    )
                )

                # Create and initialize session
                self.session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await self.session.initialize()

                # List available tools
                result = await self.session.list_tools()
                self.mcp_tools = result.tools

                self.ready.set()  # Signal initialization complete
                await asyncio.Event().wait()  # Keep alive until stopped

        try:
            self.loop.run_until_complete(setup())
        except asyncio.CancelledError:
            pass

    def start(self):
        """Start the MCP adapter and connect to the server."""
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        # Wait for initialization with timeout
        if not self.ready.wait(timeout=self.connect_timeout):
            raise TimeoutError(f"Couldn't connect to MCP server after {self.connect_timeout} seconds")

    def stop(self):
        """Stop the MCP adapter and cleanup resources."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self._shutdown()))

    async def _shutdown(self):
        """Async shutdown helper."""
        if self.loop:
            for task in asyncio.all_tasks(self.loop):
                task.cancel()

    @property
    def tools(self) -> List[BaseTool]:
        """
        Get CrewAI-compatible tools from the MCP server.

        Returns:
            List of BaseTool instances that can be used with CrewAI agents
        """
        if not self.session or not self.mcp_tools:
            raise RuntimeError("MCP adapter not initialized. Call start() first.")

        # Convert MCP tools to CrewAI tools using the official adapter
        # but handle errors gracefully
        crewai_tools = []
        for mcp_tool in self.mcp_tools:
            try:
                # Create a sync wrapper for the tool call
                def make_tool_func(tool_name):
                    def tool_func(arguments=None):
                        if not self.session or not self.loop:
                            raise RuntimeError("MCP adapter not initialized")

                        # Run the async tool call in the event loop
                        future = asyncio.run_coroutine_threadsafe(
                            self.session.call_tool(tool_name, arguments),
                            self.loop
                        )
                        return future.result()
                    return tool_func

                # Use CrewAIAdapter to convert the MCP tool with proper schema
                tool = self.crewai_adapter.adapt(make_tool_func(mcp_tool.name), mcp_tool)
                crewai_tools.append(tool)
            except Exception as e:
                # If schema conversion fails (circular refs, etc.), fall back to simple tool
                import sys
                print(f"Warning: Could not convert tool {mcp_tool.name} with schema, using simple wrapper: {e}", file=sys.stderr)
                tool = self._create_simple_tool(mcp_tool)
                crewai_tools.append(tool)

        return crewai_tools

    def _create_simple_tool(self, mcp_tool) -> BaseTool:
        """Create a simple CrewAI tool without schema validation (fallback)."""
        from crewai.tools import tool

        # Create a sync wrapper for the async MCP tool
        def tool_func(**kwargs):
            """Execute MCP tool synchronously."""
            if not self.session or not self.loop:
                raise RuntimeError("MCP adapter not initialized")

            # Run the async tool call in the event loop
            future = asyncio.run_coroutine_threadsafe(
                self.session.call_tool(mcp_tool.name, kwargs or None),
                self.loop
            )
            result = future.result()

            # Extract content from result
            if hasattr(result, 'content') and result.content:
                return str(result.content[0].text if result.content else "")
            return str(result)

        # Set tool metadata
        tool_func.__name__ = mcp_tool.name.replace("-", "_").replace("__", "_")

        # Enhanced description with schema info
        schema_desc = ""
        if mcp_tool.inputSchema and "properties" in mcp_tool.inputSchema:
            props = mcp_tool.inputSchema["properties"]
            required = mcp_tool.inputSchema.get("required", [])
            params_desc = []
            for prop_name, prop_schema in props.items():
                req_marker = " (required)" if prop_name in required else ""
                prop_type = prop_schema.get("type", "unknown")
                prop_desc = prop_schema.get("description", "")
                params_desc.append(f"  - {prop_name} ({prop_type}){req_marker}: {prop_desc}")
            schema_desc = "\n\nParameters:\n" + "\n".join(params_desc)

        tool_func.__doc__ = (mcp_tool.description or f"MCP tool: {mcp_tool.name}") + schema_desc

        return tool(tool_func)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self.tools

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def get_metamcp_adapter_from_env() -> MetaMCPAdapter:
    """
    Create a MetaMCP adapter from environment variables.

    Requires:
        - MCP_SERVER_URL: Base MCP server URL
        - MCP_API_KEY: MetaMCP API key

    Returns:
        Initialized MetaMCPAdapter instance
    """
    mcp_url = os.getenv("MCP_SERVER_URL", "")
    api_key = os.getenv("MCP_API_KEY", "")

    if not mcp_url:
        raise ValueError("MCP_SERVER_URL environment variable is required")
    if not api_key:
        raise ValueError("MCP_API_KEY environment variable is required")

    return MetaMCPAdapter(mcp_url, api_key)
