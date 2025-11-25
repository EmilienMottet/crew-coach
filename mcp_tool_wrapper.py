"""
Wrapper for MCP tools to validate and fix malformed inputs.

This module provides a defensive layer that intercepts MCP tool calls
and ensures the input format is correct before passing to the actual tool.

The main challenge is that CrewAI validates tool inputs BEFORE calling the tool's
_run() method. To work around this, we create a custom BaseTool wrapper that:
1. Accepts any input type (bypasses CrewAI's schema validation)
2. Validates and fixes the input manually in _run()
3. Delegates to the original MCP tool
"""

import json
import sys
from typing import Any, Callable, Dict, List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, create_model

# Global counter for tracking tool calls
_tool_call_counter: Dict[str, int] = {}


def reset_tool_call_counter() -> None:
    """Reset the global tool call counter."""
    global _tool_call_counter
    _tool_call_counter = {}


def get_tool_call_count(tool_name: Optional[str] = None) -> int:
    """
    Get the number of times a tool was called.

    Args:
        tool_name: Optional specific tool name. If None, returns total calls.

    Returns:
        Number of calls for the tool (or total if tool_name is None)
    """
    if tool_name is None:
        return sum(_tool_call_counter.values())
    return _tool_call_counter.get(tool_name, 0)


def get_tool_call_summary() -> Dict[str, int]:
    """Get a summary of all tool calls."""
    return dict(_tool_call_counter)


def validate_tool_input(tool_input: Any) -> Dict[str, Any]:
    """
    Validate and fix malformed tool inputs from LLM.

    Common issues:
    - LLM passes a list instead of a dict
    - LLM includes multiple parameter sets
    - LLM includes mocked responses in the input

    Args:
        tool_input: Raw input from LLM (could be dict, list, str, etc.)

    Returns:
        Valid dictionary of parameters

    Raises:
        ValueError: If input cannot be salvaged
    """
    # If it's already a dict, return as-is
    if isinstance(tool_input, dict):
        # Check if it looks like a mocked response (has "status" and "data")
        if "status" in tool_input and "data" in tool_input:
            raise ValueError(
                "Tool input appears to be a mocked response. "
                "Pass only the parameters, not the expected response."
            )
        return tool_input

    # If it's a list, try to extract the first valid parameter dict
    if isinstance(tool_input, list):
        print(
            f"⚠️  Warning: Tool received a list instead of dict. "
            f"Attempting to extract parameters from first element...",
            file=sys.stderr
        )

        # Find the first dict that looks like parameters (not a response)
        for item in tool_input:
            if isinstance(item, dict):
                # Skip items that look like mocked responses
                if "status" in item and "data" in item:
                    continue
                # This looks like a parameter dict
                print(f"✅ Extracted parameters: {item}", file=sys.stderr)
                return item

        raise ValueError(
            f"Could not extract valid parameters from list: {tool_input}. "
            f"Please pass a single dictionary with parameters only."
        )

    # If it's a string, try to parse as JSON
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
            return validate_tool_input(parsed)  # Recursive call
        except json.JSONDecodeError:
            raise ValueError(f"Tool input is a string but not valid JSON: {tool_input}")

    # Unknown type
    raise ValueError(
        f"Tool input must be a dictionary, got {type(tool_input).__name__}: {tool_input}"
    )


class PermissiveToolWrapper(BaseTool):
    """
    A permissive tool wrapper that accepts any input and validates it manually.

    This bypasses CrewAI's strict schema validation which rejects list inputs.
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    original_tool: Any = Field(description="Original MCP tool to delegate to")

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> Any:
        """
        Run the tool with manual input validation.

        This method receives kwargs from CrewAI. We need to handle cases where:
        - The LLM passes a list instead of a dict
        - The list contains multiple parameter sets
        - The list contains mocked responses
        """
        tool_name = getattr(self.original_tool, 'name', self.name)

        # If we received multiple kwargs, try to extract the real parameters
        # Sometimes CrewAI passes the entire input as a single kwarg
        if len(kwargs) == 1:
            key, value = list(kwargs.items())[0]
            # If the single kwarg looks like it contains the actual input, use it
            if isinstance(value, (dict, list)):
                try:
                    validated = validate_tool_input(value)
                    print(f"✅ Successfully validated input for {tool_name}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️  Validation warning for {tool_name}: {e}", file=sys.stderr)
                    print(f"   Attempting to use original input...", file=sys.stderr)
                    validated = kwargs
            else:
                validated = kwargs
        else:
            # Multiple kwargs - try to validate as-is
            try:
                validated = validate_tool_input(kwargs)
            except Exception as e:
                print(f"⚠️  Could not validate kwargs for {tool_name}: {e}", file=sys.stderr)
                validated = kwargs

        # Call the original tool with validated input
        original_run = getattr(self.original_tool, '_run', None) or getattr(self.original_tool, 'run', None)
        if original_run:
            return original_run(validated)
        else:
            raise RuntimeError(f"Tool {tool_name} has no _run or run method")


def wrap_mcp_tool(tool: Any) -> Any:
    """
    Wrap an MCP tool to add input validation.

    This wrapper intercepts the tool execution BEFORE CrewAI's validation
    by replacing the args_schema with a permissive one that accepts any input.

    Args:
        tool: MCP tool object with a `_run` or `run` method

    Returns:
        Wrapped tool with validation
    """
    # Store original run method
    original_run = getattr(tool, '_run', None) or getattr(tool, 'run', None)

    if not original_run:
        # Tool doesn't have a run method, return as-is
        return tool

    # Replace the args_schema with a permissive model that accepts Any
    # This bypasses CrewAI's strict validation before _run is called
    tool_name = getattr(tool, 'name', 'unknown')

    # Create a permissive schema that accepts any input as 'tool_input'
    PermissiveSchema = create_model(
        f"{tool_name.replace('-', '_').replace('__', '_').title()}PermissiveInput",
        tool_input=(Dict[str, Any], Field(default_factory=dict, description="Tool input (any format accepted)"))
    )

    # Store original schema for reference
    original_schema = getattr(tool, 'args_schema', None)
    tool._original_args_schema = original_schema
    tool.args_schema = PermissiveSchema

    def validated_run(*args, **kwargs) -> Any:
        """Run the tool with input validation."""
        tool_name = tool.name if hasattr(tool, 'name') else 'unknown'

        # Increment tool call counter
        global _tool_call_counter
        _tool_call_counter[tool_name] = _tool_call_counter.get(tool_name, 0) + 1

        # Log that the tool is being called
        print(f"\n🔧 MCP Tool Call: {tool_name}", file=sys.stderr)
        print(f"   Args: {args[:1] if args else 'none'}", file=sys.stderr)
        print(f"   Kwargs keys: {list(kwargs.keys()) if kwargs else 'none'}", file=sys.stderr)

        # Handle permissive schema input - the input comes in kwargs['tool_input']
        if 'tool_input' in kwargs and kwargs['tool_input'] is not None:
            raw_input = kwargs.pop('tool_input')
            print(f"   Raw tool_input type: {type(raw_input).__name__}", file=sys.stderr)
            try:
                validated_input = validate_tool_input(raw_input)
                print(f"   ✅ Validated input: {list(validated_input.keys()) if isinstance(validated_input, dict) else validated_input}", file=sys.stderr)
                # Call original method with validated input as kwargs
                result = original_run(**validated_input)
            except Exception as e:
                print(
                    f"❌ Tool input validation failed: {e}\n"
                    f"   Tool: {tool_name}\n"
                    f"   Raw input: {raw_input}",
                    file=sys.stderr
                )
                raise
        # If first arg looks like tool input, validate it
        elif args and len(args) > 0:
            try:
                validated_input = validate_tool_input(args[0])
                # Call original method with validated input
                result = original_run(**validated_input) if isinstance(validated_input, dict) else original_run(validated_input, *args[1:], **kwargs)
            except Exception as e:
                print(
                    f"❌ Tool input validation failed: {e}\n"
                    f"   Tool: {tool_name}\n"
                    f"   Raw input: {args[0]}",
                    file=sys.stderr
                )
                raise
        else:
            # No special input, call with original args/kwargs
            result = original_run(*args, **kwargs)

        # Log the result summary
        result_str = str(result)
        result_len = len(result_str)
        
        # Truncate if too large (100KB limit to avoid Nginx 413 errors)
        max_len = 100000
        if result_len > max_len:
            print(f"⚠️  Warning: Tool output too large ({result_len} chars), truncating to {max_len}...", file=sys.stderr)
            truncated_msg = f"... [TRUNCATED due to size: {result_len} > {max_len} chars]"
            
            if isinstance(result, str):
                result = result[:max_len] + truncated_msg
            elif isinstance(result, dict) or isinstance(result, list):
                # If it's a structured object, we can't easily truncate it while keeping it valid
                # So we convert to string, truncate, and return that
                # This might break tools expecting strict types, but it's better than a crash
                result = str(result)[:max_len] + truncated_msg
                
            result_str = str(result)

        result_summary = result_str[:200] if result else "None"
        print(f"   ✅ Result: {result_summary}{'...' if len(result_str) > 200 else ''}\n", file=sys.stderr)

        return result

    # Replace run method with validated version
    if hasattr(tool, '_run'):
        tool._run = validated_run
    elif hasattr(tool, 'run'):
        tool.run = validated_run

    # Also wrap the __call__ method if it exists (CrewAI may call this directly)
    if hasattr(tool, '__call__') and callable(tool):
        original_call = tool.__call__

        def validated_call(*args, **kwargs) -> Any:
            """Call the tool with input validation."""
            # If first arg is the tool input, validate it
            if args and len(args) > 0 and not callable(args[0]):
                try:
                    validated_input = validate_tool_input(args[0])
                    args = (validated_input,) + args[1:]
                except Exception as e:
                    print(
                        f"❌ Tool input validation failed in __call__: {e}\n"
                        f"   Tool: {tool.name if hasattr(tool, 'name') else 'unknown'}\n"
                        f"   Raw input: {args[0]}",
                        file=sys.stderr
                    )
                    # Try to proceed with original call anyway
                    pass

            return original_call(*args, **kwargs)

        # Don't replace __call__ on the class, create a wrapper instance
        # This is tricky, so we skip it for now

    return tool


def wrap_mcp_tools(tools: List[Any]) -> List[Any]:
    """
    Wrap a list of MCP tools with input validation.

    Args:
        tools: List of MCP tool objects

    Returns:
        List of wrapped tools
    """
    return [wrap_mcp_tool(tool) for tool in tools]
