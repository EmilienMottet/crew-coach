"""
Wrapper for MCP tools to validate and fix malformed inputs.

This module provides a defensive layer that intercepts MCP tool calls
and ensures the input format is correct before passing to the actual tool.

The main challenge is that CrewAI validates tool inputs BEFORE calling the tool's
_run() method. To work around this, we create a custom BaseTool wrapper that:
1. Accepts any input type (bypasses CrewAI's schema validation)
2. Validates and fixes the input manually in _run()
3. Delegates to the original MCP tool

Additional optimizations for Passio tools:
- Caches search results to avoid redundant API calls
- Filters response data to reduce token usage
"""

import asyncio
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from passio_nutrition_cache import get_passio_search_cache

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

    # If it's a list, extract all valid parameter dicts
    if isinstance(tool_input, list):
        print(
            f"⚠️  Warning: Tool received a list instead of dict. "
            f"Extracting valid parameter dicts...",
            file=sys.stderr
        )

        # Collect all dicts that look like parameters (not responses)
        valid_items = []
        for item in tool_input:
            if isinstance(item, dict):
                # Skip items that look like mocked responses
                if "status" in item and "data" in item:
                    continue
                # Skip items that look like Hexis results (hallucinated output)
                if "days" in item and isinstance(item["days"], list):
                    continue
                valid_items.append(item)

        if len(valid_items) == 0:
            raise ValueError(
                f"Could not extract valid parameters from list: {tool_input}. "
                f"Please pass a single dictionary with parameters only."
            )
        elif len(valid_items) == 1:
            # Single item - return as dict
            print(f"✅ Extracted single parameter dict: {valid_items[0]}", file=sys.stderr)
            return valid_items[0]
        else:
            # Multiple items - return batch marker for sequential processing
            print(f"✅ Extracted {len(valid_items)} parameter dicts for batch processing", file=sys.stderr)
            return {"__batch_items__": valid_items}

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
            if isinstance(validated, dict):
                return original_run(**validated)
            return original_run(validated)
        else:
            raise RuntimeError(f"Tool {tool_name} has no _run or run method")

    async def _arun(self, **kwargs: Any) -> Any:
        """
        Asynchronously run the tool.
        
        Since the underlying validation and original tool might be sync,
        we offload the execution to a thread to avoid blocking the loop.
        """
        return await asyncio.to_thread(self._run, **kwargs)


# =============================================================================
# Passio Search Cache and Response Filtering
# =============================================================================

# Essential fields to keep from Passio search results
PASSIO_ESSENTIAL_FIELDS = {
    "id", "resultId", "refCode", "displayName", "shortName", "longName",
    "type", "dataOrigin", "scoredName",
    # Nutrition fields (from get_passio_food_details)
    "protein_per_100g", "carbs_per_100g", "fat_per_100g", "calories_per_100g",
    "protein", "carbs", "fat", "calories",
    "nutrients",
}


def filter_passio_food(food: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a single Passio food item to keep only essential fields.

    Reduces token usage by ~70% while preserving all necessary data for
    ingredient validation and Hexis integration.
    """
    if not isinstance(food, dict):
        return food

    filtered = {}
    for key in PASSIO_ESSENTIAL_FIELDS:
        if key in food:
            filtered[key] = food[key]

    # Keep nested nutrients if present
    if "nutrients" in food and isinstance(food["nutrients"], dict):
        filtered["nutrients"] = food["nutrients"]

    return filtered


def filter_passio_response(result: Any) -> Any:
    """Filter Passio API response to reduce token usage.

    Handles both search results (list of foods) and detail results (single food).
    """
    if result is None:
        return result

    # Handle string results (already serialized JSON)
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            filtered = filter_passio_response(parsed)
            return json.dumps(filtered)
        except (json.JSONDecodeError, TypeError):
            return result

    # Handle dict with "foods" key (search result)
    if isinstance(result, dict):
        if "foods" in result and isinstance(result["foods"], list):
            result["foods"] = [filter_passio_food(f) for f in result["foods"]]
        else:
            # Single food item
            result = filter_passio_food(result)
        return result

    # Handle list of foods
    if isinstance(result, list):
        return [filter_passio_food(f) for f in result]

    return result


def extract_passio_data_for_cache(result: Any, query: str) -> Optional[Dict[str, Any]]:
    """Extract data from Passio search result for caching.

    Returns the first food item with essential fields for the search cache.
    """
    foods = None

    # Parse string result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "foods" in parsed:
                foods = parsed["foods"]
            elif isinstance(parsed, list):
                foods = parsed
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(result, dict) and "foods" in result:
        foods = result["foods"]
    elif isinstance(result, list):
        foods = result

    if not foods or len(foods) == 0:
        return None

    # Get the first (best match) food item
    first_food = foods[0]
    if not isinstance(first_food, dict):
        return None

    # Extract essential fields for cache
    cache_data = {
        "passio_food_id": first_food.get("id") or first_food.get("resultId"),
        "passio_ref_code": first_food.get("refCode"),
        "passio_food_name": first_food.get("displayName") or first_food.get("shortName"),
    }

    # Extract nutrition if available (from enriched results or get_passio_food_details)
    if "nutrients" in first_food:
        nutrients = first_food["nutrients"]
        cache_data["protein_per_100g"] = nutrients.get("protein", 0)
        cache_data["carbs_per_100g"] = nutrients.get("carbs", 0)
        cache_data["fat_per_100g"] = nutrients.get("fat", 0)
        cache_data["calories_per_100g"] = nutrients.get("calories", 0)
    elif "protein_per_100g" in first_food:
        cache_data["protein_per_100g"] = first_food.get("protein_per_100g", 0)
        cache_data["carbs_per_100g"] = first_food.get("carbs_per_100g", 0)
        cache_data["fat_per_100g"] = first_food.get("fat_per_100g", 0)
        cache_data["calories_per_100g"] = first_food.get("calories_per_100g", 0)

    return cache_data


def build_cached_passio_response(cached_data: Dict[str, Any]) -> str:
    """Build a Passio-like response from cached data.

    Creates a minimal response that looks like a Passio search result
    but contains only the cached food item.
    """
    food_item = {
        "id": cached_data.get("passio_food_id"),
        "resultId": cached_data.get("passio_food_id"),
        "refCode": cached_data.get("passio_ref_code"),
        "displayName": cached_data.get("passio_food_name"),
        "shortName": cached_data.get("passio_food_name"),
        "type": "cached",
        "dataOrigin": "PASSIO_CACHE",
    }

    # Add nutrition if available
    if cached_data.get("protein_per_100g") is not None:
        food_item["nutrients"] = {
            "protein": cached_data.get("protein_per_100g", 0),
            "carbs": cached_data.get("carbs_per_100g", 0),
            "fat": cached_data.get("fat_per_100g", 0),
            "calories": cached_data.get("calories_per_100g", 0),
        }
        food_item["protein_per_100g"] = cached_data.get("protein_per_100g", 0)
        food_item["carbs_per_100g"] = cached_data.get("carbs_per_100g", 0)
        food_item["fat_per_100g"] = cached_data.get("fat_per_100g", 0)
        food_item["calories_per_100g"] = cached_data.get("calories_per_100g", 0)

    response = {"foods": [food_item]}
    return json.dumps(response)


def wrap_mcp_tool(tool: Any) -> Any:
    """
    Wrap an MCP tool to add input validation and logging.

    This wrapper intercepts the tool execution and:
    1. Logs tool calls for debugging
    2. Handles list inputs that some LLMs incorrectly send
    3. Validates inputs before passing to the original tool

    Note: We NO LONGER replace the args_schema because that was causing
    the tool arguments to be lost when Pydantic validated against the
    permissive schema. The original tool's schema is preserved.

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

    tool_name = getattr(tool, 'name', 'unknown')

    # Store original schema for reference (but do NOT replace it!)
    original_schema = getattr(tool, 'args_schema', None)
    tool._original_args_schema = original_schema
    # NOTE: We no longer set tool.args_schema = PermissiveSchema
    # This was causing the actual arguments to be lost when CrewAI
    # validated {"start_date": "...", "end_date": "..."} against a schema
    # that expected {"tool_input": {...}}, resulting in tool_input={}

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

        # Remove security_context if present (CrewAI internal, not for the tool)
        kwargs.pop('security_context', None)

        # Determine the input to validate
        raw_input = None

        # Case 1: Legacy permissive schema input (tool_input key)
        # This was from the old approach - kept for backwards compatibility
        if 'tool_input' in kwargs and kwargs['tool_input']:
            raw_input = kwargs.pop('tool_input')
            print(f"   Legacy tool_input detected: {type(raw_input).__name__}", file=sys.stderr)
        # Case 2: Direct kwargs (the normal case now that we preserve original schema)
        elif kwargs:
            raw_input = kwargs
            kwargs = {}  # Clear so we don't double-pass
            print(f"   Direct kwargs: {list(raw_input.keys())}", file=sys.stderr)
        # Case 3: Positional args (fallback)
        elif args:
            raw_input = args[0]
            args = args[1:]
            print(f"   Positional arg: {type(raw_input).__name__}", file=sys.stderr)

        if raw_input is not None:
            try:
                validated_input = validate_tool_input(raw_input)

                # =============================================================
                # Passio Search Optimization: Cache + Filtering
                # =============================================================
                is_passio_search = "passio" in tool_name.lower() and "search" in tool_name.lower()

                # Inject default limit for Passio search tools to avoid context overflow
                # The Passio API can return 100+ results (150K+ chars) without a limit
                if is_passio_search:
                    if isinstance(validated_input, dict) and "limit" not in validated_input:
                        default_limit = int(os.getenv("PASSIO_DEFAULT_LIMIT", "5"))
                        validated_input["limit"] = default_limit
                        print(f"   📉 Added default limit={default_limit} to Passio search", file=sys.stderr)

                    # Check cache for Passio search queries
                    query = validated_input.get("query", "") if isinstance(validated_input, dict) else ""
                    if query:
                        search_cache = get_passio_search_cache()
                        cached_result = search_cache.get(query)
                        if cached_result:
                            print(f"   🎯 CACHE HIT for '{query}' - skipping API call", file=sys.stderr)
                            result = build_cached_passio_response(cached_result)
                            # Log the cached result summary
                            result_summary = str(result)[:200]
                            print(f"   ✅ Cached result: {result_summary}...\n", file=sys.stderr)
                            return result

                # Check for batch marker - process multiple items sequentially
                if isinstance(validated_input, dict) and "__batch_items__" in validated_input:
                    batch_items = validated_input["__batch_items__"]
                    print(f"   🔄 Batch processing {len(batch_items)} items...", file=sys.stderr)
                    results = []
                    for i, item in enumerate(batch_items, 1):
                        print(f"   📝 Processing item {i}/{len(batch_items)}: {list(item.keys())}", file=sys.stderr)
                        item_result = original_run(**item)
                        results.append(item_result)
                    result = results
                    print(f"   ✅ Batch complete: {len(results)} results", file=sys.stderr)
                else:
                    print(f"   ✅ Validated input: {list(validated_input.keys()) if isinstance(validated_input, dict) else validated_input}", file=sys.stderr)
                    # Call original method with validated input as kwargs
                    if isinstance(validated_input, dict):
                        result = original_run(**validated_input)
                    else:
                        result = original_run(validated_input, *args, **kwargs)

                    # =============================================================
                    # Passio Post-Processing: Filter + Cache
                    # =============================================================
                    if is_passio_search and result:
                        # Filter the response to reduce token usage
                        original_len = len(str(result))
                        result = filter_passio_response(result)
                        filtered_len = len(str(result))
                        reduction = ((original_len - filtered_len) / original_len * 100) if original_len > 0 else 0
                        print(f"   📉 Filtered Passio response: {original_len} → {filtered_len} chars ({reduction:.0f}% reduction)", file=sys.stderr)

                        # Cache the first result for future queries
                        query = validated_input.get("query", "") if isinstance(validated_input, dict) else ""
                        if query:
                            cache_data = extract_passio_data_for_cache(result, query)
                            if cache_data and cache_data.get("passio_ref_code"):
                                search_cache = get_passio_search_cache()
                                search_cache.set(query, cache_data)
                                print(f"   💾 Cached result for '{query}': {cache_data.get('passio_food_name')}", file=sys.stderr)

            except Exception as e:
                print(
                    f"❌ Tool input validation failed: {e}\n"
                    f"   Tool: {tool_name}\n"
                    f"   Raw input: {raw_input}",
                    file=sys.stderr
                )
                raise
        else:
            # No input at all, call with original args/kwargs
            print(f"   ⚠️ No input detected, calling with empty args", file=sys.stderr)
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

    # Add async run method if not present, wrapping the sync validated run
    if not hasattr(tool, '_arun'):
        async def validated_arun(*args, **kwargs) -> Any:
            return await asyncio.to_thread(validated_run, *args, **kwargs)
        tool._arun = validated_arun

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
