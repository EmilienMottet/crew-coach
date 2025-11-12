#!/usr/bin/env python3
"""Test that tools are correctly formatted for OpenAI function calling."""

import sys
import json
from dotenv import load_dotenv

load_dotenv()

# Simple test: just check if CrewAI tools have the necessary conversion methods
from crewai.tools.base_tool import BaseTool

def test_tool_methods():
    """Test what methods are available on CrewAI tools."""
    print("� Checking CrewAI BaseTool methods:\n")
    
    # List all methods
    methods = [m for m in dir(BaseTool) if not m.startswith('_')]
    print(f"Available methods: {', '.join(methods)}\n")
    
    # Check for specific conversion methods
    has_to_function = hasattr(BaseTool, 'to_function')
    has_args_schema = hasattr(BaseTool, 'args_schema')
    
    print(f"✅ to_function: {has_to_function}")
    print(f"✅ args_schema: {has_args_schema}\n")
    
    # Try creating a simple tool
    from crewai import tool
    
    @tool
    def test_tool(query: str) -> str:
        """A test tool that does nothing."""
        return f"Result for: {query}"
    
    print(f"🧪 Test tool created: {test_tool.name}\n")
    print(f"   Attributes: {[a for a in dir(test_tool) if not a.startswith('_')]}\n")
    
    # Try conversion
    if hasattr(test_tool, 'to_function'):
        func_def = test_tool.to_function()
        print("✅ to_function() works!")
        print(f"   Result: {json.dumps(func_def, indent=2)}\n")
        return True
    elif hasattr(test_tool, 'args_schema'):
        schema = test_tool.args_schema
        print(f"✅ args_schema exists: {schema}")
        if schema:
            json_schema = schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema
            print(f"   Schema: {json.dumps(json_schema, indent=2)}\n")
        tool_def = {
            "type": "function",
            "function": {
                "name": getattr(test_tool, 'name', str(test_tool)),
                "description": getattr(test_tool, 'description', ''),
                "parameters": schema.model_json_schema() if schema and hasattr(schema, 'model_json_schema') else {}
            }
        }
        print(f"✅ Manually constructed tool definition:")
        print(f"{json.dumps(tool_def, indent=2)}\n")
        return True
    else:
        print("❌ No conversion method found!\n")
        return False

if __name__ == "__main__":
    success = test_tool_methods()
    sys.exit(0 if success else 1)
