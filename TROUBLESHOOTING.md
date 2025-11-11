# Troubleshooting Guide

## Tool Input Validation Errors

### Symptom
```
Error: the Action Input is not a valid key, value dictionary.
```

Agent attempts to call an MCP tool but passes a list instead of a dictionary:
```json
Tool Input: [
  {"start_date": "2025-11-11", ...},
  {"start_date": "2025-11-11", ...},
  {"status": "success", "data": [...]}
]
```

### Root Cause

The LLM (Claude Sonnet 4.5 or other models) occasionally generates malformed tool calls where:
1. Parameters are wrapped in a list instead of passed as a single dictionary
2. Multiple parameter sets are included (duplicates)
3. Mocked/example responses are included in the tool call

This appears to be an issue with how the LLM interprets tool schemas in certain contexts, particularly when:
- The agent is under cognitive load (complex reasoning)
- The tool has been called multiple times in the conversation
- The agent is trying to anticipate the response format

### Solution Implemented

We've implemented a **two-layer defense**:

#### 1. Enhanced Agent Prompts (`agents/description_agent.py`)

Added explicit examples of correct and incorrect tool usage:
```python
CRITICAL TOOL USAGE RULES:
1. Call ONE tool at a time and wait for the result
2. Pass parameters as a SINGLE DICTIONARY ONLY - NEVER use a list
3. Do NOT include multiple parameter sets in one call
4. Do NOT predict or include the tool's response in your tool call

✅ CORRECT EXAMPLE:
Tool: IntervalsIcu__get_activities
Input: {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 10}

❌ WRONG - DO NOT DO THIS:
Input: [{"start_date": "2025-11-11", ...}, {"start_date": "2025-11-11", ...}]
```

#### 2. MCP Tool Wrapper (`mcp_tool_wrapper.py`)

Created a defensive wrapper that:
- Detects when input is a list instead of dict
- Extracts the first valid parameter dictionary
- Filters out mocked responses (dicts with "status" and "data" keys)
- Logs warnings when malformed inputs are detected
- Raises clear errors when input cannot be salvaged

The wrapper is applied to all MCP tools in `crew.py`:
```python
from mcp_tool_wrapper import wrap_mcp_tools

# After loading MCP tools
self.mcp_tools = wrap_mcp_tools(self.mcp_tools)
```

### Testing

Run the test suite to verify the wrapper works correctly:
```bash
python test_tool_wrapper.py
```

Expected output:
```
✅ All tests passed!
```

### Monitoring in Production

When the wrapper detects and fixes a malformed input, it logs:
```
⚠️  Warning: Tool received a list instead of dict. Attempting to extract parameters...
✅ Extracted parameters: {"start_date": "2025-11-11", ...}
```

If you see these warnings frequently, it indicates the LLM is not following tool usage instructions. Consider:
1. Switching to a different model (e.g., GPT-5, Claude Opus)
2. Reducing agent workload (simplify tasks)
3. Upgrading CrewAI to the latest version

### Alternative Solutions (Not Implemented)

1. **Model Switching**: Use a different LLM that handles tool calls more reliably
2. **CrewAI Upgrade**: Upgrade to latest CrewAI version (current: 1.3.0)
3. **Sequential Tool Calls**: Force agents to make only one tool call at a time
4. **Tool Call Parsing**: Monkey-patch CrewAI's tool call parser

### Related Issues

- CrewAI GitHub: Tool call parsing issues with Claude models
- LiteLLM: Function calling format variations across providers

## Other Common Issues

### MCP Connection Failures

See `CLAUDE.md` section "MCP Connection Failures" for diagnostics.

### LLM Provider Errors

See `CLAUDE.md` section "The requested model is not supported" for model normalization issues.
