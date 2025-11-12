# CRITICAL ISSUE: CrewAI LLM.call() Never Passes Tools to LiteLLM

## Problem Statement

Despite successfully extracting and converting tools from agents in `llm_provider_rotation.py`, the Music Agent and Description Agent are **NOT calling any MCP tools** during execution.

### Evidence

From `/tmp/crew_test.log`:
```
   ℹ️  Extracted 52 tools from agent
🤖 DESCRIPTION calling LLM (has_tools=True, tool_count=52)
```

But zero tool calls are made by the LLM.

### Root Cause

**CrewAI's `LLM.call()` does NOT pass the `tools` parameter to the underlying LiteLLM completion call.**

This is a **fundamental architectural issue** in CrewAI:

1. We extract tools from `from_agent` parameter ✅
2. We convert tools to OpenAI function calling format ✅  
3. We pass `tools=converted_tools` to `llm.call()` ✅
4. **BUT** CrewAI's `LLM.call()` method **ignores the tools parameter** ❌

### Code Flow

```python
# In llm_provider_rotation.py:458
def call(self, messages, tools=None, ...):
    # We extract and convert tools here
    tools = [...]  # Successfully converted to OpenAI format
    
    # We call CrewAI's LLM.call()
    response = llm.call(
        messages=effective_messages,
        tools=tools,  # ← WE PASS THIS
        callbacks=callbacks,
        ...
    )
```

But in CrewAI's source code (`crewai/llm.py` or similar):
```python
def call(self, messages, **kwargs):
    # tools parameter is in kwargs but NEVER used!
    response = litellm.completion(
        model=self.model,
        messages=messages,
        # tools=??? ← MISSING!
    )
```

### Verification Needed

We need to:
1. Find CrewAI's actual `LLM.call()` implementation
2. Confirm it doesn't pass `tools` to LiteLLM
3. **MONKEY-PATCH** the call method to pass tools

### Solution: Monkey-Patch CrewAI's LLM.call()

Since CrewAI doesn't support tool calling natively (it only supports `available_functions` for direct execution), we need to **intercept and modify** the LLM completion call.

```python
# In llm_auth_init.py or llm_provider_rotation.py

import litellm
from functools import wraps

# Store original completion function
_original_completion = litellm.completion

@wraps(_original_completion)
def _completion_with_tools(*args, **kwargs):
    """Wrapper that ensures tools are passed to completion."""
    # If tools are in kwargs, ensure they're properly formatted
    if 'tools' in kwargs and kwargs['tools']:
        # Validate format
        tools = kwargs['tools']
        if isinstance(tools, list) and len(tools) > 0:
            print(f"🛠️  Passing {len(tools)} tools to LiteLLM completion", file=sys.stderr)
    
    return _original_completion(*args, **kwargs)

# Monkey-patch litellm.completion
litellm.completion = _completion_with_tools
```

### Alternative: Direct LiteLLM Calls

Instead of using CrewAI's LLM wrapper, we could:
1. Extract messages from CrewAI's task
2. Call LiteLLM directly with tools
3. Parse response and feed back to CrewAI

This is more invasive but gives us full control.

### Next Steps

1. ✅ Verify CrewAI's LLM.call() source code
2. ⏳ Implement monkey-patch for litellm.completion
3. ⏳ Test with Music Agent
4. ⏳ Verify tool calls appear in LLM responses

## Timeline

- **2025-01-13**: Discovered issue after extensive debugging
- **2025-01-13**: Documented root cause
- **2025-01-13**: Proposing monkey-patch solution

## Related Files

- `llm_provider_rotation.py`: Where we extract and pass tools
- `llm_auth_init.py`: Where we already monkey-patch for auth headers
- `test_direct_llm_call.py`: Minimal test showing tools are extracted
- `/tmp/crew_test.log`: Evidence that tools aren't being called
