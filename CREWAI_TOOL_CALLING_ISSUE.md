# CrewAI Tool Calling Issue - Investigation

## 🔍 Problem Summary

**Observation**: Music Agent and Description Agent have MCP tools available but LLM is called with `has_tools=False, tool_count=0`

## Evidence from Logs

```
🎵 Found 19 Spotify tools:
   - spotify__searchSpotify
   - spotify__getNowPlaying
   ...

🎵 Creating Music Agent with 19 tools
🔍 Music agent created:
   tools parameter: 19
   agent.tools: 19

🤖 MUSIC calling LLM (has_tools=False, tool_count=0)  # ❌ PROBLEM!
```

## Root Cause Analysis

### What We've Tried

1. ✅ **Fixed endpoint compatibility** - All endpoints support function calling
2. ✅ **Removed `output_json`** - Was blocking tool calls via Instructor
3. ✅ **Added tools to Task** - `tools=agent.tools` in task creation
4. ❌ **Still no tools passed to LLM**

### The Real Problem

CrewAI's architecture doesn't automatically pass agent tools to the LLM during `call()`.

Looking at the call signature in `llm_provider_rotation.py`:
```python
def call(
    self,
    messages: str | List[LLMMessage],
    tools: List[Dict[str, BaseTool]] | None = None,  # ← This is always None!
    ...
)
```

**CrewAI never populates the `tools` parameter** when calling the LLM!

## How CrewAI Should Work

CrewAI has two execution paths:

1. **Agent Executor Mode** (with tools):
   - Agent decides when to call tools
   - Tools are executed via agent executor
   - Results fed back to agent

2. **Direct LLM Mode** (without tools):
   - LLM called directly
   - No tool calling capability

Our `RotatingLLM` is being used in **Direct LLM Mode** even though agents have tools!

## Solution Approaches

### Option 1: Use CrewAI's Agent Executor (Recommended)

CrewAI manages tool calling internally via `CrewAgentExecutor`. The problem is that our custom `RotatingLLM` might not be compatible with this execution path.

**Action**: Ensure agents use the standard CrewAI execution flow.

### Option 2: Manually Pass Tools in RotatingLLM

Modify `RotatingLLM.call()` to check if `from_agent` has tools and pass them:

```python
def call(self, messages, tools=None, from_agent=None, ...):
    # If tools not provided but agent has them, use agent's tools
    if not tools and from_agent and hasattr(from_agent, 'tools'):
        agent_tools = from_agent.tools
        if agent_tools:
            # Convert to format expected by LLM
            tools = self._convert_crewai_tools(agent_tools)
```

### Option 3: Use `function_calling_llm` Parameter

Some CrewAI versions support a separate `function_calling_llm` parameter for agents:

```python
agent = Agent(
    llm=self.music_llm,  # For text generation
    function_calling_llm=self.music_llm,  # For tool calls
    tools=spotify_tools,
)
```

## Next Steps

1. Check CrewAI documentation for proper tool calling setup
2. Verify our `RotatingLLM` is compatible with CrewAI's agent executor
3. Test if tools need to be in a specific format
4. Consider using CrewAI's built-in LLM providers instead of custom wrapper

## Temporary Workaround

Since tools aren't being called anyway, Music Agent could:
- Return empty music_tracks list
- Or be instructed to generate plausible tracks (not ideal - breaks "no hallucination" policy)

But the real fix is to enable proper MCP tool calling through CrewAI.
