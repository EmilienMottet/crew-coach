# GitHub Copilot Instructions - Strava Activity Description Crew

## Project Overview

This is a **CrewAI-based multi-agent system** that automatically generates engaging titles and descriptions for Strava running activities using data from Intervals.icu. The system enforces privacy policies and work-hours compliance.

### Key Technologies
- **CrewAI**: Multi-agent orchestration framework
- **LiteLLM**: LLM abstraction layer (supports OpenAI-compatible endpoints)
- **MCP (Model Context Protocol)**: For accessing external data sources (Intervals.icu, Strava, etc.)
- **Python 3.x**: Core language
- Integration with **n8n** workflows for automation

## Architecture

### Agent System
The project uses a **four-agent sequential workflow**:

1. **Activity Description Writer** (`agents/description_agent.py`)
   - Analyzes Intervals.icu training data
   - Generates concise titles (≤50 chars) and descriptions (≤500 chars) in English
   - Identifies workout types (tempo, intervals, easy runs, etc.)
   - Uses emoji appropriately for visual appeal

2. **Activity Soundtrack Curator** (`agents/music_agent.py`)
  - Queries Spotify MPC endpoints for the activity window
  - Collects up to five tracks actually played during the workout
  - Appends a short "Music" section to the English description (≤500 chars total)
  - Falls back gracefully when no playback data is available

3. **Privacy & Compliance Officer** (`agents/privacy_agent.py`)
   - Validates generated content for PII (personally identifiable information)
   - Checks activity timing against work hours (08:30-12:00, 14:00-17:00 CET/CEST)
   - Recommends privacy settings (public/private)
   - Sanitizes sensitive information

4. **Sports Content Translator** (`agents/translation_agent.py`)
   - Translates titles and descriptions from English to French
   - Preserves emojis and formatting
   - Adapts sports terminology appropriately for French-speaking audience
   - Maintains character limits (≤50 chars for title, ≤500 chars for description)
   - **CRITICAL**: The Strava audience is French-speaking, so final content MUST be in French

### Data Flow
```
Strava Webhook → n8n → crew.py (stdin)
  ↓
Description Agent (+ MCP tools) → Generate content (English)
  ↓
Music Agent (+ Spotify MCP) → Append soundtrack summary
  ↓
Privacy Agent → Validate & adjust
  ↓
Translation Agent → Translate to French (MANDATORY - audience is French-speaking)
  ↓
Final JSON output (stdout) → n8n → Update Strava
```

### MCP Tools Integration

**IMPORTANT: We use MetaMCPAdapter, NOT CrewAI's native `mcps` DSL**

#### Why MetaMCPAdapter?

We attempted to migrate to CrewAI's native `mcps` DSL but encountered a **critical compatibility issue**:

- **CrewAI's native DSL** expects standard MCP transports (`stdio`, `sse`, `streamable-http`)
- **MetaMCP uses a custom protocol** not recognized by CrewAI's auto-discovery
- **Result with native DSL**: `0 tools discovered`, complete failure ❌

**Current working approach** (DO NOT CHANGE without testing):
```python
# ✅ WORKING: MetaMCPAdapter (custom adapter)
from mcp_auth_wrapper import MetaMCPAdapter
from mcp_tool_wrapper import wrap_mcp_tools

adapter = MetaMCPAdapter(server_url, mcp_api_key, connect_timeout=30)
adapter.start()
tools = adapter.tools  # ✅ All tools discovered

# Pass to agents via 'tools' parameter, NOT 'mcps'
agent = create_description_agent(llm, tools=tools)
```

**What does NOT work**:
```python
# ❌ BROKEN: CrewAI native DSL
agent = Agent(
    mcps=["https://mcp.emottet.com/metamcp/..."]  # ❌ 0 tools discovered
)
```

#### Available MCP Tools
- `IntervalsIcu__get_activity_details`: Full activity metrics
- `IntervalsIcu__get_activity_intervals`: Interval/segment breakdowns
- `IntervalsIcu__get_activities`: Recent activity list
- `Spotify__get_recently_played`: Music playback history
- `Strava__*`: Strava API operations
- Weather and Toolbox utilities

#### MCP Server Configuration
```bash
# Individual server URLs (recommended)
STRAVA_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/SocialNetworkSport/mcp
INTERVALS_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/IntervalsIcu/mcp
MUSIC_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/Music/mcp
METEO_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/meteo/mcp
TOOLBOX_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/toolbox/mcp
MCP_API_KEY=your-mcp-api-key
```

#### Tool Discovery Process
1. **Connection**: `MetaMCPAdapter` connects to each configured MCP server
2. **Discovery**: Automatically discovers all available tools
3. **Filtering**: Tools are filtered by name pattern (e.g., "intervals", "spotify", "strava")
4. **Validation**: `wrap_mcp_tools()` adds input validation to prevent LLM errors
5. **Distribution**: Tools are distributed to appropriate agents based on their needs

**Do NOT remove** these files (required for MetaMCP integration):
- `mcp_auth_wrapper.py` - MetaMCP connection adapter
- `mcp_tool_wrapper.py` - Input validation wrapper
- `mcp_utils.py` - Utility functions (may be unused but kept for compatibility)

#### Future Migration to Native DSL
If CrewAI adds support for MetaMCP's custom protocol, or if MetaMCP implements standard transports, we can migrate to the native `mcps` DSL. Until then, **MetaMCPAdapter is the only working solution**.

Reference: See `DSL_COMPATIBILITY_ISSUE.md` for detailed technical analysis.

## Coding Conventions

### Language & Comments
- **Code**: Write in English (variables, functions, classes, docstrings)
- **Comments**: English for technical explanations
- **User-facing text**: French is REQUIRED for final Strava output (titles and descriptions)
- **Documentation**: French (README.md) for end-users, English for code
- **Important**: The Strava audience is French-speaking, so translation to French is MANDATORY

### Python Style
- Follow **PEP 8** conventions
- Use **type hints** for function parameters and returns
- Prefer **f-strings** for formatting
- Use **descriptive variable names** (e.g., `activity_data`, `privacy_check`)
- Add **docstrings** to all functions/classes (Google style)

### Error Handling
- Always use **try-except** blocks for external API calls (MCP, LLM)
- Log errors to **stderr** (using `sys.stderr`)
- Output user-friendly JSON errors to **stdout** (for n8n integration)
- Provide **fallback values** when parsing fails (e.g., default to private activities)

### Example Pattern
```python
try:
    result = client.call_tool("tool_name", params)
    parsed_data = json.loads(result)
except json.JSONDecodeError as e:
    print(f"⚠️ Warning: {str(e)}", file=sys.stderr)
    parsed_data = {"error": "default_safe_value"}
```

## LLM Configuration

### Authentication Methods
The system uses **API Key Bearer Token** authentication:

```python
OPENAI_API_KEY=your-api-key
# Results in: Authorization: Bearer <key>
```

**Legacy Basic Auth (REMOVED)**: Previous versions used `OPENAI_API_AUTH_TOKEN` with Basic Authentication. This has been **removed** for simplicity and better compatibility with standard OpenAI-compatible endpoints.

### LLM Endpoint Setup
- Use `OPENAI_API_BASE` for custom endpoints (e.g., `https://ccproxy.emottet.com/v1`)
- Use `OPENAI_MODEL_NAME` for model selection (e.g., `claude-sonnet-4.5`, `gpt-5-mini`)
- CrewAI uses **LiteLLM**, which provides abstraction over multiple LLM providers
- Set `drop_params=True` to ignore unsupported parameters

### Important Notes
- Always use `OPENAI_API_KEY` (Bearer token), not Basic Auth
- Test endpoint connectivity before deploying changes
- Different agents can use different models/endpoints via per-agent environment variables

## CrewAI Best Practices

### Agent Design
- **Single responsibility**: Each agent has one clear purpose
- **No delegation**: `allow_delegation=False` for focused agents
- **Tools only when needed**: Privacy agent doesn't need external tools
- **Detailed backstories**: Help LLMs understand context and constraints

### Task Creation
- Define clear **expected outputs** (JSON schema, format, constraints)
- Use **context variables** from previous tasks in sequential workflows
- Provide **example outputs** in task descriptions when possible

### Process Flow
- Use `Process.sequential` for dependent tasks
- Handle **CrewOutput** objects carefully (they may not serialize to JSON directly)
- Always convert results to strings before parsing: `str(result)`

## Environment Variables

### Required
```bash
# LLM Configuration
OPENAI_API_BASE=https://your-endpoint.com/v1
OPENAI_MODEL_NAME=claude-sonnet-4.5
OPENAI_API_KEY=your-api-key  # Bearer token authentication

# MCP Server Configuration (individual servers)
STRAVA_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/SocialNetworkSport/mcp
INTERVALS_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/IntervalsIcu/mcp
MUSIC_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/Music/mcp
METEO_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/meteo/mcp
TOOLBOX_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/toolbox/mcp
MCP_API_KEY=your-mcp-api-key

# Privacy Policy
WORK_START_MORNING=08:30
WORK_END_MORNING=12:00
WORK_START_AFTERNOON=14:00
WORK_END_AFTERNOON=17:00

# Translation Configuration (REQUIRED - audience is French-speaking)
TRANSLATION_ENABLED=true  # MUST be true for French output
TRANSLATION_TARGET_LANGUAGE=French  # Target language (French for Strava)
TRANSLATION_SOURCE_LANGUAGE=English  # Source language from Description Agent
```

### Optional
- Per-agent model configuration (e.g., `DESCRIPTION_AGENT_MODEL`, `MUSIC_AGENT_MODEL`)
- Provider rotation: `ENABLE_LLM_PROVIDER_ROTATION=true`

### Removed/Deprecated
- `OPENAI_API_AUTH_TOKEN` ❌ (replaced by `OPENAI_API_KEY`)
- `INTERVALS_MCP_TOOL_NAMES` ❌ (auto-discovery used)
- `SPOTIFY_MCP_TOOL_NAMES` ❌ (auto-discovery used)

### Translation Requirements
- **CRITICAL**: `TRANSLATION_ENABLED` MUST be set to `true` in production
- **Target Language**: Always `French` (the Strava audience is French-speaking)
- **Source Language**: `English` (Description Agent generates in English)
- Translation preserves emojis, formatting, and respects character limits

## Input/Output Contracts

### Input (stdin)
JSON array from Strava webhook:
```json
[{
  "object_type": "activity",
  "object_id": 16284886069,
  "aspect_type": "create",
  "object_data": {
    "id": 16284886069,
    "name": "Lunch Run",
    "distance": 12337,
    "moving_time": 3601,
    "type": "Run",
    "start_date_local": "2025-10-27T11:54:41Z"
  }
}]
```

### Output (stdout)
Structured JSON result (title and description in FRENCH):
```json
{
  "activity_id": 16284886069,
  "title": "🏃 Sortie tempo 12.3K - Effort soutenu",
  "description": "Sortie tempo solide avec un contrôle de l'allure. ...\n\n🎧 Musique : Daft Punk – Harder Better Faster Stronger; Justice – D.A.N.C.E.",
  "should_be_private": false,
  "privacy_check": {
    "approved": true,
    "during_work_hours": false,
    "issues": [],
    "reasoning": "No privacy issues found."
  },
  "workout_analysis": {
    "type": "Tempo Run",
    "metrics": {
      "average_pace": "4:53 /km",
      "average_hr": "141 bpm"
    }
  },
  "music_tracks": [
    "Daft Punk – Harder Better Faster Stronger",
    "Justice – D.A.N.C.E."
  ]
}
```

**Note**: Title and description are in French because the Strava audience is French-speaking. The Translation Agent converts the English content from the Description Agent to French.

## Testing & Debugging

### Manual Testing
```bash
# Activate virtual environment first
source venv/bin/activate

# Test with sample input
cat input.json | python crew.py

# Capture stderr for debugging
python crew.py < input.json 2> debug.log

# Test in verbose mode (shows agent reasoning)
cat input.json | python crew.py 2>&1 | tee full_output.log
```

### Common Issues
1. **"ModuleNotFoundError"**: Ensure venv is activated and dependencies installed
2. **"LLM Provider NOT provided"**: Check `OPENAI_API_BASE` and `OPENAI_MODEL_NAME` are set
3. **MCP connection failures**: Verify `MCP_SERVER_URL` and test with curl
4. **JSON parsing errors**: Always use defensive parsing with fallback values
5. **Work hours detection**: Ensure timezone is Europe/Paris (CET/CEST)
6. **Wrong Python version**: Verify you're using Python 3.10+ inside the venv

### Debugging Tools
- Use `verbose=True` in agents for detailed execution logs
- Check stderr for execution flow (step-by-step progress)
- Test MCP tools independently before integration
- Use `curl` to verify endpoint connectivity

## Code Modification Guidelines

### When adding new agents:
1. Create agent definition in `agents/` directory
2. Add corresponding task in `tasks/` directory
3. Update `crew.py` to instantiate and orchestrate
4. Add agent to the sequential workflow
5. Document expected input/output formats

### When adding new MCP tools:
1. Define lightweight helpers close to the agents or dedicated modules as needed
2. Use `@tool` decorator with clear description
3. Handle errors gracefully (return error dict)
4. Test tool independently before adding to agent
5. Update agent's `tools` list in initialization

### When modifying privacy rules:
1. Update `privacy_agent.py` backstory with new rules
2. Adjust environment variables if needed (e.g., new time ranges)
3. Test with activities that violate new rules
4. Document changes in README.md

### When changing LLM configuration:
1. Test endpoint compatibility with curl first
2. Update `.env.example` with new parameters
3. Handle authentication changes in `crew.py`
4. Document model-specific quirks or limitations

## Integration with n8n

### Expected Workflow
1. **Strava Webhook Trigger**: Receives activity events
2. **Execute Command Node**: Runs `python crew.py` with stdin
   - **Important**: Use absolute path to venv Python: `/path/to/crew/venv/bin/python crew.py`
   - Or activate venv in command: `source venv/bin/activate && python crew.py`
3. **Parse JSON Node**: Extracts title, description, privacy settings
4. **Conditional Node**: Routes based on `should_be_private`
5. **Strava Update Node**: Updates activity via API

### Error Handling
- Always output valid JSON (even on errors)
- Use exit code `1` for failures
- Include `error` field in JSON output
- Set `should_be_private: true` as safe default

## Privacy & Security

### Sensitive Data Protection
- **Never log** API keys, authentication tokens, or PII
- **Redact** sensitive information before outputting to logs
- **Validate** all user inputs before processing
- **Use environment variables** for all credentials

### Work Hours Compliance
- Always check activity time against configured work hours
- Default to **private** if uncertain or during work hours
- Respect timezone settings (Europe/Paris)
- Document any changes to work hour policies

## Python Environment Setup

### Virtual Environment (venv)
Always use a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Important Notes
- **Never commit** the `venv/` directory to git (should be in `.gitignore`)
- **Always activate** the venv before running the crew or installing packages
- **Document** the Python version used (e.g., Python 3.10+)
- Use `pip freeze > requirements.txt` to update dependencies list

### Verifying Installation
```bash
# Check Python version (should be 3.10+)
python --version

# Verify CrewAI installation
python -c "import crewai; print(crewai.__version__)"

# List installed packages
pip list
```

## Technical Decisions & Architecture Choices

### Why MetaMCPAdapter instead of CrewAI Native DSL?

**Date**: November 2025  
**Decision**: Use `MetaMCPAdapter` custom wrapper instead of CrewAI's native `mcps` DSL

#### Context
CrewAI 0.86.0+ introduced a native DSL for MCP integration via the `mcps` agent parameter. We attempted migration to align with CrewAI best practices and simplify the codebase.

#### Problem Encountered
```
[WARNING]: Failed to get MCP tool schemas from https://mcp.emottet.com/metamcp/...
[WARNING]: No tools discovered from MCP server
[INFO]: Successfully loaded 0 tools  ← Complete failure
```

#### Root Cause
- **CrewAI's native DSL** expects standard MCP transports:
  - `stdio`: Local processes via stdin/stdout
  - `sse`: Server-Sent Events (EventSource)
  - `streamable-http`: HTTP with streaming

- **MetaMCP uses a custom protocol** not recognized by CrewAI's transport auto-detection
- CrewAI's MCP client attempts to negotiate protocol but fails with MetaMCP endpoints
- Result: Zero tools discovered, agents have no access to external data

#### Solution
Maintain `MetaMCPAdapter` custom wrapper:
```python
# ✅ Works with MetaMCP
from mcp_auth_wrapper import MetaMCPAdapter

adapter = MetaMCPAdapter(server_url, mcp_api_key, connect_timeout=30)
adapter.start()
tools = adapter.tools  # All tools successfully discovered

agent = create_agent(llm, tools=tools)  # Via 'tools', NOT 'mcps'
```

#### Trade-offs

| Aspect | Native DSL `mcps` | MetaMCPAdapter |
|--------|-------------------|----------------|
| **Compatibility** | ❌ 0 tools with MetaMCP | ✅ Full compatibility |
| **Maintenance** | ✅ Official CrewAI | ⚠️ Custom code |
| **Simplicity** | ✅ Declarative config | ⚠️ Imperative setup |
| **Status** | ❌ Broken for MetaMCP | ✅ Production-ready |

#### Future Path
1. **Monitor CrewAI updates**: Check if future versions support MetaMCP protocol
2. **Coordinate with MetaMCP**: Consider implementing standard transport
3. **Migrate when ready**: Switch to native DSL if/when compatible

**References**:
- `DSL_COMPATIBILITY_ISSUE.md` - Technical deep-dive
- `PROBLEM_RESOLVED.md` - Migration attempt and rollback
- CrewAI MCP docs: https://docs.crewai.com/en/mcp/overview

### Why API Key instead of Basic Auth?

**Date**: November 2025  
**Decision**: Use `OPENAI_API_KEY` (Bearer token) instead of `OPENAI_API_AUTH_TOKEN` (Basic Auth)

#### Rationale
1. **Standard compatibility**: Bearer tokens are the standard for OpenAI-compatible APIs
2. **Simpler code**: No need for custom header patching (~60 lines removed)
3. **Better curl compatibility**: Direct alignment with endpoint requirements
4. **Industry standard**: Most LLM providers use Bearer authentication

#### Implementation
```bash
# Old (removed)
OPENAI_API_AUTH_TOKEN=base64_credentials

# New (current)
OPENAI_API_KEY=your-api-key
```

**Impact**: ~60 lines of OpenAI client patching code removed, cleaner LLM initialization.

---

## Quick Reference for AI Assistants

When helping with this project:
1. **Always check** existing code patterns before suggesting new approaches
2. **Respect** the four-agent architecture (Description → Music → Privacy → Translation)
3. **Follow** error handling patterns (try-except, fallback values, stderr logging)
4. **Test** MCP and LLM connectivity before debugging agent logic
5. **Document** any new environment variables or configuration options
6. **Maintain** compatibility with n8n workflow expectations (stdin/stdout JSON)
7. **CRITICAL**: Ensure translation to French is ALWAYS enabled (TRANSLATION_ENABLED=true)
8. **Remember**: The Strava audience is French-speaking, so final output MUST be in French
9. **Do NOT migrate to `mcps` DSL**: MetaMCPAdapter is the only working solution for MetaMCP
````
- **Always activate** the venv before running the crew or installing packages
- **Document** the Python version used (e.g., Python 3.10+)
- Use `pip freeze > requirements.txt` to update dependencies list

### Verifying Installation
```bash
# Check Python version (should be 3.10+)
python --version

# Verify CrewAI installation
python -c "import crewai; print(crewai.__version__)"

# List installed packages
pip list
```

## Dependencies Management

### Core Dependencies
- `crewai==0.86.0`: Agent framework (pin version for stability)
- `crewai-tools==0.17.0`: Tool decorators and utilities
- `python-dotenv==1.0.0`: Environment variable loading
- `requests==2.31.0`: HTTP client for MCP
- `openai==1.58.1`: OpenAI API compatibility
- `pytz==2025.2`: Timezone handling

### Version Updates
- Test thoroughly before updating CrewAI (breaking changes common)
- Verify LiteLLM compatibility when updating OpenAI SDK
- Check MCP protocol version compatibility
- Always test in the venv before deploying changes

## Performance Considerations

- **MCP calls**: Can be slow (network latency), minimize unnecessary calls
- **LLM inference**: Depends on endpoint, may take 5-30 seconds
- **Sequential processing**: Total time = sum of agent execution times
- **Timeout handling**: Set reasonable timeouts for external calls (30s for MCP)

## Future Enhancements (Context for AI)

Potential areas for expansion:
- Additional agents (e.g., nutrition suggestions, training plan integration)
- Support for other sports (cycling, swimming, etc.)
- Multi-language support for descriptions
- Advanced workout type detection (machine learning)
- Integration with additional data sources (weather, music, routes)

---

## Quick Reference for AI Assistants

When helping with this project:
1. **Always check** existing code patterns before suggesting new approaches
2. **Respect** the four-agent architecture (Description → Music → Privacy → Translation)
3. **Follow** error handling patterns (try-except, fallback values, stderr logging)
4. **Test** MCP and LLM connectivity before debugging agent logic
5. **Document** any new environment variables or configuration options
6. **Maintain** compatibility with n8n workflow expectations (stdin/stdout JSON)
7. **CRITICAL**: Ensure translation to French is ALWAYS enabled (TRANSLATION_ENABLED=true)
8. **Remember**: The Strava audience is French-speaking, so final output MUST be in French
