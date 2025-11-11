# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **CrewAI multi-agent system** that automatically generates French titles and descriptions for Strava running activities using Intervals.icu training data. The system enforces privacy policies, work-hours compliance, and mandatory French translation for a French-speaking audience.

**Key Technologies**: CrewAI, LiteLLM, MCP (Model Context Protocol), Python 3.x

## Common Commands

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Always activate before running commands

# Install dependencies
pip install -r requirements.txt

# Update dependencies
pip freeze > requirements.txt
```

### Running the System
```bash
# Process activity from input file
python crew.py < input.json

# Capture debug output
python crew.py < input.json 2> debug.log

# Full verbose output
cat input.json | python crew.py 2>&1 | tee full_output.log
```

### Testing
```bash
# Test MCP connectivity (RECOMMENDED - run this first!)
python test_mcp_connectivity.py

# Test with sample input
cat input.json | python crew.py

# Test privacy agent specifically
python test_privacy_agent.py

# Test translation workflow
bash test_translation.sh
```

### Debugging LLM/MCP Connectivity
```bash
# Test MCP server connectivity and tool availability (Recommended)
python test_mcp_connectivity.py

# Test LLM endpoint
curl -X POST $OPENAI_API_BASE/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $OPENAI_API_AUTH_TOKEN" \
  -d '{"model": "gpt-5", "messages": [{"role": "user", "content": "Hello"}]}'

# Test MCP server (basic curl check)
curl "$MCP_SERVER_URL"

# List available models
curl $OPENAI_API_BASE/models
```

## Connecting to MCP Servers

### MCP Server Configuration

The project uses **MetaMCP** with **dedicated MCP servers** for each service provider. Each crew connects to multiple specialized MCP endpoints.

**Environment Setup:**
```bash
# Shared MCP API Key (used by all MCP servers)
MCP_API_KEY=sk_mt_your_api_key_here

# ============================================
# MCP Servers for Strava Description Crew (crew.py)
# ============================================

# Strava (19 tools): Activity data, athlete profile, segments, routes
STRAVA_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/SocialNetworkSport/mcp

# Intervals.icu (9 tools): Training data, activity intervals, wellness
INTERVALS_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/IntervalsIcu/mcp

# Music/Spotify (19 tools): Music playback history, recently played tracks
MUSIC_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/Music/mcp

# Weather/OpenWeatherMap (11 tools): Weather context during activities
METEO_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/meteo/mcp

# Toolbox (14 tools): Utility tools (time conversion, fetch, task management)
TOOLBOX_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/toolbox/mcp

# ============================================
# MCP Servers for Meal Planning Crew (crew_mealy.py)
# ============================================

# Food (20 tools): Hexis nutrition data + USDA Food Data Central
FOOD_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/food/mcp

# Intervals.icu and Toolbox are reused from above
```

**Architecture Benefits:**
- **Separation of Concerns**: Each MCP server manages a specific domain
- **Selective Loading**: Crews only connect to servers they need
- **Fault Tolerance**: If one server fails, others remain operational
- **Tool Organization**: Tools are grouped by provider for easier filtering

### How MCP Tools are Loaded

The crews use **MetaMCPAdapter** to connect to multiple MCP servers and aggregate tools:

```python
# In crew.py or crew_mealy.py
from mcp_auth_wrapper import MetaMCPAdapter

# Connect to multiple MCP servers
mcp_servers = {
    "Strava": os.getenv("STRAVA_MCP_SERVER_URL", ""),
    "Intervals.icu": os.getenv("INTERVALS_MCP_SERVER_URL", ""),
    "Music": os.getenv("MUSIC_MCP_SERVER_URL", ""),
    # ... etc
}

mcp_adapters = []
mcp_tools = []

for server_name, server_url in mcp_servers.items():
    adapter = MetaMCPAdapter(server_url, mcp_api_key, connect_timeout=30)
    adapter.start()
    mcp_adapters.append(adapter)
    mcp_tools.extend(adapter.tools)  # Aggregate tools from all servers

# Filter tools by type
strava_tools = [t for t in mcp_tools if "strava" in t.name.lower()]
intervals_tools = [t for t in mcp_tools if "intervals" in t.name.lower()]

# Create agent with filtered tools
agent = Agent(
    role="Description Agent",
    tools=strava_tools + intervals_tools,  # Pass tools directly
    # ... other config
)
```

**What happens:**
1. Each MCP server is connected independently via `MetaMCPAdapter`
2. Tools from all servers are aggregated into a single `mcp_tools` list
3. Tools are filtered by name/type for each agent
4. Agents receive specific tool subsets relevant to their role
5. Connection is kept alive for the duration of the crew execution

### Testing MCP Connectivity

```bash
# Test complete MCP workflow (recommended)
python test_mcp_connectivity.py

# Expected output:
# 🔗 Connecting to 5 MCP servers...
#    Connecting to Strava...
#    ✅ Strava: 19 tools discovered
#    Connecting to Intervals.icu...
#    ✅ Intervals.icu: 9 tools discovered
#    Connecting to Music...
#    ✅ Music: 19 tools discovered
#    Connecting to Meteo...
#    ✅ Meteo: 11 tools discovered
#    Connecting to Toolbox...
#    ✅ Toolbox: 14 tools discovered
#
# ✅ MCP connection complete! Total tools discovered: 72
```

### MCP Server Architecture

**Multiple Specialized Servers:**
```
Strava Description Crew (crew.py) connects to:
    ├── SocialNetworkSport (19 Strava tools)
    │   ├── strava__get-activity-details
    │   ├── strava__get-activity-streams
    │   ├── strava__get-athlete-profile
    │   └── ...
    ├── IntervalsIcu (9 training tools)
    │   ├── IntervalsIcu__get_activity_details
    │   ├── IntervalsIcu__get_activity_intervals
    │   └── ...
    ├── Music (19 Spotify tools)
    │   ├── spotify__getRecentlyPlayed
    │   ├── spotify__getNowPlaying
    │   └── ...
    ├── Meteo (11 OpenWeatherMap tools)
    │   ├── OpenWeatherMap__get-current-weather
    │   ├── OpenWeatherMap__get-daily-forecast
    │   └── ...
    └── Toolbox (14 utility tools)
        ├── Time__get_current_time
        ├── Fetch__fetch
        └── ...

Meal Planning Crew (crew_mealy.py) connects to:
    ├── Food (20 nutrition tools)
    │   ├── hexis__hexis_get_nutrition_trend
    │   ├── hexis__hexis_get_meal_inspiration
    │   ├── food-data-central__search-foods
    │   └── ...
    ├── IntervalsIcu (shared, 9 tools)
    └── Toolbox (shared, 14 tools)
```

**Tool Naming Conventions:**
- **Strava**: `strava__method-name` (kebab-case)
- **Intervals.icu**: `IntervalsIcu__method_name` (PascalCase + snake_case)
- **Spotify**: `spotify__methodName` (camelCase)
- **OpenWeatherMap**: `OpenWeatherMap__method-name` (PascalCase + kebab-case)
- **Hexis**: `hexis__hexis_method_name` (all lowercase + snake_case)
- **Food Data Central**: `food-data-central__method-name` (kebab-case)
- **Toolbox**: `Tool__method_name` (varies by tool)

### MCP Tool Distribution by Agent

**Strava Description Crew (`crew.py`):**
- **Description Agent**: Strava + Intervals.icu + Weather + Toolbox
  - Needs comprehensive data to generate descriptions
  - Weather context enriches activity narratives
  - Toolbox for time/date utilities
- **Music Agent**: Spotify
  - Fetches recently played tracks during activity
  - Enriches description with soundtrack information
- **Privacy Agent**: No MCP tools (pure reasoning)
- **Translation Agent**: No MCP tools (pure reasoning)

**Meal Planning Crew (`crew_mealy.py`):**
- **Hexis Analysis Agent**: Hexis + Intervals.icu + Toolbox
  - Analyzes training load from Hexis
  - Cross-references with Intervals.icu data
  - Uses time utilities for weekly planning
- **Weekly Structure Agent**: No MCP tools (pure reasoning)
- **Meal Generation Agent**: Hexis + Food Data Central + Toolbox
  - Hexis for meal inspiration and recipes
  - Food Data Central for nutritional data
  - Toolbox for date calculations
- **Nutritional Validation Agent**: Food Data Central + Hexis
  - Validates meal nutritional accuracy
  - Cross-checks with food databases
- **Mealy Integration Agent**: Hexis
  - Integrates with Hexis meal planning system

### Using mcp-proxy for stdio Clients

**Note**: CrewAI connects directly to MCP servers via HTTP. The `mcp-proxy` tool is **only needed for stdio-based clients** (Claude Desktop, Cursor, etc.).

**For CrewAI (current setup):**
- No proxy needed - direct HTTP connection via MetaMCPAdapter
- Set individual MCP server URLs (STRAVA_MCP_SERVER_URL, INTERVALS_MCP_SERVER_URL, etc.)
- Set shared `MCP_API_KEY` environment variable
- API key is automatically appended to requests
- Multiple servers are connected simultaneously

**For stdio clients (Claude Desktop, Cursor, etc.):**
```json
{
  "mcpServers": {
    "Strava": {
      "command": "uvx",
      "args": [
        "mcp-proxy",
        "https://mcp.emottet.com/metamcp/SocialNetworkSport/mcp"
      ],
      "env": {
        "API_ACCESS_TOKEN": "sk_mt_your_api_key_here"
      }
    },
    "IntervalsIcu": {
      "command": "uvx",
      "args": [
        "mcp-proxy",
        "https://mcp.emottet.com/metamcp/IntervalsIcu/mcp"
      ],
      "env": {
        "API_ACCESS_TOKEN": "sk_mt_your_api_key_here"
      }
    }
  }
}
```

### Debugging MCP Calls

```bash
# Enable verbose mode
export CREWAI_VERBOSE=true

# Check MCP tool discovery
python crew.py < input.json 2>&1 | grep -i "mcp"

# Expected warnings (safe to ignore):
# [WARNING]: Failed to get MCP tool schemas... TaskGroup error
# [WARNING]: No tools discovered from MCP server
# [INFO]: Successfully loaded 0 tools

# Actual tool calls will still work:
# Agent calls IntervalsIcu__get_activity_details → Success ✅
```

## Architecture

### Three-Agent Sequential Workflow

**CRITICAL**: This is a mandatory three-step pipeline. The Translation Agent is NOT optional - the Strava audience is French-speaking.

1. **Description Agent** (`agents/description_agent.py`)
   - Fetches training data from Intervals.icu via MCP tools
   - Generates English title (≤50 chars) and description (≤500 chars)
   - Identifies workout structure (tempo, intervals, easy runs, etc.)
   - Uses emojis for visual appeal
  -    - **Tools**: MCP references to Intervals.icu (auto-discovered via `MCP_SERVER_URL`, optionally filtered with `INTERVALS_MCP_TOOL_NAMES`)

2. **Privacy Agent** (`agents/privacy_agent.py`)
   - Validates content for PII (personally identifiable information)
   - Checks activity timing against work hours (08:30-12:00, 14:00-17:00 Europe/Paris)
   - Recommends privacy settings (public/private)
   - Sanitizes sensitive information
   - **No tools required** - pure reasoning agent

3. **Translation Agent** (`agents/translation_agent.py`)
   - Translates English content → French (MANDATORY for French-speaking audience)
   - Preserves emojis and formatting
   - Adapts sports terminology appropriately
   - Maintains character limits (≤50 title, ≤500 description)
   - **Enabled via**: `TRANSLATION_ENABLED=true` (must be true in production)

### Data Flow

```
Strava Webhook → n8n → crew.py (stdin JSON)
  ↓
[Description Agent + MCP Tools] → Generate English content
  ↓
[Privacy Agent] → Validate & sanitize
  ↓
[Translation Agent] → Translate to French (MANDATORY)
  ↓
Final JSON (stdout) → n8n → Update Strava
```

### MCP Integration (Model Context Protocol)

The Description Agent uses **CrewAI's MCP DSL** to dynamically load tools at runtime:

- **Configuration**: Set `MCP_SERVER_URL` (comma-separated URLs supported)
- **Auto-discovery**: By default, the crew lets the MCP server advertise its available tools
- **Tool filtering**: Set `INTERVALS_MCP_TOOL_NAMES` (comma-separated) to pin a specific subset when you need deterministic access
- **DSL Syntax**: `https://mcp.server.com/path#ToolName` or `crewai-amp:...`

The system builds MCP references in `crew.py:_build_intervals_mcp_references()` and passes them to the agent via the `mcps` parameter. When no tool names are supplied, the raw MCP URL is forwarded so the server can return its catalogue. When no tool names are supplied, the raw MCP URL is forwarded so the server can return its catalogue.

### LLM Configuration

**Dual Authentication Support**:
1. **Basic Auth** (preferred): Set `OPENAI_API_AUTH_TOKEN` (base64-encoded credentials)
   - Injected via `litellm.completion` monkey-patch in `crew.py`
2. **API Key**: Set `OPENAI_API_KEY` (standard Bearer token)

**Endpoint Setup**:
- `OPENAI_API_BASE`: Custom endpoint URL (e.g., `https://ccproxy.emottet.com/copilot/v1`)
- `OPENAI_MODEL_NAME`: Model identifier (e.g., `claude-sonnet-4.5`, `gpt-5-mini`)
- **LiteLLM prefix**: Models are referenced as `openai/{model_name}` internally (except for ccproxy endpoints)

**Model Normalization**:

The system automatically normalizes model names based on the endpoint being used. This ensures compatibility across different ccproxy endpoints:

1. **Copilot Endpoint** (`/copilot/v1`):
   - Uses short model names: `gpt-5`, `gpt-5-mini`, `claude-sonnet-4.5`, `gemini-2.5-pro`
   - Example: `claude-sonnet-4.5` → `claude-sonnet-4.5` (unchanged)
   - System prompts: ✅ Enabled

2. **Codex Endpoint** (`/codex/v1`):
   - Only accepts: `gpt-5`, `gpt-5-codex`
   - Example: `claude-sonnet-4.5` → `gpt-5` (fallback)
   - System prompts: ❌ Disabled (automatically stripped)
   - Tool calls: ❌ Disabled for GPT-5 models (automatically skipped)

3. **Claude Endpoint** (`/claude/v1`):
   - Requires full versioned names: `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
   - Example: `claude-sonnet-4.5` → `claude-sonnet-4-5-20250929` (auto-mapped)
   - System prompts: ✅ Enabled

**Testing Model Normalization**:
```bash
# Test model normalization logic
python test_model_normalization.py

# Test system prompt handling
python test_system_prompt_handling.py

# Test full provider rotation chain
python test_provider_rotation.py
```

### Error Handling Philosophy

- **Defensive parsing**: All JSON parsing wrapped in try-except with safe defaults
- **Privacy-first fallbacks**: On errors, default to `should_be_private=True`
- **Stderr logging**: Debug/errors to stderr, clean JSON output to stdout
- **n8n compatibility**: Always output valid JSON, even on failure

## Critical Configuration

### Environment Variables (Required)

```bash
# LLM Configuration (required)
OPENAI_API_BASE=https://your-endpoint.com/v1
OPENAI_MODEL_NAME=claude-sonnet-4.5
OPENAI_API_AUTH_TOKEN=base64_token  # OR OPENAI_API_KEY=key

# MCP Server (required for Intervals.icu data)
MCP_SERVER_URL=https://mcp.emottet.com/metamcp/.../mcp?api_key=...

# MCP Enforcement (default: true)
# Set to false to allow running without MCP (not recommended for production)
REQUIRE_MCP=true

# Privacy Policy (required)
WORK_START_MORNING=08:30
WORK_END_MORNING=12:00
WORK_START_AFTERNOON=14:00
WORK_END_AFTERNOON=17:00

# Translation (CRITICAL - must be true in production)
TRANSLATION_ENABLED=true
TRANSLATION_TARGET_LANGUAGE=French
TRANSLATION_SOURCE_LANGUAGE=English
```

### Important Constants

- **Work hours timezone**: Always Europe/Paris (CET/CEST)
- **Title limit**: 50 characters (enforced by Strava)
- **Description limit**: 500 characters (enforced by Strava)

## Input/Output Contracts

### Input (stdin): Strava Webhook Format
```json
[{
  "object_type": "activity",
  "object_id": 16284886069,
  "object_data": {
    "id": 16284886069,
    "name": "Lunch Run",
    "distance": 12337,
    "moving_time": 3601,
    "type": "Run",
    "start_date_local": "2025-10-27T11:54:41Z",
    "average_heartrate": 141.6,
    "average_watts": 405.2
  }
}]
```

### Output (stdout): Final Result JSON
```json
{
  "activity_id": 16284886069,
  "title": "🏃 Sortie tempo 12.3K - Effort soutenu",
  "description": "Sortie tempo solide avec contrôle de l'allure...",
  "should_be_private": false,
  "privacy_check": {
    "approved": true,
    "during_work_hours": false,
    "issues": [],
    "reasoning": "No privacy issues found."
  },
  "workout_analysis": {
    "type": "Tempo Run",
    "metrics": {"average_pace": "4:53 /km", "average_hr": "141 bpm"}
  }
}
```

**Note**: Title and description are in French (mandatory for Strava audience).

## Code Modification Guidelines

### Adding New Agents
1. Create agent definition in `agents/` (follow existing pattern)
2. Add corresponding task in `tasks/`
3. Update `crew.py` to instantiate agent and task
4. Insert into sequential workflow (maintain Description → Privacy → Translation order if applicable)
5. Document expected input/output in task description

### Modifying Privacy Rules
- **Work hours**: Update environment variables in `.env`
- **PII detection**: Modify `privacy_agent.py` backstory with new rules
- **Always test** with activities that violate new rules
- **Safe defaults**: Ensure fallbacks still set `should_be_private=True`

### Changing LLM Configuration
- **Test endpoint first**: Use curl to verify connectivity
- **Update `.env.example`**: Document new parameters
- **Handle auth changes**: Modify `crew.py` initialization if needed
- **LiteLLM quirks**: Remember `openai/` prefix and `drop_params=True`

### Working with MCP Tools
- **Legacy tools** in `tools/` are fallback wrappers (mostly unused now)
- **Prefer CrewAI MCP DSL**: Use `mcps` parameter instead of manual `@tool` decorators
- **Tool discovery**: Check MCP server capabilities via curl before adding references
- **Error handling**: MCP tools should return error dicts, not raise exceptions

## Common Issues & Solutions

### "LLM Provider NOT provided"
- **Cause**: Missing or incorrect `OPENAI_API_BASE` / `OPENAI_MODEL_NAME`
- **Fix**: Verify environment variables are set and endpoint is reachable
- **Test**: `curl $OPENAI_API_BASE/models`

### "The requested model is not supported"
- **Cause**: Model name doesn't match the endpoint's expected format
- **Solution**: The system now automatically normalizes model names based on endpoint
- **Verification**:
  ```bash
  # Test model normalization
  python test_model_normalization.py

  # Test provider rotation chain
  python test_provider_rotation.py

  # Test real endpoint with curl
  curl -X POST https://ccproxy.emottet.com/copilot/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Basic $OPENAI_API_AUTH_TOKEN" \
    -d '{"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "test"}]}'
  ```
- **Manual override**: Set exact model name in `LLM_PROVIDER_ROTATION` if needed
- **Endpoint requirements**:
  - Copilot: Use short names (`gpt-5`, `claude-sonnet-4.5`)
  - Codex: Use `gpt-5` or `gpt-5-codex` only
  - Claude: Use full versions (`claude-sonnet-4-5-20250929`)

### MCP Connection Failures
- **Cause**: Invalid `MCP_SERVER_URL` or network issues
- **Fix**: Test URL with `python test_mcp_connectivity.py` or curl, check API key validity
- **Default behavior**: System will fail immediately if `REQUIRE_MCP=true` (default)
- **Testing mode**: Set `REQUIRE_MCP=false` to bypass check (not recommended in production)

### Activity Always Set to Private
- **Cause**: Work hours misconfiguration or timezone mismatch
- **Fix**: Verify `start_date_local` timezone matches Europe/Paris
- **Debug**: Check stderr logs for work hours detection reasoning

### JSON Parsing Errors
- **Cause**: LLM returning markdown-wrapped JSON (```json...```)
- **Fix**: Code already strips markdown markers (see `crew.py:232-239`)
- **Defensive**: All parsing has safe fallbacks

### Translation Not Applied
- **Cause**: `TRANSLATION_ENABLED=false` or translation agent failure
- **Fix**: Set `TRANSLATION_ENABLED=true` and check stderr logs
- **Critical**: Translation is MANDATORY in production (French audience)

### Python Version Issues
- **Requirement**: Python 3.10+
- **Check**: `python --version` inside activated venv
- **Fix**: Recreate venv with correct Python version

## Integration Notes

### n8n Workflow Integration
- **Command**: Use absolute path to venv Python: `/path/to/venv/bin/python crew.py`
- **Input**: Pass Strava webhook JSON via stdin
- **Output**: Parse stdout JSON for title, description, privacy settings
- **Error handling**: Check exit code (1 on failure) and `error` field in JSON

### Strava API Constraints
- **Title length**: Max 50 characters (enforced by API)
- **Description length**: Max 500 characters (enforced by API)
- **Privacy field**: Use `visibility: "only_me"` for private activities

## Development Best Practices

### Code Style
- **PEP 8** compliance
- **Type hints** for all function parameters/returns
- **Docstrings** (Google style) for all functions/classes
- **f-strings** for formatting
- **Descriptive names** (e.g., `activity_data`, `privacy_check`)

### Language Conventions
- **Code/comments**: English
- **User-facing output**: French (titles, descriptions for Strava)
- **Documentation**: French in README.md, English in code comments

### Testing Workflow
1. **MCP connectivity**: `python test_mcp_connectivity.py` - Verify MCP tools are accessible
2. **Manual test**: `cat input.json | python crew.py`
3. **Check stderr**: Verify agent execution flow
4. **Validate JSON**: Ensure stdout is valid JSON
5. **Verify French**: Confirm title/description are in French
6. **Test privacy**: Try activities during work hours (11:54 local time)

### Git Workflow
- **Never commit**: `venv/`, `.env`, API keys, personal data
- **Always commit**: `.env.example`, `requirements.txt`, test files
- **Commit messages**: Follow existing style (emoji + concise description)

## Performance Characteristics

- **Total execution time**: 10-60 seconds (depends on LLM endpoint speed)
- **MCP calls**: 1-3 per activity (can add latency)
- **LLM calls**: 3 per activity (Description, Privacy, Translation agents)
- **Bottleneck**: Usually LLM inference time on remote endpoints

## Security Considerations

- **PII protection**: Privacy Agent filters names, addresses, medical info
- **Work hours compliance**: Activities during 08:30-12:00 / 14:00-17:00 → private
- **Credential management**: All API keys/tokens via environment variables
- **Logging**: Never log API keys or full activity data to stderr
