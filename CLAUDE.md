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
MCP_API_KEY=sk_mt_knHh0F3WLsJcrHpTFT3VC0ChzOpfGzGnp4CdjKzFIPLDN0zHWn2XR6YTIe0KHsmk

# ============================================
# MCP Servers for Strava Description Crew (crew.py)
# ============================================

# Strava (19 tools): Activity data, athlete profile, segments, routes
STRAVA_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/SocialNetworkSport/mcp

# Intervals.icu (9 tools): Training data, activity intervals, wellness
INTERVALS_MCP_SERVER_URL=https://mcp.emottet.com/metamcp/IntervalsIcu/mcp

# Music/Spotify: Now provided by n8n (spotify_recently_played field)
# Note: MUSIC_MCP_SERVER_URL is no longer needed - n8n fetches Spotify data

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
    # Note: Music data now comes from n8n (spotify_recently_played field)
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
    ├── Music/Spotify (n8n integration)
    │   └── Data provided via spotify_recently_played field
    │       No MCP server needed - n8n fetches directly from Spotify API
    ├── Meteo (11 OpenWeatherMap tools)
    │   ├── OpenWeatherMap__get-current-weather
    │   ├── OpenWeatherMap__get-daily-forecast
    │   └── ...
    └── Toolbox (14 utility tools)
        ├── Time__get_current_time
        ├── Fetch__fetch
        └── ...

Meal Planning Crew (crew_mealy.py) connects to:
    ├── Food (26 nutrition tools)
    │   ├── hexis__hexis_get_nutrition_trend
    │   ├── hexis__hexis_get_meal_inspiration
    │   ├── hexis__hexis_search_passio_foods (query, limit) - Primary food search
    │   ├── hexis__hexis_search_passio_foods_advanced (query, limit, search_brands)
    │   ├── hexis__hexis_get_passio_food_details (ref_code) - Get full food details
    │   ├── hexis__hexis_search_passio_barcode (barcode) - Barcode lookup
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
  - **Food Search**: Use `hexis_search_passio_foods` with `limit=5` to avoid context overflow
  - **IMPORTANT**: Always specify `limit` parameter (default injected: 5). Without limit, API returns 100+ results (~150K chars)
- **Toolbox**: `Tool__method_name` (varies by tool)

### MCP Tool Distribution by Agent

**Strava Description Crew (`crew.py`):**
- **Description Agent**: Strava + Intervals.icu + Weather + Toolbox
  - Needs comprehensive data to generate descriptions
  - Weather context enriches activity narratives
  - Toolbox for time/date utilities
- **Music Agent**: No MCP tools (data from n8n)
  - Analyzes spotify_recently_played data from n8n
  - Enriches description with soundtrack information
  - n8n fetches Spotify data before calling CrewAI
- **Privacy Agent**: No MCP tools (pure reasoning)
- **Translation Agent**: No MCP tools (pure reasoning)

**Meal Planning Crew (`crew_mealy.py`):**

**HEXIS_ANALYSIS (Supervisor/Executor/Reviewer pattern):**
- **HEXIS_DATA_SUPERVISOR**: No MCP tools (pure reasoning)
  - Plans which Hexis tools to call and what data to retrieve
  - Creates `HexisDataRetrievalPlan` schema
  - Can use thinking models (no tool calls)
- **HEXIS_DATA_EXECUTOR**: Hexis tools (`hexis_get_weekly_plan`)
  - Executes the planned tool calls
  - Returns raw `RawHexisData` schema
  - NO thinking models (has_tools=True)
- **HEXIS_ANALYSIS_REVIEWER**: No MCP tools (pure reasoning)
  - Analyzes raw data and creates `HexisWeeklyAnalysis`
  - Can use thinking models (no tool calls)

**MEAL_GENERATION (Supervisor/Executor/Reviewer pattern):**
- **MEAL_PLANNING_SUPERVISOR**: No MCP tools (pure reasoning)
  - Designs meals and creates `MealPlanTemplate` schema
  - Can use thinking models (no tool calls)
- **INGREDIENT_VALIDATION_EXECUTOR**: `hexis_search_passio_foods`
  - Validates ingredients via Passio API
  - Returns `ValidatedIngredientsList` schema
  - NO thinking models (has_tools=True)
- **MEAL_RECIPE_REVIEWER**: No MCP tools (pure reasoning)
  - Calculates macros and finalizes meals
  - Can use thinking models (no tool calls)

**Other Agents:**
- **Weekly Structure Agent**: No MCP tools (pure reasoning)
- **Nutritional Validation Agent**: No MCP tools (pure reasoning)
- **Mealy Integration Agent**: Hexis (`hexis_log_meal` composite tool)
  - Integrates with Hexis meal planning system
  - NO thinking models (has_tools=True)

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

### Supervisor/Executor/Reviewer Pattern

**Problem**: Thinking models (e.g., `claude-opus-4-5-thinking-*`) hallucinate tool calls when agents have MCP tools. They simulate ReAct reasoning in their content output while returning `tool_calls=None`, causing the agent to fail.

**Solution**: Separate agents that need tools into a 3-agent pattern:

```
┌─────────────────────────────────────────────────────────────────────┐
│  SUPERVISOR (Pure Reasoning)                                        │
│  - Plans what data/actions are needed                               │
│  - Creates structured plan (Pydantic schema)                        │
│  - NO tools → thinking models allowed                               │
│  - Model tier: complex                                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (plan schema)
┌─────────────────────────────────────────────────────────────────────┐
│  EXECUTOR (Tool Execution Only)                                     │
│  - Executes tool calls as specified by Supervisor                   │
│  - Returns raw results (Pydantic schema)                            │
│  - HAS tools → thinking models EXCLUDED (has_tools=True)            │
│  - Model tier: simple (fast, reliable tool execution)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (raw data schema)
┌─────────────────────────────────────────────────────────────────────┐
│  REVIEWER (Pure Reasoning)                                          │
│  - Analyzes raw data from Executor                                  │
│  - Creates final structured output (Pydantic schema)                │
│  - NO tools → thinking models allowed                               │
│  - Model tier: intermediate                                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation in crew_mealy.py:**

```python
# LLM creation with has_tools parameter
self.hexis_data_supervisor_llm = self._create_agent_llm(
    agent_name="HEXIS_DATA_SUPERVISOR",
    default_model=complex_model_name,
    has_tools=False,  # Pure reasoning → thinking models OK
)

self.hexis_data_executor_llm = self._create_agent_llm(
    agent_name="HEXIS_DATA_EXECUTOR",
    default_model=simple_model_name,
    has_tools=True,  # Has tools → thinking models EXCLUDED
)

self.hexis_analysis_reviewer_llm = self._create_agent_llm(
    agent_name="HEXIS_ANALYSIS_REVIEWER",
    default_model=intermediate_model_name,
    has_tools=False,  # Pure reasoning → thinking models OK
)
```

**Inter-Agent Communication via Pydantic Schemas:**

```python
# schemas.py defines the data flow contracts
class HexisDataRetrievalPlan(BaseModel):
    """SUPERVISOR output → EXECUTOR input"""
    week_start_date: str
    week_end_date: str
    tool_calls: List[HexisToolCall]
    analysis_focus: List[str]

class RawHexisData(BaseModel):
    """EXECUTOR output → REVIEWER input"""
    week_start_date: str
    tool_results: List[HexisToolResult]
    weekly_plan_data: Optional[Dict[str, Any]]

class HexisWeeklyAnalysis(BaseModel):
    """REVIEWER output → Final result"""
    training_load_summary: TrainingLoadSummary
    daily_energy_needs: Dict[str, DailyEnergyNeeds]
    daily_macro_targets: Dict[str, DailyMacroTargets]
```

**Current Implementations:**
1. **HEXIS_ANALYSIS**: `hexis_data_supervisor` → `hexis_data_executor` → `hexis_analysis_reviewer`
2. **MEAL_GENERATION**: `meal_planning_supervisor` → `ingredient_validation_executor` → `meal_recipe_reviewer`

**When to Use This Pattern:**
- Agent mixes reasoning with tool calls
- Rate limiting issues (Executors use simple/cheap models)
- Need to support thinking models for complex reasoning
- Want to separate concerns for easier debugging

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
- `OPENAI_API_BASE`: Custom endpoint URL (e.g., `https://ccproxy.emottet.com/v1`)
- `OPENAI_MODEL_NAME`: Model identifier (e.g., `claude-sonnet-4.5`, `gpt-5-mini`)
- **LiteLLM prefix**: Models are referenced as `openai/{model_name}` internally (except for ccproxy endpoints)

**Endpoint**:

All models are available through a single unified endpoint: `https://ccproxy.emottet.com/v1`

Supported models include: `gpt-5`, `gpt-5-mini`, `claude-sonnet-4.5`, `claude-sonnet-4-5-20250929`, `claude-haiku-4.5`, `gemini-2.5-pro`, etc.

**Startup Logs**:
- On startup, you'll see logs like:
  ```
  🤖 MUSIC Agent:
     Model: claude-haiku-4-5
     Endpoint: https://ccproxy.emottet.com/v1
  ```

**Testing Model Normalization**:
```bash
# Test with real endpoint using curl
curl -X POST https://ccproxy.emottet.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "test"}]}'
```

### Provider Rotation and Automatic Disabling

**Automatic Quota Error Handling with Persistent Blacklist**:

When a provider reaches its quota or rate limit (HTTP 402/429 or rate limit error), the system automatically:
1. **Disables the provider** to prevent repeated failed attempts
2. **Rotates to the next available provider** in the chain
3. **Persists disabled state** to JSON file (survives process restarts)
4. **Increments strike count** with exponential backoff TTL
5. **Resets strikes** after successful call (provider recovers)

**Strike-based TTL Progression (Exponential Backoff)**:

| Strike # | TTL | Description |
|----------|-----|-------------|
| 1 | 1 hour | First failure |
| 2 | 6 hours | Second failure |
| 3 | 24 hours | Third failure |
| 4+ | 72 hours | Repeated failures (cap) |

**Configuration**:
```bash
# Enable/disable provider rotation (default: true)
ENABLE_LLM_PROVIDER_ROTATION=true

# Persistent blacklist file (default: .disabled_providers.json)
DISABLED_PROVIDERS_FILE=.disabled_providers.json

# Base TTL in seconds for strike #1 (default: 3600 = 1 hour)
# PROVIDER_BASE_TTL_SECONDS=3600

# Maximum TTL cap in seconds (default: 259200 = 72 hours)
# PROVIDER_MAX_TTL_SECONDS=259200

# Disable persistence (use in-memory only, like old behavior)
# DISABLE_PERSISTENT_BLACKLIST=false

# Define rotation chain (optional)
LLM_PROVIDER_ROTATION="fallback1|gpt-5-mini|https://endpoint1.com/v1|OPENAI_API_KEY"
```

**Blacklist File Format** (`.disabled_providers.json`):
```json
{
  "version": 1,
  "providers": {
    "claude-sonnet-4.5@https://ccproxy.emottet.com/v1": {
      "disabled_at": 1732900000.0,
      "strike_count": 2,
      "last_error": "429 rate limit exceeded",
      "ttl_seconds": 21600
    }
  }
}
```

**How it works (Persistent with Strikes)**:
```
Execution 1:
  Call 1: Provider A → 429 error → DISABLED (strike #1, 1h TTL) → Saved to file
  Call 2: Provider A SKIPPED → Fallback used

Execution 2 (new process, within 1h):
  Call 1: Provider A SKIPPED (loaded from file) → Fallback used directly

Execution 3 (after 1h):
  Call 1: Provider A available again → Try provider A
  Call 2: Provider A → 429 error → DISABLED (strike #2, 6h TTL) → Saved to file

Execution 4 (same provider succeeds after recovery):
  Call 1: Provider A → SUCCESS → Strikes reset to 0, removed from blacklist
```

**Detected error types**:
- HTTP status codes: `402` (quota exceeded), `429` (rate limit)
- Exception classes: `RateLimitError`, `RateLimitException`
- Error message keywords: `"rate limit"`, `"quota"`, `"429"`
- Truncated responses (incomplete output)

**Logs (Persistent Blacklist)**:
```
🚫 Blacklist: provider DISABLED (strike #2)
   Provider: claude-sonnet-4.5@https://ccproxy.emottet.com/v1
   Error: 429 rate limit exceeded
   TTL: 6 hours (until 2024-11-30 02:30:00)
   Next TTL will be: 24 hours

⏭️  DESCRIPTION: SKIPPING intermediate:claude-sonnet-4-5-20250929 (claude-sonnet-4-5-20250929)
   Reason: Disabled (strike #2), 5 hours remaining

✅ Blacklist: provider claude-sonnet-4.5@... SUCCESS, resetting 2 strike(s)
```

**Testing**:
```bash
# Test the automatic disabling system
python test_provider_rotation_disable.py

# Expected output:
# ✅ All rate limit detection tests passed!
# ✅ All provider disabling tests passed!
# ✅ All TTL re-enablement tests passed!
# ✅ All shared state tests passed!
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

# LLM Provider Rotation (optional, default: true)
ENABLE_LLM_PROVIDER_ROTATION=true
PROVIDER_DISABLED_TTL_SECONDS=0  # 0 = disabled for entire execution (default)

# MCP Server (required for Intervals.icu data)
MCP_SERVER_URL=https://mcp.emottet.com/metamcp/.../mcp?api_key=...

# MCP Enforcement (default: true)
# Set to false to allow running without MCP (not recommended for production)
REQUIRE_MCP=true

# Passio Food Search (optional)
# Default limit for hexis_search_passio_foods to avoid context overflow
PASSIO_DEFAULT_LIMIT=5  # default: 5, max recommended: 10

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

### Input (stdin): n8n Enhanced Format
```json
{
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
  },
  "spotify_recently_played": {
    "items": [
      {
        "track": {
          "name": "Track Name",
          "artists": [{"name": "Artist Name"}]
        },
        "played_at": "2025-10-27T11:55:00Z"
      }
    ]
  }
}
```

**Note**: n8n merges Strava activity data with Spotify playback history before sending to CrewAI.

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

**Note**: For detailed troubleshooting, see `TROUBLESHOOTING.md`.

### "Error: the Action Input is not a valid key, value dictionary"
- **Cause**: LLM passing a list instead of a dictionary to MCP tool
- **Fix**: Implemented automatic input validation wrapper (`mcp_tool_wrapper.py`)
- **Prevention**: Enhanced agent prompts with explicit examples
- **Testing**: Run `python test_tool_wrapper.py` to verify wrapper functionality
- **Details**: See `TROUBLESHOOTING.md` for full explanation

### "LLM Provider NOT provided"
- **Cause**: Missing or incorrect `OPENAI_API_BASE` / `OPENAI_MODEL_NAME`
- **Fix**: Verify environment variables are set and endpoint is reachable
- **Test**: `curl $OPENAI_API_BASE/models`

### "The requested model is not supported"
- **Cause**: Model name doesn't match the endpoint's expected format
- **Solution**: The system automatically normalizes model names based on endpoint
- **Verification**:

  ```bash
  # Test with real endpoint using curl
  curl -X POST https://ccproxy.emottet.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "test"}]}'
  ```
- **Manual override**: Set exact model name in `LLM_PROVIDER_ROTATION` if needed

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

#### Strava Description Workflow
- **Command**: Use absolute path to venv Python: `/path/to/venv/bin/python crew.py`
- **Input**: Pass Strava webhook JSON via stdin
- **Output**: Parse stdout JSON for title, description, privacy settings
- **Error handling**: Check exit code (1 on failure) and `error` field in JSON

#### Weekly Meal Plan Workflow (with Intervals.icu Integration)
- **Workflow File**: `n8n/workflows/meal-planning-weekly.json`
- **Workflow ID**: `VSOAzIak3SfoMqVR`
- **Schedule**: Every Sunday at 20:00 (Europe/Paris)

**Workflow Steps**:
1. **Calculate Week Dates**: Computes next week's Monday-Sunday range
2. **Fetch Training Events**: Retrieves planned workouts from intervals.icu API
3. **Calculate New Training Times**: Applies scheduling rules:
   - **Monday**: Running → 12:00-14:00, Cycling → 18:00
   - **Tuesday/Wednesday**: Longest workout → 12:00-14:00, 2nd workout → 18:00
   - **Thursday/Friday**: Same as Monday
   - **Saturday/Sunday**: Longest workout → 09:00, 2nd workout → 17:00
4. **Check Updates Needed**: Skips update step if no changes required
5. **Update Training Time**: Updates intervals.icu via PUT API calls
6. **Merge & Continue**: Aggregates results
7. **Generate Meal Plan**: Calls `crew_mealy.py` via HTTP (crew:8000)
8. **Notifications**: Telegram notifications on success/failure

**Required Environment Variables**:
```bash
# Intervals.icu Configuration
INTERVALS_ICU_ATHLETE_ID=i55249
INTERVALS_ICU_API_KEY=QVBJX0tFWTpyMjl1Mnppamxlcjh4ZzJlNm5lazRnOGw=  # Base64 encoded

# Meal Plan Service
# Ensure crew_mealy.py is running on crew:8000
```

**Note**: Telegram chat ID (258732739) is hardcoded in the workflow for simplicity.

**API Authentication**:
- The `INTERVALS_ICU_API_KEY` is a Base64-encoded string in format: `API_KEY:your_api_key`
- Used with Basic Auth header: `Authorization: Basic QVBJX0tFWTpyMjl1Mnppamxlcjh4ZzJlNm5lazRnOGw=`

**Importing Workflow to n8n**:
```bash
# Option 1: Import via n8n UI
# Settings → Import from File → Select n8n/workflows/meal-planning-weekly.json

# Option 2: Import via n8n API
curl -X POST http://localhost:5678/api/v1/workflows \
  -H "Content-Type: application/json" \
  -H "X-N8N-API-KEY: $N8N_API_KEY" \
  -d @n8n/workflows/meal-planning-weekly.json
```

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
