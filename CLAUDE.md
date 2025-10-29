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
# Test with sample input
cat input.json | python crew.py

# Test privacy agent specifically
python test_privacy_agent.py

# Test translation workflow
bash test_translation.sh
```

### Debugging LLM/MCP Connectivity
```bash
# Test LLM endpoint
curl -X POST $OPENAI_API_BASE/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic $OPENAI_API_AUTH_TOKEN" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'

# Test MCP server
curl "$MCP_SERVER_URL"

# List available models
curl $OPENAI_API_BASE/models
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
- `OPENAI_API_BASE`: Custom endpoint URL (e.g., `https://ghcopilot.emottet.com/v1`)
- `OPENAI_MODEL_NAME`: Model identifier (e.g., `gpt-4`, `gpt-5-mini`)
- **LiteLLM prefix**: Models are referenced as `openai/{model_name}` internally

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
OPENAI_MODEL_NAME=gpt-4
OPENAI_API_AUTH_TOKEN=base64_token  # OR OPENAI_API_KEY=key

# MCP Server (required for Intervals.icu data)
MCP_SERVER_URL=https://mcp.emottet.com/metamcp/.../mcp?api_key=...

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

### MCP Connection Failures
- **Cause**: Invalid `MCP_SERVER_URL` or network issues
- **Fix**: Test URL with curl, check API key validity
- **Fallback**: System logs warning but continues (Description Agent operates without live data)

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
1. **Manual test**: `cat input.json | python crew.py`
2. **Check stderr**: Verify agent execution flow
3. **Validate JSON**: Ensure stdout is valid JSON
4. **Verify French**: Confirm title/description are in French
5. **Test privacy**: Try activities during work hours (11:54 local time)

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
