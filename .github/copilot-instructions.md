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
The project uses a **two-agent sequential workflow**:

1. **Activity Description Writer** (`agents/description_agent.py`)
   - Analyzes Intervals.icu training data
   - Generates concise titles (≤50 chars) and descriptions (≤500 chars)
   - Identifies workout types (tempo, intervals, easy runs, etc.)
   - Uses emoji appropriately for visual appeal

2. **Privacy & Compliance Officer** (`agents/privacy_agent.py`)
   - Validates generated content for PII (personally identifiable information)
   - Checks activity timing against work hours (08:30-12:00, 14:00-17:00 CET/CEST)
   - Recommends privacy settings (public/private)
   - Sanitizes sensitive information

### Data Flow
```
Strava Webhook → n8n → crew.py (stdin)
  ↓
Description Agent (+ MCP tools) → Generate content
  ↓
Privacy Agent → Validate & adjust
  ↓
Final JSON output (stdout) → n8n → Update Strava
```

### MCP Tools (`tools/`)
- `intervals_tools.py`: Wrappers for Intervals.icu data
  - `get_intervals_activity_details`: Full activity metrics
  - `get_intervals_activity_intervals`: Interval/segment breakdowns
  - `get_recent_intervals_activities`: Recent activity list
- `mcp_client.py`: JSON-RPC 2.0 client for MCP server communication

## Coding Conventions

### Language & Comments
- **Code**: Write in English (variables, functions, classes, docstrings)
- **Comments**: English for technical explanations
- **User-facing text**: French is acceptable in agent prompts/descriptions
- **Documentation**: French (README.md) for end-users, English for code

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
The system supports **two authentication patterns**:

1. **Basic Authentication** (preferred for custom endpoints):
   ```python
   OPENAI_API_AUTH_TOKEN=base64_encoded_credentials
   # Results in: Authorization: Basic <token>
   ```

2. **API Key** (standard OpenAI):
   ```python
   OPENAI_API_KEY=your-api-key
   # Results in: Authorization: Bearer <key>
   ```

### LLM Endpoint Setup
- Use `OPENAI_API_BASE` for custom endpoints (e.g., `https://ghcopilot.emottet.com/v1`)
- Use `OPENAI_MODEL_NAME` for model selection (e.g., `gpt-4`, `gpt-5-mini`)
- CrewAI uses **LiteLLM**, which requires the `openai/` prefix for model names
- Set `drop_params=True` to ignore unsupported parameters

### Important Notes
- The `api_key` parameter in LLM initialization may be a dummy value when using Basic Auth
- LiteLLM's `completion` function is monkey-patched to inject custom headers
- Always test endpoint connectivity before deploying changes

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
OPENAI_MODEL_NAME=gpt-4
OPENAI_API_AUTH_TOKEN=base64_token  # OR OPENAI_API_KEY

# MCP Server
MCP_SERVER_URL=https://mcp.emottet.com/metamcp/.../mcp?api_key=...

# Privacy Policy
WORK_START_MORNING=08:30
WORK_END_MORNING=12:00
WORK_START_AFTERNOON=14:00
WORK_END_AFTERNOON=17:00
```

### Optional
- `OPENAI_API_KEY`: Alternative to `OPENAI_API_AUTH_TOKEN`

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
Structured JSON result:
```json
{
  "activity_id": 16284886069,
  "title": "🏃 12.3K Tempo Run - Strong Effort",
  "description": "Solid tempo run with controlled pace...",
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
  }
}
```

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
1. Add wrapper function in `tools/intervals_tools.py` or create new file
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
2. **Respect** the two-agent architecture (don't add agents without discussion)
3. **Follow** error handling patterns (try-except, fallback values, stderr logging)
4. **Test** MCP and LLM connectivity before debugging agent logic
5. **Document** any new environment variables or configuration options
6. **Maintain** compatibility with n8n workflow expectations (stdin/stdout JSON)
