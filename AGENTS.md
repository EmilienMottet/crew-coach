# Repository Guidelines

## Project Structure & Module Organization
The Python crews live in `agents/`, while task prompts and assembly helpers reside in `tasks/` and integrations in `tools/`. `crew.py` streams Strava activity payloads, `crew_mealy.py` coordinates weekly meal planning, and `server.py` exposes both flows over FastAPI. Tests and runnable demos live at the repository root (`test_*.py`, `.sh` scripts), and Docker assets (`Dockerfile`, `docker-compose.yml`) support containerised execution.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` — install the CrewAI stack locally.
- `python crew.py < input.json` — generate a Strava activity description from sample data.
- `python crew_mealy.py` — launch the meal-planning workflow for the upcoming week.
- `python server.py` or `uvicorn server:app --reload` — start the FastAPI gateway on port 8000.
- `docker compose run --rm crew < input.json` — run the Strava pipeline inside the container image.

## Coding Style & Naming Conventions
Use four-space indentation and keep functions typed, mirroring existing `pydantic` models and `TypedDict` usage. Prefer descriptive module names (`privacy_agent.py`, `meal_generation_agent.py`) and snake_case for variables. Keep docstrings concise and imperative, and write inline comments only for non-obvious orchestration steps or API assumptions.

## Testing Guidelines
Scenario tests are plain Python scripts; execute them with `python test_meal_planning.py`, `python test_mcp_connectivity.py`, or the FastAPI smoke script `bash test_server_meal_planning.sh`. When adding features, supplement with focused unit checks (Pytest-compatible) that cover success paths plus minimal failure cases, and ensure API responses still include the fallback payload on errors.

## Commit & Pull Request Guidelines
Follow the existing commit convention: lead with an emoji, add a short code in parentheses for the scope, then a lower-case verb phrase (e.g. `✨ feat(agents): add privacy safeguards`). PRs should summarise behavioural changes, list affected agents/tasks, include reproduction steps or curl examples for API updates, and link any related issues or n8n workflows. Attach screenshots or JSON snippets when responses change.

## Security & Configuration Tips
Keep `.env` local; never commit secrets or MCP keys. Validate that `OPENAI_API_AUTH_TOKEN` or `OPENAI_API_KEY` aligns with your deployment target, and confirm `MCP_SERVER_URL` points at the correct MetaMCP instance before running multi-agent tasks. Rotate Basic Auth tokens immediately if logs surface sensitive payloads.
