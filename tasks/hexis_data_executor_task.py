"""Task for the Hexis Data Executor to retrieve data via tool calls.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict

from crewai import Task


def create_hexis_data_executor_task(
    agent: Any,
    week_start_date: str,
    retrieval_plan: Dict[str, Any],
) -> Task:
    """
    Create a task for the Executor to retrieve Hexis data.

    Args:
        agent: The Executor agent
        week_start_date: Start date of the week (ISO format YYYY-MM-DD)
        retrieval_plan: The data retrieval plan from the Supervisor

    Returns:
        Configured Task instance
    """
    # Calculate week end date
    start_dt = datetime.fromisoformat(week_start_date)
    end_dt = start_dt + timedelta(days=6)
    week_end_date = end_dt.strftime("%Y-%m-%d")

    # Format the plan for the task description
    plan_json = json.dumps(retrieval_plan, indent=2)

    description = f"""Execute the Hexis data retrieval plan and return raw data.

**YOUR ROLE**: You are the EXECUTOR. You EXECUTE tool calls and return raw data.
Do NOT analyze or interpret the data - the Reviewer agent will do that.

DATA RETRIEVAL PLAN FROM SUPERVISOR:
{plan_json}

YOUR TASK:
1. Execute the tool calls specified in the plan above
2. For the PRIMARY call: hexis_get_weekly_plan with start_date="{week_start_date}", end_date="{week_end_date}"
3. Collect all results
4. Return the raw data in structured format

EXECUTION INSTRUCTIONS:
- Call hexis_get_weekly_plan with the exact parameters
- If a tool call fails, record the error and continue
- Return ALL data received from the tools

OUTPUT REQUIREMENTS - RawHexisData JSON:

After executing the tool calls, return a JSON object:
```json
{{
  "week_start_date": "{week_start_date}",
  "week_end_date": "{week_end_date}",
  "tool_results": [
    {{
      "tool_name": "hexis_get_weekly_plan",
      "success": true,
      "result": {{ ... raw tool result ... }},
      "error_message": null
    }}
  ],
  "total_calls": 1,
  "successful_calls": 1,
  "weekly_plan_data": {{ ... extracted data from hexis_get_weekly_plan ... }},
  "retrieval_notes": "Successfully retrieved all planned data"
}}
```

CRITICAL:
- Execute the tool calls FIRST, then format the output
- Include the COMPLETE raw data from tools in the result
- The weekly_plan_data field should contain the main payload from hexis_get_weekly_plan

Return ONLY the JSON object after tool execution, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with raw Hexis data.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "tool_results": [...], "weekly_plan_data": {{...}}, ...}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
