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
    # Extract dates from plan (may span different ranges with batching)
    week_end_date = retrieval_plan.get("week_end_date", week_start_date)

    # Count batches from the plan
    tool_calls = retrieval_plan.get("tool_calls", [])
    num_batches = len(tool_calls)

    # Format the plan for the task description
    plan_json = json.dumps(retrieval_plan, indent=2)

    # Build batch execution instructions
    batch_instructions = ""
    for i, tc in enumerate(tool_calls, 1):
        params = tc.get("parameters", {})
        batch_instructions += f"""
Batch {i}: hexis_get_weekly_plan(start_date="{params.get('start_date')}", end_date="{params.get('end_date')}")"""

    description = f"""Execute the Hexis data retrieval plan and return raw data.

**YOUR ROLE**: You are the EXECUTOR. You EXECUTE tool calls and return raw data.
Do NOT analyze or interpret the data - the Reviewer agent will do that.

DATA RETRIEVAL PLAN FROM SUPERVISOR:
{plan_json}

YOUR TASK - Execute {num_batches} BATCHED tool call(s):
{batch_instructions}

EXECUTION INSTRUCTIONS:
1. Execute EACH tool call in the plan (there are {num_batches} batches)
2. For each batch, call hexis_get_weekly_plan with the specified start_date and end_date
3. Record each result in tool_results array
4. If a tool call fails, record the error and continue to next batch
5. Return ALL data received from ALL batches

OUTPUT REQUIREMENTS - RawHexisData JSON:

After executing ALL tool calls, return a JSON object:
```json
{{
  "week_start_date": "{week_start_date}",
  "week_end_date": "{week_end_date}",
  "tool_results": [
    {{
      "tool_name": "hexis__hexis_get_weekly_plan",
      "success": true,
      "result": {{ ... raw tool result for batch 1 ... }},
      "error_message": null
    }},
    {{
      "tool_name": "hexis__hexis_get_weekly_plan",
      "success": true,
      "result": {{ ... raw tool result for batch 2 ... }},
      "error_message": null
    }}
  ],
  "total_calls": {num_batches},
  "successful_calls": {num_batches},
  "weekly_plan_data": null,
  "retrieval_notes": "Successfully retrieved all {num_batches} batch(es)"
}}
```

CRITICAL:
- Execute ALL {num_batches} tool call(s) FIRST, then format the output
- Include the COMPLETE raw data from EACH batch in tool_results
- The weekly_plan_data can be null - the Reviewer will aggregate from tool_results
- Each tool_result should contain the full "data" with "days" array from Hexis

Return ONLY the JSON object after tool execution, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with raw Hexis data from all {num_batches} batch(es).
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "tool_results": [...], "total_calls": {num_batches}, ...}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
