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

    description = f"""Execute the Hexis data retrieval plan.

**YOUR ROLE**: You are the EXECUTOR. You EXECUTE tool calls.
The raw data is automatically captured by Python - you do NOT need to copy it.

DATA RETRIEVAL PLAN FROM SUPERVISOR:
{plan_json}

YOUR TASK - Execute {num_batches} BATCHED tool call(s):
{batch_instructions}

EXECUTION INSTRUCTIONS:
1. Execute EACH tool call in the plan (there are {num_batches} batches)
2. For each batch, call hexis_get_weekly_plan with the specified start_date and end_date
3. After EACH call, note if it succeeded or failed
4. Continue until all {num_batches} batches are executed

**IMPORTANT**: The raw data is captured automatically by Python.
You do NOT need to copy or return the full data in your response.
Just confirm the calls were made.

OUTPUT REQUIREMENTS - Execution Summary JSON:

After executing ALL tool calls, return a BRIEF JSON confirmation:
```json
{{
  "week_start_date": "{week_start_date}",
  "week_end_date": "{week_end_date}",
  "batches_executed": [
    {{"batch": 1, "start_date": "...", "end_date": "...", "success": true}},
    {{"batch": 2, "start_date": "...", "end_date": "...", "success": true}},
    {{"batch": 3, "start_date": "...", "end_date": "...", "success": true}}
  ],
  "total_calls": {num_batches},
  "successful_calls": {num_batches},
  "retrieval_notes": "Successfully executed all {num_batches} batch(es)"
}}
```

CRITICAL:
- Execute ALL {num_batches} tool call(s)
- Do NOT copy raw data into your response (it's captured automatically)
- Just return a brief confirmation of which batches succeeded/failed
- Keep your response under 1000 characters

Return ONLY the brief JSON confirmation, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Brief JSON confirmation of tool execution for all {num_batches} batch(es).
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "batches_executed": [...], "total_calls": {num_batches}, "successful_calls": N}}
NO raw data, NO markdown fences, ONLY brief JSON confirmation under 1000 chars.""",
    )
