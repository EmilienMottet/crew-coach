"""Task for the Hexis Data Supervisor to plan data retrieval.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from crewai import Task


def create_hexis_data_supervisor_task(
    agent: Any, week_start_date: str, num_days: int = 7
) -> Task:
    """
    Create a task for the Supervisor to plan Hexis data retrieval.

    Args:
        agent: The Supervisor agent
        week_start_date: Start date of the week to plan (ISO format YYYY-MM-DD)
        num_days: Number of days to plan (default 7)

    Returns:
        Configured Task instance
    """
    # Calculate week end date based on num_days
    start_dt = datetime.fromisoformat(week_start_date)
    end_dt = start_dt + timedelta(days=num_days - 1)
    week_end_date = end_dt.strftime("%Y-%m-%d")

    # Generate batches (max 3 days per batch to avoid large JSON responses)
    BATCH_SIZE = 3
    batches = []
    current_start = start_dt
    priority = 1
    while current_start <= end_dt:
        batch_end = min(current_start + timedelta(days=BATCH_SIZE - 1), end_dt)
        batches.append(
            {
                "start_date": current_start.strftime("%Y-%m-%d"),
                "end_date": batch_end.strftime("%Y-%m-%d"),
                "priority": priority,
            }
        )
        priority += 1
        current_start = batch_end + timedelta(days=1)

    # Build tool_calls JSON for prompt
    tool_calls_json = ",\n    ".join(
        f"""{{
      "tool_name": "hexis__hexis_get_weekly_plan",
      "parameters": {{
        "start_date": "{b['start_date']}",
        "end_date": "{b['end_date']}"
      }},
      "purpose": "Retrieve data for {b['start_date']} to {b['end_date']}",
      "priority": {b['priority']}
    }}"""
        for b in batches
    )

    num_batches = len(batches)
    batch_info = (
        f"Split into {num_batches} batches of max {BATCH_SIZE} days each to avoid large responses."
        if num_batches > 1
        else "Single batch (3 days or less)."
    )

    description = f"""Plan the data retrieval strategy for Hexis training data analysis.

**YOUR ROLE**: You are the SUPERVISOR. You PLAN the data retrieval but do NOT execute it.
A separate Executor agent will make the actual tool calls based on your plan.

DATE RANGE:
- Start date: {week_start_date}
- End date: {week_end_date}
- Total days: {num_days}
- {batch_info}

YOUR TASK:
1. Analyze what data is needed for comprehensive nutritional planning
2. Create a retrieval plan specifying which Hexis tools to call
3. The tool_calls are PRE-BATCHED for you - use them exactly as shown
4. Identify key analysis focus areas

OUTPUT REQUIREMENTS - HexisDataRetrievalPlan JSON:

Return a JSON object with this exact structure:
```json
{{
  "week_start_date": "{week_start_date}",
  "week_end_date": "{week_end_date}",
  "tool_calls": [
    {tool_calls_json}
  ],
  "analysis_focus": [
    "training_load",
    "daily_energy_needs",
    "macro_targets",
    "recovery_status",
    "workout_timing"
  ],
  "special_considerations": "Focus on carbohydrate periodization based on training intensity"
}}
```

GUIDELINES:
- For meal planning, hexis_get_weekly_plan is the PRIMARY data source
- The Executor will execute ALL {num_batches} tool call(s) and aggregate results
- Be specific about parameters (dates must be in YYYY-MM-DD format)
- DO NOT modify the tool_calls - they are pre-batched to avoid API limits

Return ONLY the JSON object, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with data retrieval plan.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "tool_calls": [...], "analysis_focus": [...], "special_considerations": "..."}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
