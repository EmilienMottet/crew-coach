"""Task for the Hexis Data Supervisor to plan data retrieval.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from crewai import Task


def create_hexis_data_supervisor_task(agent: Any, week_start_date: str) -> Task:
    """
    Create a task for the Supervisor to plan Hexis data retrieval.

    Args:
        agent: The Supervisor agent
        week_start_date: Start date of the week to plan (ISO format YYYY-MM-DD)

    Returns:
        Configured Task instance
    """
    # Calculate week end date
    start_dt = datetime.fromisoformat(week_start_date)
    end_dt = start_dt + timedelta(days=6)
    week_end_date = end_dt.strftime("%Y-%m-%d")

    description = f"""Plan the data retrieval strategy for Hexis training data analysis.

**YOUR ROLE**: You are the SUPERVISOR. You PLAN the data retrieval but do NOT execute it.
A separate Executor agent will make the actual tool calls based on your plan.

DATE RANGE:
- Start date: {week_start_date}
- End date: {week_end_date}

YOUR TASK:
1. Analyze what data is needed for comprehensive nutritional planning
2. Create a retrieval plan specifying which Hexis tools to call
3. Prioritize the tool calls (most important first)
4. Identify key analysis focus areas

OUTPUT REQUIREMENTS - HexisDataRetrievalPlan JSON:

Return a JSON object with this exact structure:
```json
{{
  "week_start_date": "{week_start_date}",
  "week_end_date": "{week_end_date}",
  "tool_calls": [
    {{
      "tool_name": "hexis_get_weekly_plan",
      "parameters": {{
        "start_date": "{week_start_date}",
        "end_date": "{week_end_date}"
      }},
      "purpose": "Retrieve complete weekly training schedule and Hexis nutritional targets",
      "priority": 1
    }}
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
- Include both training AND nutrition data retrieval
- The Executor will execute these tool calls in order of priority
- Be specific about parameters (dates must be in YYYY-MM-DD format)

Return ONLY the JSON object, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with data retrieval plan.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "tool_calls": [...], "analysis_focus": [...], "special_considerations": "..."}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
