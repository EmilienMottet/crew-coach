"""Task for the Description Supervisor to plan data retrieval.

Part of the Supervisor/Executor/Reviewer pattern for DESCRIPTION.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task


def create_description_supervisor_task(agent: Any, activity_data: Dict[str, Any]) -> Task:
    """
    Create a task for the Supervisor to plan activity data retrieval.

    Args:
        agent: The Supervisor agent
        activity_data: Raw activity data from Strava webhook

    Returns:
        Configured Task instance
    """
    object_data = activity_data.get("object_data", {})
    strava_id = object_data.get("id", "unknown")
    activity_type = object_data.get("type", "Run")
    distance = object_data.get("distance", 0) / 1000  # Convert to km
    moving_time = object_data.get("moving_time", 0)
    start_date = object_data.get("start_date_local", "")

    # Extract date for Intervals.icu lookup
    start_date_str = start_date.split('T')[0] if start_date and 'T' in start_date else start_date

    description = f"""Plan the data retrieval strategy for creating an activity description.

**YOUR ROLE**: You are the SUPERVISOR. You PLAN the data retrieval but do NOT execute it.
A separate Executor agent will make the actual tool calls based on your plan.

ACTIVITY METADATA:
- Strava ID: {strava_id}
- Activity Type: {activity_type}
- Distance: {distance:.2f} km
- Moving Time: {moving_time / 60:.0f} minutes
- Date: {start_date_str}

RAW STRAVA DATA:
{json.dumps(object_data, indent=2)[:2000]}

YOUR TASK:
1. Analyze the activity metadata to understand what kind of workout this was
2. Plan which Intervals.icu tools should be called
3. Specify exact parameters for each tool call
4. Identify key data points to focus on for the description

OUTPUT REQUIREMENTS - ActivityDataRetrievalPlan JSON:

Return a JSON object with this exact structure:
```json
{{
  "activity_id": "{strava_id}",
  "activity_date": "{start_date_str}",
  "activity_type": "{activity_type}",
  "tool_calls": [
    {{
      "tool_name": "IntervalsIcu__get_activities",
      "parameters": {{
        "start_date": "{start_date_str}",
        "end_date": "{start_date_str}",
        "limit": 10
      }},
      "purpose": "Find the Intervals.icu activity ID matching this Strava activity",
      "priority": 1
    }},
    {{
      "tool_name": "IntervalsIcu__get_activity_details",
      "parameters": {{
        "activity_id": "TO_BE_FILLED_BY_EXECUTOR"
      }},
      "purpose": "Get detailed training metrics (pace, HR, power, TSS, etc.)",
      "priority": 2
    }},
    {{
      "tool_name": "IntervalsIcu__get_activity_streams",
      "parameters": {{
        "activity_id": "TO_BE_FILLED_BY_EXECUTOR",
        "stream_types": "core_temperature,skin_temperature"
      }},
      "purpose": "Get CORE body temperature data for thermal stress analysis",
      "priority": 3
    }}
  ],
  "data_focus": ["pace", "heart_rate", "power", "core_temperature", "workout_structure"],
  "description_style": "engaging"
}}
```

GUIDELINES:
- Always include the 3 standard tool calls in order
- Adapt data_focus based on activity type (Run → pace, Ride → power)
- The Executor will fill in the activity_id from the first tool call's result
- Keep the plan simple and focused

Return ONLY the JSON object, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with data retrieval plan.
Format: {{"activity_id": "{strava_id}", "activity_date": "{start_date_str}", "activity_type": "{activity_type}", "tool_calls": [...], "data_focus": [...], "description_style": "..."}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
