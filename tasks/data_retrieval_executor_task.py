"""Task for the Data Retrieval Executor to retrieve activity data via tool calls.

Part of the Supervisor/Executor/Reviewer pattern for DESCRIPTION.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task


def create_data_retrieval_executor_task(
    agent: Any,
    activity_data: Dict[str, Any],
    retrieval_plan: Dict[str, Any],
) -> Task:
    """
    Create a task for the Executor to retrieve activity data.

    Args:
        agent: The Executor agent
        activity_data: Raw activity data from Strava webhook
        retrieval_plan: The data retrieval plan from the Supervisor

    Returns:
        Configured Task instance
    """
    object_data = activity_data.get("object_data", {})
    strava_id = str(object_data.get("id", "unknown"))
    activity_type = object_data.get("type", "Run")
    activity_date = retrieval_plan.get("activity_date", "")

    # Format the plan for the task description
    plan_json = json.dumps(retrieval_plan, indent=2)

    description = f"""Execute the activity data retrieval plan and return raw data.

**YOUR ROLE**: You are the EXECUTOR. You EXECUTE tool calls and return raw data.
Do NOT analyze or interpret the data - the Reviewer agent will do that.

DATA RETRIEVAL PLAN FROM SUPERVISOR:
{plan_json}

YOUR TASK:
1. Execute the tool calls specified in the plan above
2. For the FIRST call: IntervalsIcu__get_activities with start_date="{activity_date}", end_date="{activity_date}", limit=10
3. Extract the Intervals.icu activity ID from the first result
4. For SECOND call: IntervalsIcu__get_activity_details with the activity_id from step 3
5. For THIRD call: IntervalsIcu__get_activity_streams with the activity_id and stream_types="core_temperature,skin_temperature"
6. Collect all results and return raw data

EXECUTION INSTRUCTIONS:
- Call tools ONE AT A TIME
- Use the EXACT parameters from the plan
- Extract the activity ID from first result to use in subsequent calls
- If a tool call fails, record the error and continue
- Return ALL data received from the tools

CRITICAL - TOOL INPUT FORMAT:
- Tool inputs must be a DICTIONARY, never a list
- Example: {{"start_date": "{activity_date}", "end_date": "{activity_date}", "limit": 10}}

OUTPUT REQUIREMENTS - RawActivityData JSON:

After executing the tool calls, return a JSON object:
```json
{{
  "activity_id": "{strava_id}",
  "activity_date": "{activity_date}",
  "activity_type": "{activity_type}",
  "tool_results": [
    {{
      "tool_name": "IntervalsIcu__get_activities",
      "success": true,
      "result": {{ ... raw tool result ... }},
      "error_message": null
    }},
    {{
      "tool_name": "IntervalsIcu__get_activity_details",
      "success": true,
      "result": {{ ... raw tool result ... }},
      "error_message": null
    }},
    {{
      "tool_name": "IntervalsIcu__get_activity_streams",
      "success": true,
      "result": {{ ... raw tool result ... }},
      "error_message": null
    }}
  ],
  "total_calls": 3,
  "successful_calls": 3,
  "intervals_activity_id": "i107537962",
  "activity_details": {{ ... extracted details ... }},
  "activity_streams": {{ ... extracted streams ... }},
  "retrieval_notes": "Successfully retrieved all planned data"
}}
```

CRITICAL:
- Execute the tool calls FIRST, then format the output
- Include the COMPLETE raw data from tools in the result
- Extract key fields into activity_details and activity_streams for easier processing

Return ONLY the JSON object after tool execution, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with raw activity data.
Format: {{"activity_id": "{strava_id}", "activity_date": "{activity_date}", "activity_type": "{activity_type}", "tool_results": [...], "intervals_activity_id": "...", "activity_details": {{...}}, "activity_streams": {{...}}, ...}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
