"""Task for analyzing Hexis training data for nutritional planning."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

from crewai import Task
from crewai.tasks.task_output import TaskOutput

# HexisWeeklyAnalysis schema left imported elsewhere when structured outputs are re-enabled


def validate_hexis_analysis_output(task_output):
    """Ensure the Hexis analysis task returns a JSON object with required fields."""
    raw_output = task_output.raw.strip()

    try:
        payload: Dict[str, Any] = json.loads(raw_output)
    except json.JSONDecodeError:
        return False, (
            "Your final answer must be a JSON object with the required fields. "
            "Return only JSON and ensure it's valid (no thoughts or markdown)."
        )

    required_fields = {
        "week_start_date",
        "week_end_date",
        "training_load_summary",
        "recovery_status",
        "daily_energy_needs",
        "daily_macro_targets",
        "nutritional_priorities",
    }
    missing = [field for field in required_fields if field not in payload]
    if missing:
        return False, (
            "The JSON output is missing required keys: " + ", ".join(sorted(missing))
        )

    return True, task_output.raw


def create_hexis_analysis_task(agent: Any, week_start_date: str) -> Task:
    """
    Create a task for analyzing Hexis training data.

    Args:
        agent: The agent responsible for this task
        week_start_date: Start date of the week to plan (ISO format YYYY-MM-DD)

    Returns:
        Configured Task instance
    """
    # Calculate week end date (6 days after start)
    start_dt = datetime.fromisoformat(week_start_date)
    end_dt = start_dt + timedelta(days=6)
    week_end_date = end_dt.strftime("%Y-%m-%d")

    description = f"""Analyze Hexis data for week {week_start_date} to {week_end_date}.

Your task is to retrieve training and nutrition data, then create a structured analysis.

FIRST: Call hexis_get_weekly_plan with start_date="{week_start_date}", end_date="{week_end_date}"

THEN: Based on the tool results, create this JSON analysis:

{{
  "week_start_date": "{week_start_date}",
  "week_end_date": "{week_end_date}",
    "training_load_summary": {{
    "total_weekly_tss": 0,
    "total_training_time_minutes": 0,
    "training_days": 0,
    "rest_days": 0,
    "key_sessions": [],
    "weekly_load_classification": "...",
    "ctl_trend": "...",
    "atl_trend": "...",
    "tsb_trend": "..."
  }},
  "recovery_status": {{
    "overall_assessment": "...",
    "key_observations": ["..."],
    "recovery_recommendations": ["..."],
    "readiness_indicators": ["..."]
  }},
  "daily_energy_needs": {{
    "YYYY-MM-DD": {{
      "bmr": 0,
      "exercise_calories": 0,
      "neat_calories": 0,
      "tdee": 0,
      "hexis_target": 0,
      "energy_balance": 0,
      "workout_energy_expenditure": 0
    }}
  }},
  "daily_macro_targets": {{
    "YYYY-MM-DD": {{
      "protein_g": 0,
      "carbs_g": 0,
      "fat_g": 0,
      "calories": 0,
      "carbs_per_kg": 0.0,
      "protein_per_kg": 0.0,
      "fat_per_kg": 0.0,
      "hexis_source": "HEXIS"
    }}
  }},
  "nutritional_priorities": ["...", "...", "..."]
}}

Fill in actual data from Hexis where available. Use the tool results to populate real macro targets and energy needs for each day.

Return ONLY the JSON object, no explanations.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="""Complete JSON object with all required fields populated from Hexis data analysis.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "training_load_summary": {{}}, ...}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""".format(
            week_start_date=week_start_date, week_end_date=week_end_date
        ),
        # DISABLED: guardrail causes infinite loop when validation fails and agent retries tool call
        # guardrail=validate_hexis_analysis_output,
        # DISABLED: output_json causes auth issues with instructor
        # output_json=HexisWeeklyAnalysis,
    )
