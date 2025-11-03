"""Task for creating the weekly nutrition structure."""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task

from schemas import WeeklyNutritionPlan


def create_weekly_structure_task(agent: Any, hexis_analysis: Dict[str, Any]) -> Task:
    """
    Create a task for structuring the weekly nutrition plan.

    Args:
        agent: The agent responsible for this task
        hexis_analysis: The Hexis analysis output (HexisWeeklyAnalysis as dict)

    Returns:
        Configured Task instance
    """
    hexis_json = json.dumps(hexis_analysis, indent=2)

    description = f"""
Transform Hexis analysis into structured weekly nutrition plan with daily targets and meal timing.

HEXIS INPUT:
{hexis_json}

TASKS:
1. For each day (Mon-Sun): extract date, calories, macros from Hexis; add training context (workout type/intensity) and meal timing guidance
2. Meal timing: consider workout schedule (morning/afternoon/evening), pre/post workout nutrition, macro distribution
3. Training context examples: "Hard intervals 90min", "Easy run 45min", "Rest day", "Long run 2h tempo"
4. Weekly summary: training focus, nutritional themes (carb cycling/recovery), special considerations, expected outcomes

Return JSON: week_start_date, week_end_date, daily_targets (array with day_name/date/calories/macros/training_context/meal_timing_notes for all 7 days), weekly_summary.
No markdown fences. Be specific and actionable.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the WeeklyNutritionPlan schema with complete daily targets",
        output_json=WeeklyNutritionPlan,
    )
