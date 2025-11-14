"""Task for analyzing Hexis training data for nutritional planning."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from crewai import Task

from schemas import HexisWeeklyAnalysis


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

    description = f"""
Analyze Hexis training data for week {week_start_date} to {week_end_date}.

WORKFLOW:
1. Call `hexis_get_weekly_plan` with start_date="{week_start_date}", end_date="{week_end_date}"
   to retrieve training and nutrition data

2. AFTER receiving the tool results, analyze the data and construct a complete JSON response

ANALYZE THE FOLLOWING from the tool results:
- Training schedule: workouts (type/duration/TSS)
- Recovery metrics: HRV, sleep quality, readiness scores
- Load trends: CTL, ATL, TSB, training stress
- **Nutritional data**: daily macro targets (protein/carbs/fat), calorie recommendations, meal plans

CONSTRUCT JSON OUTPUT with these fields (cover all 7 days):
- week_start_date: "{week_start_date}"
- week_end_date: "{week_end_date}"
- training_load_summary: {{total_weekly_tss, total_training_time_minutes, training_days, rest_days, key_sessions[], weekly_load_classification, ctl_trend, atl_trend, tsb_trend}}
- recovery_status: {{overall_assessment, key_observations[], recovery_recommendations[], readiness_indicators[]}}
- daily_energy_needs: {{"YYYY-MM-DD": {{bmr, exercise_calories, neat_calories, tdee, hexis_target, energy_balance, workout_energy_expenditure}}, ...}}
- daily_macro_targets: {{"YYYY-MM-DD": {{protein_g, carbs_g, fat_g, calories, carbs_per_kg, protein_per_kg, fat_per_kg, hexis_source}}, ...}}
- nutritional_priorities: [priority descriptions based on training load and goals]

CRITICAL INSTRUCTIONS:
1. You MUST call the tool first to get real data
2. After receiving tool results, you MUST analyze and generate the complete JSON
3. Return ONLY the final JSON object - no "Thought:", no "Action:", no markdown fences
4. Use actual data from Hexis where available (macros, meals, energy targets)
5. Calculate estimated values only where Hexis data is missing

DO NOT return the action format - return the analysis JSON directly.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="""Complete JSON object with all required fields populated from Hexis data analysis.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "training_load_summary": {{}}, ...}}
NO markdown fences, NO explanatory text, ONLY the JSON object.""".format(week_start_date=week_start_date, week_end_date=week_end_date),
        # DISABLED: output_json causes auth issues with instructor
        # output_json=HexisWeeklyAnalysis,
    )
