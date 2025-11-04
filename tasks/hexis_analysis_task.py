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

STEPS:
1. Fetch Hexis data using `hexis_get_weekly_plan` (or related tools):
   - Training schedule: workouts (type/duration/TSS)
   - Recovery metrics: HRV, sleep quality, readiness scores
   - Load trends: CTL, ATL, TSB, training stress
   - **Nutritional data**: daily macro targets (protein/carbs/fat), calorie recommendations, meal plans
   
2. Analyze each day considering:
   - Workout intensity and energy expenditure
   - Expected glycogen depletion from training
   - Recovery needs based on HRV and sleep data
   - **Hexis-provided nutritional targets** (if available, use these as primary source)
   
3. Calculate daily nutrition requirements (for 70kg athlete, BMR 1800 kcal):
   - TDEE (BMR + exercise calories)
   - Carbs: 3-5 g/kg (rest), 5-7 (low), 6-8 (moderate), 8-12 (high intensity)
   - Protein: 1.6-2.2 g/kg
   - Fat: remainder (min 0.8 g/kg)
   - **PRIORITY**: Use Hexis nutritional targets when available, fall back to calculations if needed
   
4. Identify nutritional priorities:
   - Pre/post workout fueling strategies
   - Training adaptation goals (e.g., low-carb training, carb-loading)
   - Special considerations (race prep, recovery weeks)

CRITICAL: The `hexis_get_weekly_plan` tool returns BOTH workout AND nutrition data.
Make sure to extract and include nutritional targets from Hexis in your analysis.

Return JSON: week_start_date, week_end_date, training_load_summary, recovery_status, 
daily_energy_needs (dict with days), daily_macro_targets (dict with protein_g/carbs_g/fat_g/calories per day), 
nutritional_priorities (list).
Cover all 7 days, no markdown fences.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the HexisWeeklyAnalysis schema with complete weekly analysis",
        output_json=HexisWeeklyAnalysis,
    )
