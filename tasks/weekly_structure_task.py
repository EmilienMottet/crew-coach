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
    Transform the Hexis training analysis into a structured weekly nutrition plan
    with detailed daily targets and meal timing guidance.

    HEXIS ANALYSIS INPUT:
    {hexis_json}

    YOUR MISSION:

    1. CREATE DAILY NUTRITION TARGETS
        For each day of the week (Monday through Sunday):
        - Extract the date from the Hexis analysis
        - Use the calorie target from Hexis
        - Use the macro targets from Hexis (protein, carbs, fat)
        - Interpret the training context (workout type, intensity)
        - Determine optimal meal timing based on workout schedule

    2. PROVIDE MEAL TIMING GUIDANCE
        For each day, consider:
        - When is the workout scheduled? (morning/afternoon/evening)
        - What meals should be eaten before the workout?
        - What timing is optimal for recovery nutrition?
        - How should macros be distributed across meals?

        Example meal timing notes:
        - "High-carb breakfast 2-3h before morning intervals"
        - "Moderate lunch, pre-workout snack, post-workout recovery dinner"
        - "Balanced distribution across 3 main meals + 2 snacks"
        - "Front-load carbs: 40% breakfast, 30% lunch, 30% dinner"

    3. CREATE TRAINING CONTEXT DESCRIPTIONS
        For each day, provide clear training context:
        - "Hard interval session (90 min, high intensity)"
        - "Easy recovery run (45 min, low intensity)"
        - "Rest day - active recovery"
        - "Long endurance run (2h, moderate intensity)"
        - "Tempo workout (60 min, threshold effort)"

    4. WRITE WEEKLY SUMMARY
        Synthesize the week's nutrition strategy:
        - Overall training focus (base building, peak training, taper, etc.)
        - Key nutritional themes (carb cycling, recovery focus, etc.)
        - Special considerations (race week, travel, etc.)
        - Expected outcomes (maintain weight, fuel performance, optimize recovery)

        Example summary:
        "Moderate training week with 3 quality sessions requiring strategic carbohydrate
        periodization. Emphasis on recovery nutrition post-intervals and long run.
        Maintaining energy balance with slight carb cycling to match training demands."

    5. OUTPUT STRUCTURED PLAN
        Return valid JSON matching WeeklyNutritionPlan schema:
        - Week date range (from Hexis analysis)
        - Daily targets array (7 days)
        - Weekly summary

    IMPORTANT GUIDELINES:
    - Be specific and actionable in meal timing notes
    - Consider realistic meal schedules (breakfast ~7am, lunch ~12pm, dinner ~7pm)
    - Account for workout timing in meal distribution
    - Ensure daily targets align with Hexis recommendations
    - Make training context clear and descriptive

    EXAMPLE OUTPUT STRUCTURE:
    {{
      "week_start_date": "2025-01-06",
      "week_end_date": "2025-01-12",
      "daily_targets": [
        {{
          "day_name": "Monday",
          "date": "2025-01-06",
          "calories": 2800,
          "macros": {{"protein_g": 140, "carbs_g": 350, "fat_g": 78, "calories": 2800}},
          "training_context": "Hard interval session (90 min, VO2max intervals)",
          "meal_timing_notes": "High-carb breakfast 2-3h pre-workout. Recovery-focused post-workout meal within 30min. Balanced dinner."
        }},
        ...
      ],
      "weekly_summary": "Moderate training week with strategic carb periodization..."
    }}

    OUTPUT CONTRACT:
    - Respond with valid JSON matching the WeeklyNutritionPlan schema
    - Do not wrap JSON in markdown fences
    - Include all 7 days with complete information
    - Be specific and practical in meal timing guidance
    """

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the WeeklyNutritionPlan schema with complete daily targets",
        output_json=WeeklyNutritionPlan,
    )
