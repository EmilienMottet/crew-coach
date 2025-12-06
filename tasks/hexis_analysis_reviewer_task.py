"""Task for the Hexis Analysis Reviewer to synthesize data into structured output.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict

from crewai import Task


def create_hexis_analysis_reviewer_task(
    agent: Any,
    week_start_date: str,
    raw_hexis_data: Dict[str, Any],
) -> Task:
    """
    Create a task for the Reviewer to analyze raw Hexis data.

    Args:
        agent: The Reviewer agent
        week_start_date: Start date of the week (ISO format YYYY-MM-DD)
        raw_hexis_data: The raw data from the Executor

    Returns:
        Configured Task instance
    """
    # Calculate week end date
    start_dt = datetime.fromisoformat(week_start_date)
    end_dt = start_dt + timedelta(days=6)
    week_end_date = end_dt.strftime("%Y-%m-%d")

    # Format the raw data for the task description
    # Truncate if too large to avoid context overflow
    raw_data_str = json.dumps(raw_hexis_data, indent=2)
    if len(raw_data_str) > 15000:
        raw_data_str = raw_data_str[:15000] + "\n... [truncated for brevity]"

    description = f"""Analyze the raw Hexis data and create a structured weekly analysis.

**YOUR ROLE**: You are the REVIEWER. You ANALYZE the raw data and create structured output.
You do NOT make any tool calls.

RAW HEXIS DATA FROM EXECUTOR:
{raw_data_str}

YOUR TASK:
1. Parse the raw data to extract training and nutrition information
2. Calculate training load metrics and trends
3. Assess recovery status
4. Extract daily energy needs and macro targets from Hexis data
5. **CRITICAL**: Extract per-meal targets from Hexis mealRecommendation data
6. Identify key nutritional priorities
7. Create the final HexisWeeklyAnalysis JSON

OUTPUT REQUIREMENTS - HexisWeeklyAnalysis JSON:

Return a JSON object with this exact structure:
```json
{{
  "week_start_date": "{week_start_date}",
  "week_end_date": "{week_end_date}",
  "training_load_summary": {{
    "total_weekly_tss": 0,
    "total_training_time_minutes": 0,
    "training_days": 0,
    "rest_days": 0,
    "key_sessions": [
      {{
        "date": "YYYY-MM-DD",
        "title": "...",
        "activity": "...",
        "duration_minutes": 0,
        "calories": 0,
        "intensity_rpe": 0,
        "description": "..."
      }}
    ],
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
  "daily_meal_targets": {{
    "YYYY-MM-DD": {{
      "date": "YYYY-MM-DD",
      "meals": [
        {{
          "meal_type": "Breakfast",
          "time": "08:00:00.000Z",
          "carb_code": "MEDIUM",
          "calories": 674,
          "protein_g": 51,
          "carbs_g": 41,
          "fat_g": 34
        }},
        {{
          "meal_type": "Lunch",
          "time": "13:30:00.000Z",
          "carb_code": "HIGH",
          "calories": 915,
          "protein_g": 51,
          "carbs_g": 117,
          "fat_g": 27
        }},
        {{
          "meal_type": "PM Snack",
          "time": "18:30:00.000Z",
          "carb_code": "MEDIUM",
          "calories": 369,
          "protein_g": 27,
          "carbs_g": 27,
          "fat_g": 17
        }},
        {{
          "meal_type": "Dinner",
          "time": "20:30:00.000Z",
          "carb_code": "MEDIUM",
          "calories": 867,
          "protein_g": 51,
          "carbs_g": 96,
          "fat_g": 31
        }}
      ]
    }}
  }},
  "nutritional_priorities": ["...", "...", "..."]
}}
```

GUIDELINES:
- Use ACTUAL data from the raw Hexis response to populate fields
- Extract macro targets directly from Hexis data (hexis_source: "HEXIS")
- Calculate training metrics from workout data
- Identify 4-6 key nutritional priorities based on the training schedule
- Be specific and actionable in recommendations

**CRITICAL - DAILY_MEAL_TARGETS EXTRACTION**:
- For each day, extract meal data from the "meals" array in the raw Hexis response
- ONLY include meals with mealType "MAIN" or "SNACK" (skip "INTRA_FUELLING")
- Map mealName to meal_type: "Breakfast", "Lunch", "PM Snack" → "Afternoon Snack", "Dinner"
- Extract from mealRecommendation.macros: energy → calories, protein → protein_g, carb → carbs_g, fat → fat_g
- Extract carbCode directly from the meal or mealRecommendation
- The carb_code field MUST be one of: "LOW", "MEDIUM", "HIGH"
- Include the meal time field as-is
- ALL four meal types (Breakfast, Lunch, Afternoon Snack, Dinner) MUST be present for each day

Return ONLY the JSON object, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with structured Hexis analysis.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "training_load_summary": {{}}, "recovery_status": {{}}, "daily_energy_needs": {{}}, "daily_macro_targets": {{}}, "daily_meal_targets": {{}}, "nutritional_priorities": [...]}}
CRITICAL: daily_meal_targets MUST contain per-meal targets (Breakfast, Lunch, Afternoon Snack, Dinner) with carb_code for each day.
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
