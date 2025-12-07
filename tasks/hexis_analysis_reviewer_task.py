"""Task for the Hexis Analysis Reviewer to synthesize data into structured output.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

from crewai import Task


def _extract_meal_relevant_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only meal-relevant data from raw Hexis response.

    This creates a compact representation that preserves all 7 days
    while removing verbose fields not needed for meal target extraction.

    Args:
        raw_data: The full raw Hexis data from the Executor

    Returns:
        A compact dict with only meal-relevant fields for all days
    """
    compact: Dict[str, Any] = {
        "week_start_date": raw_data.get("week_start_date"),
        "week_end_date": raw_data.get("week_end_date"),
        "days": [],
        "training_summary": [],  # Keep minimal training info
    }

    weekly_plan = raw_data.get("weekly_plan_data", {})
    # Hexis API returns data nested under "data" key
    days = weekly_plan.get("data", {}).get("days", [])

    for day in days:
        day_compact: Dict[str, Any] = {
            "dayString": day.get("dayString"),
            "meals": [],
            "dailyTargets": day.get("dailyTargets", {}),  # Keep daily macro totals
        }

        # Extract workouts for training context (compact version)
        workouts = day.get("workouts", [])
        if workouts:
            for w in workouts:
                compact["training_summary"].append({
                    "date": day.get("dayString"),
                    "title": w.get("title", ""),
                    "activity": w.get("activity", ""),
                    "duration_minutes": w.get("duration", 0),
                    "tss": w.get("tss", 0),
                })

        # Extract meals (only MAIN and SNACK, skip INTRA_FUELLING)
        for meal in day.get("meals", []):
            meal_type = meal.get("mealType", "")
            if meal_type in ["MAIN", "SNACK"]:
                meal_rec = meal.get("mealRecommendation", {})
                macros = meal_rec.get("macros", {})
                meal_compact = {
                    "mealName": meal.get("mealName"),
                    "mealType": meal_type,
                    "time": meal.get("time"),
                    "carbCode": meal.get("carbCode") or meal_rec.get("carbCode"),
                    "macros": {
                        "energy": macros.get("energy", 0),
                        "protein": macros.get("protein", 0),
                        "carb": macros.get("carb", 0),
                        "fat": macros.get("fat", 0),
                    }
                }
                day_compact["meals"].append(meal_compact)

        compact["days"].append(day_compact)

    return compact


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

    # Extract only meal-relevant data to avoid truncation issues
    # This preserves ALL 7 days while removing verbose fields
    compact_data = _extract_meal_relevant_data(raw_hexis_data)
    raw_data_str = json.dumps(compact_data, indent=2)

    # Log the size reduction for debugging
    original_size = len(json.dumps(raw_hexis_data))
    compact_size = len(raw_data_str)
    days_count = len(compact_data.get("days", []))
    print(
        f"   Hexis data compacted: {original_size:,} -> {compact_size:,} chars "
        f"({100 * compact_size // max(1, original_size)}%), {days_count} days preserved",
        file=sys.stderr,
    )

    description = f"""Analyze the COMPACT Hexis data and create a structured weekly analysis.

**YOUR ROLE**: You are the REVIEWER. You ANALYZE the pre-processed data and create structured output.
You do NOT make any tool calls.

COMPACT HEXIS DATA (pre-processed to include only meal-relevant fields):
{raw_data_str}

**DATA STRUCTURE EXPLANATION**:
- `days[]`: Array of ALL {days_count} days in the week (MUST process all of them!)
- `days[].dayString`: The date (YYYY-MM-DD format)
- `days[].meals[]`: Array of MAIN and SNACK meals for this day
- `days[].meals[].mealName`: "Breakfast", "Lunch", "PM Snack", "Dinner"
- `days[].meals[].carbCode`: "LOW", "MEDIUM", or "HIGH"
- `days[].meals[].macros`: {{energy, protein, carb, fat}} values from Hexis
- `days[].dailyTargets`: Daily macro totals from Hexis
- `training_summary[]`: Compact workout info for training context

YOUR TASK:
1. **CRITICAL: Process ALL {days_count} DAYS** from the days[] array
2. Calculate training load metrics from training_summary
3. Assess recovery status based on training load
4. Extract daily energy needs and macro targets from each day's dailyTargets
5. **CRITICAL**: Extract per-meal targets from each day's meals[] array
6. Identify 4-6 key nutritional priorities
7. Create the final HexisWeeklyAnalysis JSON with data for ALL {days_count} days

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
- Use ACTUAL data from the compact Hexis data to populate all fields
- Extract macro targets from each day's dailyTargets (hexis_source: "HEXIS")
- Calculate training metrics from training_summary array
- Identify 4-6 key nutritional priorities based on the training schedule
- Be specific and actionable in recommendations

**CRITICAL - DAILY_MEAL_TARGETS EXTRACTION**:
- **ITERATE OVER ALL {days_count} DAYS** in the days[] array
- For each day, use dayString as the date key (e.g., "2025-12-08")
- Extract meals from the meals[] array for that day
- Map mealName to meal_type: "Breakfast", "Lunch", "PM Snack" → "Afternoon Snack", "Dinner"
- Map macros fields: energy → calories, protein → protein_g, carb → carbs_g, fat → fat_g
- Use carbCode directly (must be "LOW", "MEDIUM", or "HIGH")
- Include the meal time field as-is
- ALL four meal types (Breakfast, Lunch, Afternoon Snack, Dinner) MUST be present for EACH of the {days_count} days

**VALIDATION**: Your output MUST contain exactly {days_count} entries in daily_meal_targets, one for each day.

Return ONLY the JSON object, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with structured Hexis analysis for ALL {days_count} DAYS.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "training_load_summary": {{}}, "recovery_status": {{}}, "daily_energy_needs": {{}}, "daily_macro_targets": {{}}, "daily_meal_targets": {{}}, "nutritional_priorities": [...]}}
CRITICAL: daily_meal_targets MUST contain exactly {days_count} entries with per-meal targets (Breakfast, Lunch, Afternoon Snack, Dinner) and carb_code for EACH day.
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
