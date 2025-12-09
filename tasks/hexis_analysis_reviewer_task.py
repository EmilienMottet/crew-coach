"""Task for the Hexis Analysis Reviewer to synthesize data into structured output.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

from crewai import Task


def _extract_daily_meal_targets(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically extract per-meal targets from raw Hexis data.

    This extracts daily_meal_targets DIRECTLY from the raw Hexis API response,
    without relying on LLM synthesis. This is more reliable than asking the LLM
    to generate this data, which can truncate after 4/7 days.

    Args:
        raw_data: The raw Hexis data from the Executor

    Returns:
        Dict mapping date strings to meal targets:
        {
            "2025-12-08": {
                "date": "2025-12-08",
                "meals": [
                    {"meal_type": "Breakfast", "time": "...", "carb_code": "...", ...},
                    ...
                ]
            },
            ...
        }
    """
    daily_meal_targets: Dict[str, Any] = {}

    weekly_plan = raw_data.get("weekly_plan_data") or {}
    # Hexis API returns data nested under "data" key
    days = weekly_plan.get("data", {}).get("days", []) if weekly_plan else []

    # FALLBACK: If weekly_plan_data is empty, aggregate from tool_results
    # This handles batched execution where data comes from multiple tool calls
    if not days:
        tool_results = raw_data.get("tool_results", [])
        all_batch_days = []
        batch_count = 0
        for tool_result in tool_results:
            tool_name = tool_result.get("tool_name", "")
            if tool_name == "hexis__hexis_get_weekly_plan":
                result = tool_result.get("result", {})
                batch_days = result.get("data", {}).get("days", [])
                if batch_days:
                    all_batch_days.extend(batch_days)
                    batch_count += 1
        if all_batch_days:
            # Deduplicate by dayString (in case of overlap between batches)
            seen_days: Dict[str, Any] = {}
            for day in all_batch_days:
                day_str = day.get("dayString")
                if day_str and day_str not in seen_days:
                    seen_days[day_str] = day
            days = list(seen_days.values())
            print(
                f"   📦 Aggregated {len(days)} days from {batch_count} batch(es)",
                file=sys.stderr,
            )

    # Map Hexis meal names to our standard format
    meal_name_map = {
        "Breakfast": "Breakfast",
        "Lunch": "Lunch",
        "PM Snack": "Afternoon Snack",
        "Dinner": "Dinner",
    }

    for day in days:
        day_string = day.get("dayString")
        if not day_string:
            continue

        meals_data: List[Dict[str, Any]] = []

        for meal in day.get("meals", []):
            meal_type = meal.get("mealType", "")
            # Only process MAIN meals and SNACK (skip INTRA_FUELLING)
            if meal_type not in ["MAIN", "SNACK"]:
                continue

            meal_name = meal.get("mealName", "")
            # Map to standard meal type name
            standardized_name = meal_name_map.get(meal_name, meal_name)

            meal_rec = meal.get("mealRecommendation", {})
            macros = meal_rec.get("macros", {})

            # Get carb code from meal or mealRecommendation
            carb_code = meal.get("carbCode") or meal_rec.get("carbCode") or "MEDIUM"

            meal_entry = {
                "meal_type": standardized_name,
                "time": meal.get("time", ""),
                "carb_code": carb_code,
                "calories": int(macros.get("energy", 0)),
                "protein_g": int(macros.get("protein", 0)),
                "carbs_g": int(macros.get("carb", 0)),
                "fat_g": int(macros.get("fat", 0)),
            }
            meals_data.append(meal_entry)

        # Sort meals by time for consistent ordering
        meals_data.sort(key=lambda m: m.get("time", ""))

        daily_meal_targets[day_string] = {
            "date": day_string,
            "meals": meals_data,
        }

    return daily_meal_targets


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

    weekly_plan = raw_data.get("weekly_plan_data") or {}
    # Hexis API returns data nested under "data" key
    days = weekly_plan.get("data", {}).get("days", []) if weekly_plan else []

    # FALLBACK: If weekly_plan_data is empty, aggregate from tool_results
    # This handles batched execution where data comes from multiple tool calls
    if not days:
        tool_results = raw_data.get("tool_results", [])
        all_batch_days = []
        batch_count = 0
        for tool_result in tool_results:
            tool_name = tool_result.get("tool_name", "")
            if tool_name == "hexis__hexis_get_weekly_plan":
                result = tool_result.get("result", {})
                batch_days = result.get("data", {}).get("days", [])
                if batch_days:
                    all_batch_days.extend(batch_days)
                    batch_count += 1
        if all_batch_days:
            # Deduplicate by dayString (in case of overlap between batches)
            seen_days: Dict[str, Any] = {}
            for day in all_batch_days:
                day_str = day.get("dayString")
                if day_str and day_str not in seen_days:
                    seen_days[day_str] = day
            days = list(seen_days.values())
            print(
                f"   📦 Compact extraction: {len(days)} days from {batch_count} batch(es)",
                file=sys.stderr,
            )

    for day in days:
        # Hexis API returns macros at day level in "macros" field, not "dailyTargets"
        day_macros = day.get("macros", {})
        day_compact: Dict[str, Any] = {
            "dayString": day.get("dayString"),
            "meals": [],
            "dailyTargets": {
                "energy": day_macros.get("energy", 0),
                "protein": day_macros.get("protein", 0),
                "carb": day_macros.get("carb", 0),
                "fat": day_macros.get("fat", 0),
                "carbsPerKg": day.get("carbsPerKg", 0),
            },
            "totalDailyEnergyExpenditure": day.get("totalDailyEnergyExpenditure", 0),
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

**YOUR ROLE**: You are the REVIEWER. You ANALYZE training data and create structured output.
You do NOT make any tool calls.

**NOTE**: daily_meal_targets will be extracted automatically in Python code - you do NOT need to generate it.
Focus on training analysis, energy needs, macro targets, and nutritional priorities.

COMPACT HEXIS DATA:
{raw_data_str}

**DATA STRUCTURE**:
- `days[]`: Array of {days_count} days in the week
- `days[].dayString`: Date (YYYY-MM-DD)
- `days[].dailyTargets`: Daily macro totals from Hexis
- `training_summary[]`: Workout info for training context

YOUR TASK:
1. Calculate training load metrics from training_summary
2. Assess recovery status based on training load
3. Extract daily energy needs from each day's dailyTargets
4. Extract daily macro targets from each day's dailyTargets
5. Identify 4-6 key nutritional priorities
6. Create the HexisWeeklyAnalysis JSON

OUTPUT - HexisWeeklyAnalysis JSON:

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
      {{"date": "YYYY-MM-DD", "title": "...", "activity": "...", "duration_minutes": 0, "calories": 0, "intensity_rpe": 0, "description": "..."}}
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
    "YYYY-MM-DD": {{"bmr": 0, "exercise_calories": 0, "neat_calories": 0, "tdee": 0, "hexis_target": 0, "energy_balance": 0, "workout_energy_expenditure": 0}}
  }},
  "daily_macro_targets": {{
    "YYYY-MM-DD": {{"protein_g": 0, "carbs_g": 0, "fat_g": 0, "calories": 0, "carbs_per_kg": 0.0, "protein_per_kg": 0.0, "fat_per_kg": 0.0, "hexis_source": "HEXIS"}}
  }},
  "nutritional_priorities": ["...", "...", "..."]
}}
```

GUIDELINES:
- Use ACTUAL data from the Hexis data to populate all fields
- Extract macro targets from each day's dailyTargets (set hexis_source: "HEXIS")
- Calculate training metrics from training_summary array
- daily_energy_needs and daily_macro_targets MUST have an entry for EACH of the {days_count} days
- Identify 4-6 key nutritional priorities based on training schedule

Return ONLY the JSON object, no explanations."""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"""Complete JSON object with structured Hexis analysis for ALL {days_count} DAYS.
Format: {{"week_start_date": "{week_start_date}", "week_end_date": "{week_end_date}", "training_load_summary": {{}}, "recovery_status": {{}}, "daily_energy_needs": {{}}, "daily_macro_targets": {{}}, "nutritional_priorities": [...]}}
CRITICAL: daily_energy_needs and daily_macro_targets MUST contain exactly {days_count} entries, one for each day.
NO markdown fences, NO explanatory text, ONLY the JSON object.""",
    )
