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
    Analyze Hexis training data to determine nutritional requirements for the upcoming week.

    TARGET WEEK:
    - Start Date: {week_start_date}
    - End Date: {week_end_date}

    YOUR MISSION:

    1. USE HEXIS MCP TOOLS TO RETRIEVE DATA
        - Fetch training schedule for the week ({week_start_date} to {week_end_date})
        - Retrieve workout details (type, duration, intensity, TSS)
        - Get recovery metrics (HRV, sleep quality, readiness scores)
        - Check training load trends (CTL, ATL, TSB)

    2. ANALYZE TRAINING DEMANDS
        For each day of the week:
        - Identify workout type and intensity
        - Estimate energy expenditure (calories burned)
        - Determine glycogen depletion level
        - Assess recovery requirements

    3. CALCULATE NUTRITIONAL NEEDS
        For each day, determine:
        - Total daily energy expenditure (TDEE)
        - Carbohydrate needs (g/kg body weight adjusted for training)
          * Rest days: 3-5 g/kg
          * Low-intensity days: 5-7 g/kg
          * Moderate training: 6-8 g/kg
          * High-intensity/long workouts: 8-12 g/kg
        - Protein needs (1.6-2.2 g/kg for athletes)
        - Fat needs (remainder of calories, minimum 0.8 g/kg)

    4. IDENTIFY NUTRITIONAL PRIORITIES
        Based on training phase and demands:
        - Pre-workout fueling needs
        - Post-workout recovery priorities
        - Adaptation goals (e.g., "train low, compete high")
        - Special considerations (race week, taper, overload)

    5. OUTPUT STRUCTURED ANALYSIS
        Return valid JSON matching HexisWeeklyAnalysis schema:
        - Week date range
        - Training load summary
        - Recovery status
        - Daily energy needs (calories per day)
        - Daily macro targets (protein/carbs/fat in grams per day)
        - Nutritional priorities list

    IMPORTANT ASSUMPTIONS:
    - Assume athlete weighs 70kg (adjust if data available from Hexis)
    - Base metabolic rate ~1800 kcal/day (sedentary)
    - Add exercise calories on top of BMR
    - Maintain energy balance unless specific goals (cut/bulk)

    EXAMPLE OUTPUT STRUCTURE:
    {{
      "week_start_date": "{week_start_date}",
      "week_end_date": "{week_end_date}",
      "training_load_summary": "Moderate week with 3 quality sessions...",
      "recovery_status": "Good recovery, ready for training...",
      "daily_energy_needs": {{
        "Monday": 2800,
        "Tuesday": 2400,
        ...
      }},
      "daily_macro_targets": {{
        "Monday": {{"protein_g": 140, "carbs_g": 350, "fat_g": 78, "calories": 2800}},
        ...
      }},
      "nutritional_priorities": [
        "High-carb breakfast before Wednesday intervals",
        "Recovery focus on Thursday",
        "Carb-loading for Sunday long run"
      ]
    }}

    OUTPUT CONTRACT:
    - Respond with valid JSON matching the HexisWeeklyAnalysis schema
    - Do not wrap JSON in markdown fences
    - Ensure all 7 days of the week are covered
    - Be specific and actionable in nutritional priorities
    """

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the HexisWeeklyAnalysis schema with complete weekly analysis",
        output_json=HexisWeeklyAnalysis,
    )
