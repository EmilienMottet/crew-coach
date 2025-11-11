"""Task for integrating meal plans into Mealy."""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task

from schemas import MealyIntegrationResult


def create_mealy_integration_task(
    agent: Any,
    weekly_meal_plan: Dict[str, Any],
    validation_result: Dict[str, Any],
) -> Task:
    """
    Create a task for integrating the meal plan into Mealy.

    Args:
        agent: The agent responsible for this task
        weekly_meal_plan: The validated meal plan (WeeklyMealPlan as dict)
        validation_result: The validation assessment (NutritionalValidation as dict)

    Returns:
        Configured Task instance
    """
    meal_plan_json = json.dumps(weekly_meal_plan, indent=2)
    validation_json = json.dumps(validation_result, indent=2)

    description = f"""
    Integrate the validated weekly meal plan into Mealy with comprehensive
    error handling and status reporting.

    VALIDATION RESULT:
    {validation_json}

    MEAL PLAN TO INTEGRATE:
    {meal_plan_json}

    YOUR MISSION:

    1. VALIDATE INPUT DATA
        - Check that validation_result.approved == True
        - If not approved, do NOT proceed with integration
        - Report that integration was skipped due to validation failure
        - If approved, proceed to integration

    2. PREPARE FOR INTEGRATION
        - Extract week_start_date and week_end_date
        - Count total meals to sync (typically 21-28 meals: 3 meals + snacks × 7 days)
        - Initialize tracking for sync results

    3. USE MEALY MCP TOOLS TO SYNC MEALS
        For each day in daily_plans (Monday through Sunday):
          For each meal in the day's meals list:

            A. Format meal data for Mealy
               - Extract: day_name, date, meal_type, meal_name, description
               - Extract: calories, protein_g, carbs_g, fat_g
               - Extract: ingredients, recipe_notes, preparation_time_min

            B. Call Mealy MCP tool to create meal
               Example: create_meal(
                 date="2025-01-06",
                 meal_type="Breakfast",
                 name="Greek Yogurt Parfait with Berries",
                 description="Protein-rich breakfast...",
                 calories=450,
                 protein_g=30,
                 carbs_g=55,
                 fat_g=10,
                 ingredients=["200g Greek yogurt", "1 cup mixed berries", ...],
                 recipe_notes="Layer yogurt with berries and granola...",
                 preparation_time_min=10
               )

            C. Capture result
               - On success: Record mealy_id from response
               - On failure: Capture error message
               - Create MealySyncStatus entry:
                 {{
                   "day_name": "Monday",
                   "date": "2025-01-06",
                   "meal_type": "Breakfast",
                   "success": true,
                   "mealy_id": "meal_12345",
                   "error_message": null
                 }}

            D. Continue with next meal (don't stop on errors)

    4. HANDLE ERRORS GRACEFULLY
        Common errors and how to handle:
        - Duplicate meal exists: Skip or update (depending on Mealy API)
        - Invalid data format: Log error, suggest data fix
        - Network timeout: Log error, suggest retry
        - Authentication failure: Log error, check MCP configuration

        For each error:
        - Log detailed error message
        - Continue processing remaining meals
        - Include error in sync_statuses

    5. RETRIEVE MEALY WEEK URL (if available)
        - After syncing all meals, try to get week view URL
        - Example MCP call: get_week_url(start_date="2025-01-06")
        - If available, include in mealy_week_url field
        - If not available, set to null

    6. CALCULATE SUMMARY STATISTICS
        - Count total meals synced successfully
        - Count total failures
        - Calculate success rate: (successes / total) × 100%

    7. WRITE INTEGRATION SUMMARY
        Provide clear, actionable summary:

        Example (all successful):
        "Successfully integrated 24 meals for the week of 2025-01-06 to 2025-01-12.
        All meals are now available in Mealy. View your week at: [URL]"

        Example (partial failure):
        "Integrated 22 of 24 meals for the week of 2025-01-06 to 2025-01-12.
        2 meals failed: Monday Dinner (duplicate exists), Wednesday Snack (network timeout).
        Please retry failed meals or create manually in Mealy."

        Example (validation failed):
        "Integration skipped: Meal plan did not pass validation. Issues found:
        [list issues]. Please address validation issues before integration."

    8. OUTPUT STRUCTURED RESULT
        Return valid JSON matching MealyIntegrationResult schema:
        - week_start_date, week_end_date
        - total_meals_created (count of successes)
        - sync_statuses (list of all sync attempts with results)
        - mealy_week_url (if available)
        - summary (human-readable description)

    IMPORTANT GUIDELINES:

    - ALWAYS attempt to sync all meals, even if some fail
    - Provide detailed error messages (not just "failed")
    - Be idempotent (safe to run multiple times)
    - Log everything for debugging
    - Give actionable feedback to user
    - If MCP tools are unavailable, report clearly and fail gracefully

    EXAMPLE OUTPUT STRUCTURE:
    {{
      "week_start_date": "2025-01-06",
      "week_end_date": "2025-01-12",
      "total_meals_created": 24,
      "sync_statuses": [
        {{
          "day_name": "Monday",
          "date": "2025-01-06",
          "meal_type": "Breakfast",
          "success": true,
          "mealy_id": "meal_12345",
          "error_message": null
        }},
        {{
          "day_name": "Monday",
          "date": "2025-01-06",
          "meal_type": "Lunch",
          "success": true,
          "mealy_id": "meal_12346",
          "error_message": null
        }},
        ...
      ],
      "mealy_week_url": "https://meal.emottet.com/week/2025-01-06",
      "summary": "Successfully integrated 24 meals for the week. All meals are now available in Mealy."
    }}

    OUTPUT CONTRACT:
    - Respond with valid JSON matching the MealyIntegrationResult schema
    - Do not wrap JSON in markdown fences
    - Include sync status for EVERY meal attempted
    - Provide detailed, actionable summary
    - Handle errors gracefully and report them clearly
    """

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the MealyIntegrationResult schema with complete sync status",
        # output_json=MealyIntegrationResult,
    )
