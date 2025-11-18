"""Task for integrating meal plans into Mealy."""
from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict

from crewai import Task

from schemas import MealyIntegrationResult


def create_mealy_integration_task(
    agent: Any,
    weekly_meal_plan: Dict[str, Any],
    validation_result: Dict[str, Any],
    planned_day_count: int | None = None,
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

    planned_days = planned_day_count or len(weekly_meal_plan.get("daily_plans", []))
    if not planned_days:
        planned_days = 1

    planned_days = max(1, planned_days)
    expected_meals = planned_days * 4

    description = dedent(
        f"""
        Integrate the validated weekly meal plan into Mealy with comprehensive
        error handling and status reporting.

        VALIDATION RESULT:
        {validation_json}

        MEAL PLAN TO INTEGRATE:
        {meal_plan_json}

        SCOPE NOTE:
        - This plan intentionally covers {planned_days} consecutive day(s) starting from week_start_date.
        - Expect approximately {expected_meals} meal entries (4 meals per day: Breakfast, Lunch, Afternoon Snack, Dinner).
        - Do not flag missing data for days beyond this requested range; integrate exactly what is provided.

        YOUR MISSION:

        1. VALIDATE INPUT DATA
            - Check that validation_result.approved == True
            - If not approved, do NOT proceed with integration
            - Report that integration was skipped due to validation failure
            - If approved, proceed to integration

        2. PREPARE FOR INTEGRATION
            - Extract week_start_date and week_end_date
            - Count total meals to sync (expected ≈ {expected_meals} meals: 4 meals × {planned_days} day(s))
            - Initialize tracking for sync results

        3. USE MEALY BULK UPLOAD TO SYNC ALL MEALS AT ONCE

            A. Convert WeeklyMealPlan to Mealy bulk format
               Build an entries array with ALL meals from the requested range:

               entries = []
               For each day in daily_plans:
                 For each meal in day.meals:
                   entry = {{
                     "date": day.date,  # ISO format "YYYY-MM-DD"
                     "title": f"{{meal.meal_type}}: {{meal.meal_name}}",
                     "entry_type": map_meal_type_to_mealy(meal.meal_type)
                   }}
                   entries.append(entry)

               Meal type mapping:
               - "Breakfast" → "breakfast"
               - "Lunch" → "lunch"
               - "Dinner" → "dinner"
               - "Afternoon Snack" or "Snack" → "side"

            B. Call mealy__create_mealplan_bulk ONCE with all entries
               Example:
               result = mealy__create_mealplan_bulk(entries=entries)

               This sends all {expected_meals} meal(s) ({planned_days} day(s) × 4 meals) in a single API call.

            C. Parse bulk response and create sync statuses
               The tool returns a response indicating success/failure.
               For each entry in your entries list:
                 - If bulk call succeeded: Mark as success
                 - If bulk call failed: Mark all as failed with error message

               Create MealySyncStatus for each meal:
               {{
                 "day_name": "Monday",
                 "date": "2025-01-06",
                 "meal_type": "Breakfast",
                 "success": true/false,
                 "mealy_id": null,  # Bulk doesn't return individual IDs
                 "error_message": error if failed
               }}

            D. FALLBACK: If bulk upload fails, consider individual upload
               If mealy__create_mealplan_bulk is not available or fails,
               you can fall back to individual meal creation (if available).
               However, bulk is preferred for efficiency.

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
            "Successfully integrated every planned meal ({expected_meals} entries) for the requested period
            from 2025-01-06 to 2025-01-12. All meals are now available in Mealy. View your week at: [URL]"

            Example (partial failure):
            "Integrated most meals (all but two of the planned {expected_meals}) for the period
            2025-01-06 to 2025-01-12. Remaining issues: Monday Dinner (duplicate exists), Wednesday Snack (network timeout).
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
          "total_meals_created": {expected_meals},
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
          "summary": "Successfully integrated every planned meal ({expected_meals} total). All meals are now available in Mealy."
        }}

        OUTPUT CONTRACT:
        - Respond with valid JSON matching the MealyIntegrationResult schema
        - Do not wrap JSON in markdown fences
        - Include sync status for EVERY meal attempted
        - Provide detailed, actionable summary
        - Handle errors gracefully and report them clearly
        """
    ).strip()

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the MealyIntegrationResult schema with complete sync status",
        output_json=MealyIntegrationResult,
    )
