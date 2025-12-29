"""Task for integrating meal plans into Hexis."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict

from crewai import Task

from schemas import HexisIntegrationResult


def create_hexis_integration_task(
    agent: Any,
    weekly_meal_plan: Dict[str, Any],
    validation_result: Dict[str, Any],
    nutrition_plan: Dict[str, Any] | None = None,
    planned_day_count: int | None = None,
) -> Task:
    """
    Create a task for integrating the meal plan into Hexis.

    Args:
        agent: The agent responsible for this task
        weekly_meal_plan: The validated meal plan (WeeklyMealPlan as dict)
        validation_result: The validation assessment (NutritionalValidation as dict)
        nutrition_plan: The nutrition targets to use for validation (prevents hallucination)

    Returns:
        Configured Task instance
    """
    meal_plan_json = json.dumps(weekly_meal_plan, indent=2)
    validation_json = json.dumps(validation_result, indent=2)

    # Build explicit daily targets section to prevent LLM hallucination
    daily_targets_section = ""
    if nutrition_plan and "daily_targets" in nutrition_plan:
        targets_list = []
        for t in nutrition_plan.get("daily_targets", []):
            targets_list.append(
                {
                    "date": t.get("date", "?"),
                    "day_name": t.get("day_name", "?"),
                    "calories": t.get("calories", 0),
                    "protein_g": t.get("protein_g", 0),
                    "carbs_g": t.get("carbs_g", 0),
                    "fat_g": t.get("fat_g", 0),
                }
            )
        daily_targets_json = json.dumps(targets_list, indent=2)
        daily_targets_section = f"""
        ⚠️ CIBLES JOURNALIÈRES DE RÉFÉRENCE (NE PAS DÉDUIRE D'AUTRES SOURCES):
        {daily_targets_json}

        IMPORTANT: Si vous devez valider les macros des repas, utilisez UNIQUEMENT ces cibles ci-dessus.
        Ne déduisez PAS les cibles à partir d'autres données dans le contexte!
        """

    planned_days = planned_day_count or len(weekly_meal_plan.get("daily_plans", []))
    if not planned_days:
        planned_days = 1

    planned_days = max(1, planned_days)
    expected_meals = planned_days * 4

    description = dedent(
        f"""
        Integrate the validated weekly meal plan into Hexis with comprehensive
        error handling and status reporting.

        VALIDATION RESULT:
        {validation_json}
        {daily_targets_section}
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

        3. USE hexis_log_meal TO LOG EACH MEAL

            ⚠️⚠️⚠️ CRITICAL - READ THIS FIRST ⚠️⚠️⚠️

            **validated_ingredients IS MANDATORY** when present in the meal data!
            If you do NOT pass validated_ingredients, the tool will search for the meal name
            (e.g., "Greek Yogurt Protein Bowl") which will FAIL because composed recipes
            don't exist in the Passio database - only individual foods do.

            ❌ WRONG (will fail):
            hexis_log_meal(
                meal_type="Breakfast",
                date="2025-12-04",
                meal_name="Greek Yogurt Protein Bowl with Nuts",
                calories=450,
                protein=25,
                carbs=55,
                fat=12
            )
            → Error: "No food found for 'Greek Yogurt Protein Bowl with Nuts'"

            ✅ CORRECT (will succeed):
            hexis_log_meal(
                meal_type="Breakfast",
                date="2025-12-04",
                meal_name="Greek Yogurt Protein Bowl with Nuts",
                calories=450,
                protein=25,
                carbs=55,
                fat=12,
                validated_ingredients=[
                    {{"name": "200g Greek yogurt", "passio_food_id": "abc123", "passio_food_name": "Greek Yogurt"}}
                ]
            )
            → Success: Uses pre-validated Passio ID, skips search

            For each day in the meal plan, for each meal:

            You MUST call `hexis_log_meal`. You cannot generate the Hexis ID yourself.
            Any ID you invent will be wrong. You MUST get it from the tool.

            Call `hexis_log_meal` with:
            - day_name
            - date
            - meal_type
            - meal_name (descriptive name like "Chicken Salad", NOT "Lunch" or "Dinner")
            - description (ingredients or details to help search if validated_ingredients missing)
            - calories
            - protein
            - carbs
            - fat
            - **validated_ingredients** (MANDATORY if present in meal data - copy the entire array!)
              Format: [{{"name": "120g chicken breast", "passio_food_id": "abc123", "passio_food_name": "Chicken Breast", "quantity_g": 120}}, ...]

            The tool will return a success status and the Hexis ID.

            Track the result:
            - If successful: Mark sync status as success
            - If failed: Log error message, mark as failed

        4. HANDLE ERRORS GRACEFULLY
           Common errors and how to handle:
           - Invalid data format: Log error, suggest data fix
           - Network timeout: Log error, suggest retry
           - Authentication failure: Log error, check MCP configuration

           For each error:
           - Log detailed error message
           - Continue processing remaining meals
           - Include error in sync_statuses

        5. CALCULATE SUMMARY STATISTICS
           - Count total meals synced successfully
           - Count total failures
           - Calculate success rate: (successes / total) × 100%

        6. WRITE INTEGRATION SUMMARY
           Provide clear, actionable summary:

           Example (all successful):
           "Successfully integrated every planned meal ({expected_meals} entries) for the requested period
           from 2025-01-06 to 2025-01-12. All meals are now logged in Hexis."

           Example (partial failure):
           "Integrated most meals (all but two of the planned {expected_meals}) for the period
           2025-01-06 to 2025-01-12. Remaining issues: Monday Dinner (API error), Wednesday Snack (network timeout).
           Please retry failed meals manually in Hexis."

           Example (validation failed):
           "Integration skipped: Meal plan did not pass validation. Issues found:
           [list issues]. Please address validation issues before integration."

        7. OUTPUT STRUCTURED RESULT
           - CRITICAL: Do NOT output this JSON until you have successfully called hexis_log_meal for all meals.
           - If you have not called hexis_log_meal for a meal, GO BACK and do it.
           Return valid JSON matching HexisIntegrationResult schema:
           - week_start_date, week_end_date
           - total_meals_created (count of successes)
           - sync_statuses (list of all sync attempts with results)
           - summary (human-readable description)

        IMPORTANT GUIDELINES:

        - ALWAYS attempt to sync all meals, even if some fail
        - Provide detailed error messages (not just "failed")
        - Be idempotent (safe to run multiple times)
        - Log everything for debugging
        - Give actionable feedback to user
        - If MCP tools are unavailable, report clearly and fail gracefully
        - CRITICAL: Do NOT try to search for individual ingredients. Use the Custom Food strategy.

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
              "hexis_id": null,
              "error_message": null
            }},
            {{
              "day_name": "Monday",
              "date": "2025-01-06",
              "meal_type": "Lunch",
              "success": true,
              "hexis_id": null,
              "error_message": null
            }},
            ...
          ],
          "summary": "Successfully integrated every planned meal ({expected_meals} total). All meals are now logged in Hexis."
        }}

        OUTPUT CONTRACT:
        - Respond with valid JSON matching the HexisIntegrationResult schema
        - Do not wrap JSON in markdown fences
        - Include sync status for EVERY meal attempted
        - Provide detailed, actionable summary
        - Handle errors gracefully and report them clearly
        """
    ).strip()

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the HexisIntegrationResult schema with complete sync status",
        # DISABLED: output_json causes auth issues with instructor when using custom LLM configs
        # output_json=HexisIntegrationResult,
    )


# Keep old name for backwards compatibility
def create_mealy_integration_task(
    agent: Any,
    weekly_meal_plan: Dict[str, Any],
    validation_result: Dict[str, Any],
    nutrition_plan: Dict[str, Any] | None = None,
    planned_day_count: int | None = None,
) -> Task:
    """Deprecated: Use create_hexis_integration_task instead."""
    return create_hexis_integration_task(
        agent, weekly_meal_plan, validation_result, nutrition_plan, planned_day_count
    )
