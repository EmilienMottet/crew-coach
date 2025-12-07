"""Task for the Meal Recipe Reviewer - macro calculation and final assembly."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from crewai import Task


def create_meal_recipe_reviewer_task(
    agent: Any,
    meal_plan_template: Dict[str, Any],
    validated_ingredients: Dict[str, Any],
    daily_target: Dict[str, Any],
    pre_calc_summary: Optional[str] = None,
) -> Task:
    """
    Create a task for the Reviewer to finalize the meal plan.

    The Reviewer calculates accurate macros from validated ingredients
    and assembles the final DailyMealPlan output.

    Args:
        agent: The Reviewer agent (no tools)
        meal_plan_template: Output from the Supervisor (MealPlanTemplate)
        validated_ingredients: Output from the Executor (ValidatedIngredientsList)
        daily_target: Original nutritional targets for validation
        pre_calc_summary: Optional pre-calculated macro summary (reduces LLM work)

    Returns:
        Task for meal plan finalization
    """
    template_json = json.dumps(meal_plan_template, indent=2)
    validated_json = json.dumps(validated_ingredients, indent=2)
    target_json = json.dumps(daily_target, indent=2)

    # Extract target values for easy reference
    target_calories = daily_target.get("calories", 2500)
    target_protein = daily_target.get("macros", {}).get("protein_g", 180)
    target_carbs = daily_target.get("macros", {}).get("carbs_g", 280)
    target_fat = daily_target.get("macros", {}).get("fat_g", 75)

    # Use simplified prompt if macros are pre-calculated
    if pre_calc_summary:
        description = f"""
Finalize the meal plan by assembling the DailyMealPlan JSON.

**YOUR ROLE**: You are the REVIEWER. Assemble the final output from pre-calculated data.
You do NOT make tool calls. The macros have been PRE-CALCULATED for you.

{pre_calc_summary}

MEAL PLAN FROM SUPERVISOR:
{template_json}

VALIDATED INGREDIENTS (with pre-calculated macros):
{validated_json}

YOUR TASK (SIMPLIFIED - MACROS ALREADY CALCULATED):

1. USE THE PRE-CALCULATED VALUES
   Each ingredient already has: protein_g, carbs_g, fat_g, calories
   Each meal has: calculated_totals with summed macros
   Daily totals are in: calculated_daily_totals

2. CHECK COMPLIANCE (already done above)
   If STATUS shows "⚠️ Out of range", adjust portions as noted in DELTAS section.

3. ASSEMBLE FINAL OUTPUT
   Create DailyMealPlan JSON using the pre-calculated values.

OUTPUT FORMAT - DailyMealPlan JSON:
```json
{{
  "day_name": "{meal_plan_template.get('day_name', 'Unknown')}",
  "date": "{meal_plan_template.get('date', 'Unknown')}",
  "meals": [
    {{
      "meal_type": "Breakfast",
      "meal_name": "...",
      "description": "...",
      "calories": <from calculated_totals>,
      "protein_g": <from calculated_totals>,
      "carbs_g": <from calculated_totals>,
      "fat_g": <from calculated_totals>,
      "preparation_time_min": 15,
      "ingredients": ["200g Greek yogurt", "100g mixed berries"],
      "validated_ingredients": [<copy from VALIDATED INGREDIENTS>],
      "recipe_notes": "..."
    }}
  ],
  "daily_totals": {{
    "protein_g": <from calculated_daily_totals>,
    "carbs_g": <from calculated_daily_totals>,
    "fat_g": <from calculated_daily_totals>,
    "calories": <from calculated_daily_totals>
  }},
  "notes": "..."
}}
```

Return ONLY valid JSON without markdown fences or commentary.
"""
    else:
        # Full prompt for when macros are not pre-calculated
        description = f"""
Finalize the meal plan by calculating accurate macros and assembling the DailyMealPlan.

**YOUR ROLE**: You are the REVIEWER. You VALIDATE and FINALIZE the meal plan.
You do NOT make tool calls. You calculate macros and assemble the final output.

DAILY TARGET:
{target_json}

MEAL PLAN FROM SUPERVISOR:
{template_json}

VALIDATED INGREDIENTS FROM EXECUTOR:
{validated_json}

YOUR TASK:

1. CALCULATE ACCURATE MACROS FROM EXECUTOR DATA

   For each ingredient in validated_ingredients, use the Passio data provided by Executor:
   - protein_per_100g, carbs_per_100g, fat_per_100g, calories_per_100g

   CALCULATION FORMULA:
   macros = (quantity_g / 100) × macros_per_100g

   EXAMPLE:
   - Ingredient: "120g chicken breast"
   - From Executor: protein_per_100g=31, carbs_per_100g=0, fat_per_100g=3.6, calories_per_100g=165
   - Calculation:
     * protein = (120/100) × 31 = 37.2g
     * carbs = (120/100) × 0 = 0g
     * fat = (120/100) × 3.6 = 4.3g
     * calories = (120/100) × 165 = 198 kcal

   For each meal, sum all ingredients' macros.
   For daily totals, sum all meals' macros

2. VALIDATE AGAINST TARGETS (STRICT)
   - Calories: {target_calories} kcal (within ±10%)
   - Protein: {target_protein}g (within ±5%)
   - Carbs: {target_carbs}g (within ±5%)
   - Fat: {target_fat}g (within ±5%)

   If outside tolerance, adjust portions accordingly.

3. ASSEMBLE FINAL OUTPUT
   Create DailyMealPlan JSON with:
   - Calculated macro values
   - validated_ingredients array from Executor output
   - Daily totals matching actual calculations
   - Notes explaining any adjustments

OUTPUT FORMAT - DailyMealPlan JSON:
```json
{{
  "day_name": "{meal_plan_template.get('day_name', 'Unknown')}",
  "date": "{meal_plan_template.get('date', 'Unknown')}",
  "meals": [
    {{
      "meal_type": "Breakfast",
      "meal_name": "Greek Yogurt Parfait with Berries",
      "description": "Creamy Greek yogurt layered with mixed berries and granola",
      "calories": 450,
      "protein_g": 25.0,
      "carbs_g": 55.0,
      "fat_g": 12.0,
      "preparation_time_min": 15,
      "ingredients": ["200g Greek yogurt", "100g mixed berries", "40g granola"],
      "validated_ingredients": [
        {{
          "name": "200g Greek yogurt",
          "passio_food_id": "abc123",
          "passio_food_name": "Greek Yogurt, plain",
          "quantity_g": 200,
          "validation_status": "found"
        }}
      ],
      "recipe_notes": "Layer yogurt and berries in a bowl. Top with granola."
    }}
  ],
  "daily_totals": {{
    "protein_g": {target_protein},
    "carbs_g": {target_carbs},
    "fat_g": {target_fat},
    "calories": {target_calories}
  }},
  "notes": "Protein target met. Minor portion adjustment to lunch."
}}
```

Return ONLY valid JSON without markdown fences or commentary.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON conforming to DailyMealPlan schema with calculated macros and validated_ingredients",
    )
