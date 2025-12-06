"""Task for the Ingredient Validation Executor - tool calls for Passio search."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from crewai import Task


def create_ingredient_validation_executor_task(
    agent: Any,
    meal_plan_template: Dict[str, Any],
) -> Task:
    """
    Create a task for the Executor to validate ingredients.

    The Executor searches the Passio food database for each ingredient
    and records the passio_food_id. This is CRITICAL for Hexis integration.

    Args:
        agent: The Executor agent (with hexis_search_passio_foods tool)
        meal_plan_template: Output from the Supervisor (MealPlanTemplate)

    Returns:
        Task for ingredient validation
    """
    template_json = json.dumps(meal_plan_template, indent=2)

    # Extract all ingredients to validate
    ingredients_list: List[str] = []
    if "meals" in meal_plan_template:
        for meal in meal_plan_template["meals"]:
            meal_name = meal.get("meal_name", "Unknown")
            ingredients = meal.get("ingredients_to_validate", [])
            for ing in ingredients:
                ingredients_list.append(f"[{meal_name}] {ing}")

    ingredients_preview = "\n".join(f"  - {ing}" for ing in ingredients_list[:20])
    if len(ingredients_list) > 20:
        ingredients_preview += f"\n  ... and {len(ingredients_list) - 20} more"

    description = f"""
Validate ingredients from the meal plan against the Passio food database.

**YOUR ROLE**: You are the EXECUTOR. You ONLY make tool calls to validate ingredients.
You do NOT design meals or calculate macros. Just search and record IDs.

MEAL PLAN FROM SUPERVISOR:
{template_json}

INGREDIENTS TO VALIDATE ({len(ingredients_list)} total):
{ingredients_preview}

YOUR TASK (2 STEPS PER INGREDIENT):

STEP 1: Search for the ingredient
- Call `hexis_search_passio_foods` with a simple search term
- Record passio_food_id, passio_ref_code, passio_food_name

STEP 2: Get nutritional details
- Call `hexis_get_passio_food_details` with the ref_code from Step 1
- Extract and record nutritional data per 100g:
  * protein_per_100g
  * carbs_per_100g
  * fat_per_100g
  * calories_per_100g

Focus on proteins, carbs, and fats - skip minor spices/herbs

SEARCH STRATEGY:
- Use SIMPLE terms: "chicken breast", "rice", "olive oil", "eggs"
- If not found, try simpler: "chicken" instead of "chicken breast"
- Search in English first, then French if needed
- Take the FIRST result if multiple matches

CRITICAL - CAPTURE refCode:
For each food found, you MUST record BOTH:
- passio_food_id: The resultId or id from the search result
- passio_ref_code: The refCode field (base64 string) - THIS IS REQUIRED for Hexis API!
Without refCode, meal logging will fail with 400 Bad Request.

IMPORTANT - USE limit PARAMETER:
When calling hexis_search_passio_foods, ALWAYS specify limit=5 to reduce response size
and avoid context overflow. Without limit, the API returns 100+ results (~150K chars per query).

EXAMPLE TOOL CALLS:

For "120g chicken breast":
```
# Step 1: Search
Action: hexis_search_passio_foods
Action Input: {{"query": "chicken breast", "limit": 5}}
# Result: {{"id": "abc123", "name": "Chicken Breast, raw", "refCode": "eyJ..."}}

# Step 2: Get nutrition details
Action: hexis_get_passio_food_details
Action Input: {{"ref_code": "eyJ..."}}
# Result: {{"protein": 31.0, "carbs": 0.0, "fat": 3.6, "calories": 165}}
```

For "200g cooked quinoa":
```
# Step 1: Search
Action: hexis_search_passio_foods
Action Input: {{"query": "quinoa", "limit": 5}}

# Step 2: Get nutrition details
Action: hexis_get_passio_food_details
Action Input: {{"ref_code": "<refCode from search result>"}}
```

OUTPUT FORMAT - ValidatedIngredientsList JSON:
```json
{{
  "day_name": "{meal_plan_template.get('day_name', 'Unknown')}",
  "date": "{meal_plan_template.get('date', 'Unknown')}",
  "validated_meals": [
    {{
      "meal_type": "Breakfast",
      "meal_name": "Greek Yogurt Parfait",
      "validated_ingredients": [
        {{
          "name": "200g Greek yogurt",
          "passio_food_id": "abc123",
          "passio_ref_code": "eyJ...",
          "passio_food_name": "Greek Yogurt, plain",
          "quantity_g": 200,
          "validation_status": "found",
          "protein_per_100g": 10.0,
          "carbs_per_100g": 4.0,
          "fat_per_100g": 0.7,
          "calories_per_100g": 59.0
        }},
        {{
          "name": "100g mixed berries",
          "passio_food_id": "def456",
          "passio_ref_code": "eyJ...",
          "passio_food_name": "Berries, mixed frozen",
          "quantity_g": 100,
          "validation_status": "found",
          "protein_per_100g": 0.7,
          "carbs_per_100g": 12.0,
          "fat_per_100g": 0.3,
          "calories_per_100g": 57.0
        }}
      ],
      "validation_success": true
    }}
    // ... more meals
  ],
  "total_validations": 16,
  "successful_validations": 14,
  "substitutions_made": 2
}}
```

IMPORTANT - MACRO DATA IS REQUIRED:
The Reviewer agent uses protein_per_100g, carbs_per_100g, fat_per_100g, calories_per_100g
to calculate accurate daily totals. Without this data, validation will fail!

VALIDATION STATUS VALUES:
- "found": Ingredient found exactly in Passio database
- "substituted": Similar ingredient used (add substitution_note)
- "not_found": Could not find in database (rare - try harder!)

IMPORTANT:
- Validate at least the MAIN ingredients for each meal (3-5 per meal)
- Skip trivial ingredients (salt, pepper, herbs)
- If substituting, note what was substituted and why
- Count your validations accurately in the totals

Return ONLY valid JSON without markdown fences or commentary.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON conforming to ValidatedIngredientsList schema with passio_food_id for each validated ingredient",
    )
