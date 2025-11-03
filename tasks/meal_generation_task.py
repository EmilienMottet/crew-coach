"""Task for generating complete weekly meal plans."""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task

from schemas import WeeklyMealPlan


def create_meal_generation_task(
    agent: Any,
    weekly_nutrition_plan: Dict[str, Any]
) -> Task:
    """
    Create a task for generating a complete weekly meal plan.

    Args:
        agent: The agent responsible for this task
        weekly_nutrition_plan: The structured nutrition plan (WeeklyNutritionPlan as dict)

    Returns:
        Configured Task instance
    """
    nutrition_json = json.dumps(weekly_nutrition_plan, indent=2)

    description = f"""
Generate complete weekly meal plan hitting nutrition targets with variety and appeal.

NUTRITION PLAN:
{nutrition_json}

TASKS:
1. Optional: use Mealy MCP for user preferences, dietary restrictions, past meals
2. For each day (Mon-Sun), create:
   - Breakfast (25-30% cals): quick prep, consider workout timing
   - Lunch (30-35% cals): balanced, portable
   - Dinner (30-35% cals): recovery-focused if post-workout
   - Snacks (10-20% cals, 1-2): strategic timing
3. Variety: rotate proteins (chicken/fish/beef/eggs/legumes), cooking methods, cuisines, grains, vegetables
4. Each meal: meal_type, meal_name, description, calories, protein_g, carbs_g, fat_g, preparation_time_min, ingredients (list with quantities), recipe_notes (brief cooking instructions)
5. Daily totals must match targets (±50 cal)
6. Shopping list: organized by category, aggregated quantities
7. Meal prep tips: batch cooking, pre-prep tasks, storage

Guidelines: no repeated meals, common ingredients, realistic home cooking, whole foods, athletic needs (pre-workout: digestible/high carb/low fat, post-workout: protein+carbs).

Return JSON: week_start_date, week_end_date, daily_meal_plans (array with day_name/date/meals array for 7 days), shopping_list, meal_prep_tips.
No markdown fences. Make delicious and practical.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the WeeklyMealPlan schema with complete daily meal plans, shopping list, and prep tips",
        output_json=WeeklyMealPlan,
    )
