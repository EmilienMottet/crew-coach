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
    
    # Check if there's validation feedback from a previous attempt
    validation_feedback = weekly_nutrition_plan.get('validation_feedback')
    feedback_section = ""
    
    if validation_feedback:
        attempt = validation_feedback.get('attempt', 0)
        issues = validation_feedback.get('issues', [])
        recommendations = validation_feedback.get('recommendations', [])
        
        feedback_section = f"""

⚠️⚠️⚠️ CRITICAL - PREVIOUS ATTEMPT #{attempt} WAS REJECTED ⚠️⚠️⚠️

The validator found {len(issues)} issues with your previous meal plan.
You MUST address these issues in this new attempt:

ISSUES FOUND:
{chr(10).join(f"  {i+1}. {issue}" for i, issue in enumerate(issues[:5]))}

RECOMMENDATIONS TO FOLLOW:
{chr(10).join(f"  • {rec}" for rec in recommendations[:5])}

SPECIFIC ACTIONS TO TAKE:
1. REDUCE portion sizes if calories were too high
2. INCREASE portions if calories were too low
3. ADJUST protein sources if protein was off-target
4. REBALANCE carbs/fats if macros were imbalanced
5. DO NOT repeat the same mistakes

This is attempt #{attempt + 1}. Make it count!
"""

    description = f"""
Generate complete weekly meal plan hitting nutrition targets with variety and appeal.

NUTRITION PLAN:
{nutrition_json}
{feedback_section}

⚠️ CRITICAL REQUIREMENT - STRICT MACRO ADHERENCE:
YOU MUST MATCH THE DAILY TARGETS EXACTLY. This is NON-NEGOTIABLE.
- Calories: Within ±3% (max ±75 kcal for 2500 kcal target)
- Protein: Within ±5g of target
- Carbs: Within ±10g of target  
- Fat: Within ±5g of target

MACRO CALCULATION RULES:
1. START with the daily targets from the nutrition plan
2. DIVIDE calories across meals: Breakfast 25%, Lunch 35%, Dinner 35%, Snacks 5%
3. For EACH meal, calculate exact portions needed:
   - Protein: 4 kcal/g
   - Carbs: 4 kcal/g
   - Fat: 9 kcal/g
4. VERIFY daily totals add up to targets BEFORE finalizing
5. ADJUST portion sizes (not add/remove meals) if totals don't match

PORTION SIZE GUIDELINES (to prevent overshooting):
- Chicken breast: 100-140g raw (not 150-180g)
- Fish: 120-150g raw
- Rice/pasta (cooked): 100-150g (not 200g+)
- Protein powder: 1 scoop (30g) per shake (not 1.5-2 scoops)
- Oats: 50-80g dry (not 100-150g)
- Avoid multiple high-calorie snacks on same day

TASKS:
1. Optional: use Mealy MCP for user preferences, dietary restrictions, past meals
2. For each day (Mon-Sun), create:
   - Breakfast (25% daily cals): quick prep, consider workout timing
   - Lunch (35% daily cals): balanced, portable
   - Dinner (35% daily cals): recovery-focused if post-workout
   - Snacks (5-10% daily cals, 1-2 max): strategic timing only when needed
3. Variety: rotate proteins (chicken/fish/beef/eggs/legumes), cooking methods, cuisines, grains, vegetables
4. Each meal: meal_type, meal_name, description, calories, protein_g, carbs_g, fat_g, preparation_time_min, ingredients (list with quantities), recipe_notes (brief cooking instructions)
5. ⚠️ VERIFY: Sum all meal macros and compare to daily targets. Adjust portions if needed.
6. Shopping list: organized by category, aggregated quantities
7. Meal prep tips: batch cooking, pre-prep tasks, storage

Guidelines: no repeated meals, common ingredients, realistic home cooking, whole foods, athletic needs (pre-workout: digestible/high carb/low fat, post-workout: protein+carbs).

⚠️ BEFORE FINALIZING: Double-check EVERY day's totals match the targets from the nutrition plan.

Return JSON: week_start_date, week_end_date, daily_meal_plans (array with day_name/date/meals array for 7 days), shopping_list, meal_prep_tips.
No markdown fences. Make delicious and practical.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the WeeklyMealPlan schema with complete daily meal plans, shopping list, and prep tips",
        # output_json=WeeklyMealPlan,
    )
