"""Task for validating the nutritional quality of generated meal plans."""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task

from schemas import NutritionalValidation


def create_nutritional_validation_task(
    agent: Any,
    weekly_meal_plan: Dict[str, Any],
    weekly_nutrition_plan: Dict[str, Any],
) -> Task:
    """
    Create a task for validating the weekly meal plan.

    Args:
        agent: The agent responsible for this task
        weekly_meal_plan: The generated meal plan (WeeklyMealPlan as dict)
        weekly_nutrition_plan: The target nutrition plan (WeeklyNutritionPlan as dict)

    Returns:
        Configured Task instance
    """
    meal_plan_json = json.dumps(weekly_meal_plan, indent=2)
    nutrition_plan_json = json.dumps(weekly_nutrition_plan, indent=2)

    description = f"""
    Validate the generated weekly meal plan to ensure nutritional adequacy, accuracy,
    variety, and practical feasibility for athletic performance.

    TARGET NUTRITION PLAN:
    {nutrition_plan_json}

    GENERATED MEAL PLAN TO VALIDATE:
    {meal_plan_json}

    YOUR MISSION:

    1. VALIDATE MACRONUTRIENT ACCURACY
        For each day of the week:
        - Sum the calories from all meals
        - Sum protein, carbs, and fat from all meals
        - Compare to the daily targets from the nutrition plan
        - Calculate variance:
          * Calories: (actual - target) / target * 100
          * Macros: (actual - target) / target * 100
        - Assess accuracy for each day:
          * "Excellent" if <2% variance for protein/carbs, <5% for calories
          * "Good" if 2-5% variance
          * "Needs adjustment" if >5% variance

        Create macro_accuracy dict with assessment for each day:
        {{
          "Monday": "Excellent - Within 1% of targets",
          "Tuesday": "Good - 3% over on carbs, acceptable",
          ...
        }}

    2. EVALUATE MEAL VARIETY
        Assess variety across the week:
        - Count unique meals (should be 21+ unique meals for 7 days × 3 main meals)
        - Check protein source rotation (should include 5+ different sources)
        - Review vegetable diversity (should include 10+ different vegetables)
        - Check cuisine diversity (should span 3+ cuisine types)
        - Identify any repeated meals (flag if any)

        Assign variety_score:
        - "Excellent" if 21+ unique meals, diverse proteins, varied cuisines
        - "Good" if 18-20 unique meals, some repetition acceptable
        - "Needs improvement" if <18 unique meals or excessive repetition

    3. ASSESS PRACTICAL FEASIBILITY
        Evaluate practicality:
        - Check preparation times (flag if >50% of meals >45 min)
        - Review ingredient accessibility (flag if exotic ingredients)
        - Assess portion realism (flag if portions seem unrealistic)
        - Evaluate meal prep friendliness
        - Check shopping list manageability

        Assign practicality_score:
        - "Excellent" if all criteria met, realistic for busy athlete
        - "Good" if mostly practical with minor challenges
        - "Needs improvement" if unrealistic time demands or ingredient availability

    4. CHECK MICRONUTRIENT DIVERSITY
        Review food groups:
        - Vegetables: Should include 10+ different types across the week
        - Fruits: Should include 5+ different types
        - Proteins: Should include 5+ sources (animal and/or plant)
        - Grains: Should include 4+ types (rice, quinoa, oats, bread, pasta, etc.)
        - Healthy fats: Should include varied sources (olive oil, nuts, avocado, fish)

        Flag if any food group is underrepresented.

    5. VALIDATE ATHLETIC PERFORMANCE OPTIMIZATION
        Check performance nutrition principles:
        - Pre-workout meals: Should be moderate-high carb, low fat, easily digestible
        - Post-workout meals: Should include protein (20-40g) + carbs
        - Rest days: Should be balanced, not overly carb-heavy
        - Hydration-rich foods: Adequate fruits and vegetables
        - Anti-inflammatory foods: Include omega-3 sources, colorful vegetables

        Flag if any critical performance nutrition principles are violated.

    6. IDENTIFY ISSUES (IF ANY)
        List specific issues found:
        - "Monday exceeds calorie target by 8% (2698 vs 2500 kcal)"
        - "Only 3 protein sources used across the week (chicken, eggs, salmon)"
        - "Dinner preparation times average 55 minutes, may be unrealistic for weeknights"
        - "Insufficient pre-workout carbohydrate in Wednesday breakfast before intervals"

        Categorize issues by severity:
        - Critical: Macro targets missed by >10%, repeated meals, missing food groups
        - Moderate: Macro targets off by 5-10%, limited variety, challenging prep times
        - Minor: Macro targets off by 2-5%, minor variety issues, ingredient availability

    7. PROVIDE RECOMMENDATIONS
        For each issue identified, suggest specific fix:
        - "Reduce Monday dinner portion by 20g chicken and 50g rice to meet calorie target"
        - "Add turkey, beef, and legumes to increase protein source variety"
        - "Replace Wednesday breakfast with quick oatmeal + banana for pre-workout carbs"
        - "Simplify 2-3 dinners to reduce average prep time below 40 minutes"

        Prioritize recommendations by impact (fix critical issues first).

    8. MAKE APPROVAL DECISION
        Set approved = True ONLY if:
        - All daily macros within ±50 kcal of targets
        - Macro variance <5% for protein and carbs
        - At least 18 unique meals across the week
        - Adequate micronutrient diversity (all food groups represented)
        - No critical performance nutrition issues
        - Practical feasibility is reasonable

        Set approved = False if any critical issues exist.

    9. WRITE VALIDATION SUMMARY
        Provide clear, concise summary:
        - Overall assessment (approved or needs revision)
        - Key strengths of the meal plan
        - Critical issues (if any)
        - Overall quality rating

        Example summary:
        "Meal plan APPROVED with minor recommendations. Macro accuracy is excellent across
        all days (average variance <2%). Meal variety is good with 20 unique meals and
        diverse protein sources. Practicality is excellent with realistic prep times.
        Minor recommendation: add more colorful vegetables for micronutrient optimization."

    OUTPUT CONTRACT:
    - Respond with valid JSON matching the NutritionalValidation schema
    - Do not wrap JSON in markdown fences
    - Be thorough and specific in assessments
    - Provide actionable recommendations
    - Make clear approval decision based on objective criteria
    """

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the NutritionalValidation schema with complete validation assessment",
        output_json=NutritionalValidation,
    )
