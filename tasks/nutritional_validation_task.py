"""Task for validating the nutritional quality of generated meal plans."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List

from crewai import Task

from schemas import NutritionalValidation


def create_nutritional_validation_task(
    agent: Any,
    weekly_meal_plan: Dict[str, Any],
    weekly_nutrition_plan: Dict[str, Any],
    planned_day_count: int | None = None,
) -> Task:
    """Create a task for validating the weekly meal plan."""

    planned_day_names: List[str] = [
        (plan.get("day_name") or plan.get("date") or "Unspecified day")
        for plan in weekly_meal_plan.get("daily_plans", [])
    ]

    requested_days = planned_day_count or len(planned_day_names)
    if not requested_days:
        requested_days = len(weekly_nutrition_plan.get("daily_targets", [])) or 7

    requested_days = max(1, requested_days)

    if planned_day_names:
        requested_days = min(requested_days, len(planned_day_names))

    limited_nutrition_plan = deepcopy(weekly_nutrition_plan)
    if limited_nutrition_plan.get("daily_targets"):
        limited_nutrition_plan["daily_targets"] = limited_nutrition_plan[
            "daily_targets"
        ][:requested_days]
        last_target = (
            limited_nutrition_plan["daily_targets"][-1]
            if limited_nutrition_plan["daily_targets"]
            else None
        )
        last_date = (
            (last_target.get("date") if isinstance(last_target, dict) else None)
            if last_target
            else None
        )
        if last_date:
            limited_nutrition_plan["week_end_date"] = last_date

    meal_plan_json = json.dumps(weekly_meal_plan, indent=2)
    nutrition_plan_json = json.dumps(limited_nutrition_plan, indent=2)

    expected_meals = requested_days * 4
    protein_sources_target = min(5, max(3, requested_days + 1))
    vegetable_target = min(10, max(4, requested_days * 3))
    fruit_target = min(5, max(2, requested_days * 2))
    grain_target = min(4, max(2, requested_days))
    healthy_fats_target = min(5, max(2, requested_days))
    planned_day_list = (
        ", ".join(planned_day_names[:requested_days])
        if planned_day_names
        else "None provided"
    )

    description = f"""
    Validate the generated weekly meal plan to ensure nutritional adequacy, accuracy,
    variety, and practical feasibility for athletic performance.

    SCOPE NOTE:
    - The user explicitly requested plans for the first {requested_days} consecutive day(s).
    - Planned day window: {planned_day_list}
    - Focus all checks on those {requested_days} day(s). Days beyond this window are out of scope and must not be flagged as issues.
    - Expect four meals per planned day (Breakfast, Lunch, Afternoon Snack, Dinner). A second snack is optional and should not be required for approval.

    TARGET NUTRITION PLAN:
    {nutrition_plan_json}

    GENERATED MEAL PLAN TO VALIDATE:
    {meal_plan_json}

    YOUR MISSION:

    1. VALIDATE MACRONUTRIENT ACCURACY
        For each of the {requested_days} planned day(s):
        - Sum the calories from all meals
        - Sum protein, carbs, and fat from all meals
        - Compare to the daily targets from the nutrition plan
        - Calculate variance:
          * Calories: (actual - target) / target * 100
          * Macros: (actual - target) / target * 100
        - Assess accuracy for each planned day:
          * "Excellent" if <2% variance for protein/carbs, <5% for calories
          * "Good" if 2-5% variance
          * "Needs adjustment" if >5% variance

        Create macro_accuracy dict with assessment for each planned day:
        {{
          "Monday": "Excellent - Within 1% of targets",
          "Tuesday": "Good - 3% over on carbs, acceptable",
          ...
        }}

    2. EVALUATE MEAL VARIETY
        Assess variety across the planned {requested_days} day(s):
        - Count unique meals (target: ≥ {expected_meals} unique dishes across Breakfast/Lunch/Snack/Dinner for the planned window)
        - Check protein source rotation (aim for ≥ {protein_sources_target} different primary sources across the planned days)
        - Review vegetable diversity (aim for ≥ {vegetable_target} different vegetables across those days)
        - Fruit diversity goal: ≥ {fruit_target} distinct fruits across the planned days
        - Grain/starch diversity goal: ≥ {grain_target} different grain or starch bases
        - Healthy fat diversity: ≥ {healthy_fats_target} distinct sources (olive oil, nuts, avocado, fatty fish, etc.)
        - Identify duplicated meals within the planned window (note but allow minor repeats if practicality demands)

        Assign variety_score:
        - "Excellent" if all diversity targets are met (or proportionally satisfied for the planned window)
        - "Good" if most targets are met with minor repetition or slight shortfall
        - "Needs improvement" if key diversity targets are missed or repetition is high

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
        Review food groups across the planned day(s):
        - Vegetables: Aim for ≥ {vegetable_target} different types
        - Fruits: Aim for ≥ {fruit_target} different types
        - Proteins: Aim for ≥ {protein_sources_target} different sources (animal and/or plant)
        - Grains/starches: Aim for ≥ {grain_target} types (rice, quinoa, oats, bread, pasta, etc.)
        - Healthy fats: Aim for ≥ {healthy_fats_target} varied sources (olive oil, nuts, avocado, fish)

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
        - "Only 3 protein sources used across the planned window"
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
        - At least {expected_meals} unique meals across the planned window
        - Adequate micronutrient diversity (targets above satisfied proportionally for the planned day count)
        - No critical performance nutrition issues
        - Practical feasibility is reasonable

        Set approved = False if any critical issues exist. Missing days beyond the requested {requested_days} day(s) must NOT be considered a critical issue.

    9. WRITE VALIDATION SUMMARY
        Provide clear, concise summary:
        - Overall assessment (approved or needs revision)
        - Key strengths of the meal plan
        - Critical issues (if any)
        - Overall quality rating

        Example summary:
        "Meal plan APPROVED with minor recommendations. Macro accuracy is excellent across
        the planned window (average variance <2%). Meal variety is good with distinct dishes
        and protein sources. Practicality is excellent with realistic prep times. Minor
        recommendation: add more colorful vegetables for micronutrient optimization."

    OUTPUT CONTRACT:
    - Respond with valid JSON matching the NutritionalValidation schema EXACTLY
    - Required top-level fields (no nesting allowed):
      * approved: boolean
      * validation_summary: string (overall summary)
      * macro_accuracy: dict (key=day_name, value=assessment string)
      * variety_score: string (overall variety assessment)
      * practicality_score: string (overall practicality assessment)
      * issues_found: list of strings
      * recommendations: list of strings
    - Do NOT nest variety_score or practicality_score inside other objects
    - Do NOT create extra top-level fields like "per_day_macros" or "approval_decision_basis"
    - Do not wrap JSON in markdown fences
    - Be thorough and specific in assessments
    - Provide actionable recommendations
    - Make clear approval decision based on objective criteria
    """

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the NutritionalValidation schema with complete validation assessment",
        # DISABLED: output_json causes auth issues with instructor when using custom LLM configs
        # The JSON is parsed manually in crew_mealy.py instead
        # output_json=NutritionalValidation,
    )
