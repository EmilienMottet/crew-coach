"""Task for the Meal Planning Supervisor - pure reasoning meal design."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable

from crewai import Task


def create_meal_planning_supervisor_task(
    agent: Any,
    daily_target: Dict[str, Any],
    weekly_context: Dict[str, Any],
    previous_days: Iterable[Dict[str, Any]] | None = None,
    validation_feedback: Dict[str, Any] | None = None,
    meal_targets: list[Dict[str, Any]] | None = None,
) -> Task:
    """
    Create a task for the Supervisor to design a day's meal plan.

    The Supervisor designs meals WITHOUT making tool calls. The output
    is a MealPlanTemplate that will be passed to the Executor for
    ingredient validation.

    Args:
        agent: The Supervisor agent
        daily_target: Nutritional targets for the day (daily totals)
        weekly_context: Weekly context and strategy
        previous_days: Already planned days (to avoid duplicates)
        validation_feedback: Feedback from previous validation attempts
        meal_targets: Per-meal targets from Hexis (REQUIRED - contains exact calories/macros/carbCode)

    Returns:
        Task for meal plan design

    Raises:
        ValueError: If meal_targets is missing or incomplete
    """
    # Validate meal_targets - REQUIRED as per user preference
    if not meal_targets:
        raise ValueError(
            "Hexis meal targets are required. meal_targets cannot be None or empty."
        )

    # Build per-meal targets section with dynamic meal count
    meal_count = len(meal_targets)
    meal_types_list = [mt.get("meal_type", "Unknown") for mt in meal_targets]
    meal_types_str = ", ".join(meal_types_list)

    meal_targets_lines = []
    for i, mt in enumerate(meal_targets, 1):
        carb_code = mt.get("carb_code", "MEDIUM")
        carb_guidance = {
            "LOW": "proteins, healthy fats, low-GI vegetables",
            "MEDIUM": "balanced carbs from whole grains, fruits",
            "HIGH": "high-carb recovery foods, pasta, rice, bread",
        }.get(carb_code, "balanced macros")

        meal_targets_lines.append(
            f"  {i}. {mt.get('meal_type', 'Unknown')}: "
            f"{mt.get('calories', 0)} kcal, "
            f"{mt.get('protein_g', 0)}g protein, "
            f"{mt.get('carbs_g', 0)}g carbs, "
            f"{mt.get('fat_g', 0)}g fat | "
            f"carbCode={carb_code} → {carb_guidance}"
        )
    meal_targets_section = "\n".join(meal_targets_lines)

    target_json = json.dumps(daily_target, indent=2)
    weekly_json = json.dumps(weekly_context, indent=2)
    previous_days_json = json.dumps(list(previous_days or []), indent=2)

    # Extract target values for the mandatory fields section
    day_target_calories = daily_target.get("calories", 0)
    day_target_macros = daily_target.get("macros", {})
    day_target_protein = day_target_macros.get("protein_g", 0)
    day_target_carbs = day_target_macros.get("carbs_g", 0)
    day_target_fat = day_target_macros.get("fat_g", 0)
    day_name_value = daily_target.get("day_name", "Monday")
    day_date_value = daily_target.get("date", "2025-01-20")

    feedback_section = ""
    if validation_feedback:
        attempt = validation_feedback.get("previous_attempt", 0)
        issues = validation_feedback.get("issues_found", [])
        recommendations = validation_feedback.get("recommendations", [])

        feedback_section = f"""

⚠️ PRIOR VALIDATION ATTEMPT #{attempt} FAILED - REVISE YOUR PLAN

Issues Found:
{chr(10).join(f"  - {issue}" for issue in issues[:8])}

Recommendations:
{chr(10).join(f"  ✓ {rec}" for rec in recommendations[:8])}

This is attempt #{attempt + 1}. Please adjust your meal plan accordingly.
"""

        # Handle per-day macro validation feedback
        macro_issues = validation_feedback.get("macro_issues", [])
        adjustments = validation_feedback.get("adjustments_needed", [])
        missing_meals = validation_feedback.get("missing_meals", [])

        if macro_issues or adjustments or missing_meals:
            feedback_section += f"""

⚠️ PER-DAY MACRO VALIDATION FAILED:

"""
            if missing_meals:
                feedback_section += f"""Missing meal types (MUST be included):
{chr(10).join(f"  ❌ {meal}" for meal in missing_meals)}

"""
            if macro_issues:
                feedback_section += f"""Macro Issues:
{chr(10).join(f"  - {issue}" for issue in macro_issues[:5])}

"""
            if adjustments:
                feedback_section += f"""REQUIRED ADJUSTMENTS (you MUST apply these):
{chr(10).join(f"  ✓ {adj}" for adj in adjustments[:5])}
"""

    description = f"""
Design a complete meal plan for a single day that matches the nutritional targets.

**YOUR ROLE**: You are the SUPERVISOR. You DESIGN meals using your culinary expertise.
You do NOT make any tool calls. A separate Executor agent will validate your
ingredient choices against the Passio food database.

**CRITICAL**: You MUST create exactly {meal_count} meals, one for each Hexis target:
  → {meal_types_str}

DAILY TARGET:
{target_json}

WEEKLY CONTEXT (use for variety and strategy):
{weekly_json}

EXISTING DAYS TO AVOID DUPLICATES:
{previous_days_json}
{feedback_section}

⚠️ MANDATORY ROOT FIELDS - YOUR JSON MUST INCLUDE THESE (validation fails without them):
Copy these EXACT values from the DAILY TARGET above into your JSON response:
  "target_calories": {day_target_calories}
  "target_protein": {day_target_protein}
  "target_carbs": {day_target_carbs}
  "target_fat": {day_target_fat}

OUTPUT REQUIREMENTS - MealPlanTemplate JSON:

Return a JSON object with exactly {meal_count} meals in the "meals" array.
Each meal MUST match one of the Hexis meal targets below.

```json
{{
  "day_name": "{day_name_value}",
  "date": "{day_date_value}",
  "training_context": "Rest day - moderate carbs, high protein",
  "target_calories": {day_target_calories},
  "target_protein": {day_target_protein},
  "target_carbs": {day_target_carbs},
  "target_fat": {day_target_fat},
  "meals": [
    {{
      "meal_type": "Breakfast",
      "meal_name": "Greek Yogurt Parfait with Berries",
      "description": "Creamy Greek yogurt layered with mixed berries and granola",
      "ingredients_to_validate": [
        "200g Greek yogurt",
        "100g mixed berries",
        "40g granola",
        "15g honey"
      ],
      "estimated_calories": 450,
      "estimated_protein": 25.0,
      "estimated_carbs": 55.0,
      "estimated_fat": 12.0,
      "preparation_time_min": 5,
      "recipe_notes": "Layer yogurt and berries in a bowl. Top with granola. Drizzle honey."
    }}
    // ... {meal_count - 1} more meals for: {meal_types_str}
  ]
}}
```

HEXIS MEAL TARGETS - CREATE ONE MEAL FOR EACH (exactly {meal_count} meals):
{meal_targets_section}

CARBCODE GUIDANCE:
- LOW: Prioritize proteins (chicken, fish, eggs), healthy fats (avocado, nuts, olive oil),
       low-GI vegetables (leafy greens, broccoli, zucchini)
- MEDIUM: Balanced carbs from whole grains (quinoa, brown rice), fruits, starchy vegetables
- HIGH: High-carb recovery foods (pasta, white rice, bread), sports nutrition, fruits

INGREDIENT NAMING RULES (CRITICAL for Executor validation):
- Use SIMPLE, COMMON ingredient names
- Include quantities in grams when possible
- Examples of GOOD names: "chicken breast", "brown rice", "olive oil", "eggs"
- Examples of BAD names: "free-range chicken", "artisanal bread", "specialty cheese"
- The Executor will search the Passio database for these exact terms

MACRO ESTIMATION:
- Match each meal's macros to the Hexis targets as closely as possible
- Aim for each meal within ±10% of its Hexis target
- Daily totals should be within ±5% of the day's target
- The Reviewer will verify against Hexis targets

VARIETY RULES:
- Do not repeat meals from EXISTING DAYS
- Rotate protein sources, cuisines, and cooking methods
- Use seasonal ingredients when possible
- Prioritize FROZEN vegetables for convenience

PRACTICAL CONSTRAINTS:
- Breakfast prep: <15 minutes
- Snack prep: <10 minutes
- Lunch/Dinner prep: <45 minutes
- Use ingredients from French supermarkets (Leclerc, Carrefour)
- Never use smoked salmon (user preference)

Return ONLY valid JSON without markdown fences or commentary.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"Valid JSON conforming to MealPlanTemplate schema with exactly {meal_count} meals ({meal_types_str}) and estimated macros matching Hexis targets",
    )
