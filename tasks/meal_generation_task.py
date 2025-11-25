"""Task for generating a single day's meal plan."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable

from crewai import Task


def create_meal_generation_task(
    agent: Any,
    daily_target: Dict[str, Any],
    weekly_context: Dict[str, Any],
    previous_days: Iterable[Dict[str, Any]] | None = None,
    validation_feedback: Dict[str, Any] | None = None,
) -> Task:
    """Create a task for generating meals for a single day."""
    target_json = json.dumps(daily_target, indent=2)
    weekly_json = json.dumps(weekly_context, indent=2)
    previous_days_json = json.dumps(list(previous_days or []), indent=2)

    feedback_section = ""
    if validation_feedback:
        attempt = validation_feedback.get("previous_attempt", 0)
        issues = validation_feedback.get("issues_found", [])
        recommendations = validation_feedback.get("recommendations", [])
        macro_accuracy = validation_feedback.get("macro_accuracy", {})
        summary = validation_feedback.get("validation_summary", "")

        # Format macro accuracy issues
        macro_issues = ""
        if macro_accuracy:
            macro_lines = [f"  - {day}: {status}" for day, status in macro_accuracy.items()]
            macro_issues = "\n".join(macro_lines)

        feedback_section = f"""

⚠️ PRIOR VALIDATION ATTEMPT #{attempt} FAILED - MUST FIX THESE ISSUES

Validation Summary: {summary}

Macro Accuracy Issues per Day:
{macro_issues if macro_issues else "  (no specific day-level issues)"}

Critical Issues to Fix:
{chr(10).join(f"  {idx + 1}. {issue}" for idx, issue in enumerate(issues[:8]))}

MANDATORY Adjustments - Implement ALL of these:
{chr(10).join(f"  ✓ {rec}" for rec in recommendations[:8])}

This is attempt #{attempt + 1}. You MUST fix the protein/macro deficiencies by:
- Increasing portion sizes as recommended above
- Adding protein-rich ingredients where needed
- Recalculating all totals to ensure they meet targets within tolerance
"""

    description = f"""
Plan the meals for a single day while matching nutritional targets exactly.

DAILY TARGET:
{target_json}

WEEKLY CONTEXT (use for variety and strategy):
{weekly_json}

EXISTING DAYS TO AVOID DUPLICATES:
{previous_days_json}
{feedback_section}

MANDATORY OUTPUT STRUCTURE:
- Return valid JSON that fits the DailyMealPlan schema (day_name, date, meals array, daily_totals, notes optional).
- Provide exactly FOUR meals in the listed order: Breakfast, Lunch, Afternoon Snack, Dinner.
- Each meal must include: meal_type, meal_name, description, calories, protein_g, carbs_g, fat_g, preparation_time_min, ingredients (with precise quantities), recipe_notes.

STRICT MACRO ACCURACY:
- Calories within ±3 percent of the target.
- Protein within ±5 g, carbs within ±10 g, fat within ±5 g.
- Balance the day using these reference splits: Breakfast 25 percent, Lunch 35 percent, Dinner 30 to 35 percent, Afternoon Snack 5 to 10 percent.
- Recalculate and adjust portion sizes until totals match the target macros.

VARIETY AND PRACTICALITY RULES:
- Do not repeat meals already present in PREVIOUS DAYS.
- Rotate protein sources, cooking methods, and cuisines across the week.
- Keep preparation realistic (breakfast < 15 min, snack < 10 min, lunch/dinner < 45 min).
- Use whole foods and avoid highly processed items (but FROZEN vegetables/fruits are highly encouraged).
- **INGREDIENT AVAILABILITY (FRANCE):**
    - Use ONLY ingredients found in standard French supermarkets (Leclerc, Carrefour).
    - **Prioritize FROZEN vegetables/fruits (Surgelés) where practical for convenience and cost.**
    - Avoid US-specific items (e.g., "ranch dressing", "Monterey Jack", "collard greens").
    - Use French culinary terms for cuts of meat/fish where appropriate.
- Respect dietary note: never use smoked salmon; substitute with fresh fish options.
- Align timing with training context (reference meal_timing_notes from the target).

QUALITY EXPECTATIONS:
- Ingredients should be listed as "quantity unit ingredient" (for example "120 g chicken breast").
- Description should mention cooking technique and key flavors.
- Recipe notes should contain short, actionable steps (3 to 4 bullet-style sentences).
- Include optional hydration or recovery notes when relevant.

Return only JSON, without markdown fences or commentary.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON that conforms to the DailyMealPlan schema with four meals matching the provided targets",
    )
