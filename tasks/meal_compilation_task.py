"""Task for consolidating daily plans into a weekly meal plan."""
from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from crewai import Task


def create_meal_compilation_task(
    agent: Any,
    nutrition_plan: Dict[str, Any],
    daily_plans: Sequence[Dict[str, Any]],
) -> Task:
    """Create a task that merges daily plans into a weekly structure."""
    nutrition_json = json.dumps(nutrition_plan, indent=2)
    daily_plans_json = json.dumps(list(daily_plans), indent=2)

    description = f"""
Merge the provided daily meal plans into a cohesive weekly plan without changing the meals.

NUTRITION STRATEGY (reference only, do not regenerate meals):
{nutrition_json}

DAILY MEAL PLANS TO CONSOLIDATE:
{daily_plans_json}

REQUIRED OUTPUT (WeeklyMealPlan schema):
- Keep the day order identical to the input.
- Copy meals exactly as provided. Never rename or modify ingredients, macros, or notes.
- For each day, ensure daily_totals reflect the sum of the four meals. Recalculate if necessary.
- Build a shopping_list array where each entry follows "quantity unit ingredient" format. Aggregate identical ingredients and sum quantities, preferring metric units.
- Provide at least six meal_prep_tips highlighting batch cooking, ingredient reuse, storage, and timing cues throughout the week.

QUALITY RULES:
- Group shopping list items by category labels in parentheses (for example "Produce - 6 bananas").
- Highlight any opportunities to reuse sauces or prepped components.
- Note when leftovers can be repurposed safely.
- Mention make-ahead steps for busy training days.

Return only JSON, no markdown fences. The result must validate against WeeklyMealPlan.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the WeeklyMealPlan schema with consolidated shopping list and prep tips",
    )
