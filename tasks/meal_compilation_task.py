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
- Build a **smart, categorized shopping list in FRENCH**.
    - **Consolidate**: Sum up quantities for identical items (e.g., "200g poulet" + "300g poulet" = "500g poulet").
    - **Categorize**: Group items by section (Fruits & Légumes, Viande/Poisson, Produits Laitiers, Épicerie, Surgelés, etc.).
    - **Format**: Use the format "Category: Item (Quantity)" (e.g., "Fruits & Légumes: Bananes (6)", "Viande/Poisson: Blanc de poulet (500g)").
    - **Units**: **STRICTLY USE METRIC UNITS** (g, kg, ml, l) or standard French units (c.à.s, c.à.c, tranches). **DO NOT USE** oz, lb, cup, tbsp, tsp. Convert them if necessary.
    - **Pantry Staples**: For common spices/oils, just list the name or "Vérifier stock" unless a large specific amount is needed.
- Provide at least six meal_prep_tips highlighting batch cooking, ingredient reuse, storage, and timing cues throughout the week.

QUALITY RULES:
- The shopping list must be easy to read for a grocery pickup order (Drive).
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
