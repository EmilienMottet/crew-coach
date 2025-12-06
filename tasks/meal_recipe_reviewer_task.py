"""Task for the Meal Recipe Reviewer - macro calculation and final assembly."""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task


def create_meal_recipe_reviewer_task(
    agent: Any,
    meal_plan_template: Dict[str, Any],
    validated_ingredients: Dict[str, Any],
    daily_target: Dict[str, Any],
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

   IMPORTANT: DO NOT USE HARDCODED VALUES!
   Use the protein_per_100g, carbs_per_100g, fat_per_100g, calories_per_100g
   fields from each validated ingredient. These are real Passio database values.

   For each meal, sum all ingredients' macros.
   For daily totals, sum all meals' macros

2. VALIDATE AGAINST TARGETS (STRICT - YOU MUST MEET THESE)
   - Calories: {target_calories} kcal (MUST be within ±10% = {int(target_calories * 0.90)}-{int(target_calories * 1.10)})
   - Protein: {target_protein}g (MUST be within ±5% = {int(target_protein * 0.95)}-{int(target_protein * 1.05)}g)
   - Carbs: {target_carbs}g (MUST be within ±5% = {int(target_carbs * 0.95)}-{int(target_carbs * 1.05)}g)
   - Fat: {target_fat}g (MUST be within ±5% = {int(target_fat * 0.95)}-{int(target_fat * 1.05)}g)

   ⚠️ If your calculations show totals OUTSIDE these ranges, you MUST adjust portions!

   **ALGORITHME D'AJUSTEMENT OBLIGATOIRE - SUIVRE CES ETAPES:**

   ÉTAPE A: Calculer les écarts
   - delta_protein = calculated_protein - {target_protein}
   - delta_carbs = calculated_carbs - {target_carbs}
   - delta_fat = calculated_fat - {target_fat}

   ÉTAPE B: Si delta > 5% de la cible, AJUSTER:

   ▼ PROTÉINES TROP ÉLEVÉES (delta_protein > 0):
   Choisir UN ingrédient riche en protéines et RÉDUIRE:
   - Poulet: réduire de delta_protein ÷ 0.31 grammes
     Ex: -13g protein → réduire poulet de 42g (13÷0.31=42)
   - Yaourt grec: réduire de delta_protein ÷ 0.10 grammes
     Ex: -13g protein → réduire yaourt de 130g (13÷0.10=130)
   - Saumon: réduire de delta_protein ÷ 0.20 grammes
   - Œufs: retirer delta_protein ÷ 6.5 œufs (1 œuf = 6.5g protein)

   ▼ PROTÉINES TROP BASSES (delta_protein < 0):
   Ajouter ingrédients riches en protéines (voir section TO ADD PROTEIN)

   ▼ GLUCIDES TROP ÉLEVÉS (delta_carbs > 0):
   - Riz cuit: réduire de delta_carbs ÷ 0.23 grammes
   - Pâtes cuites: réduire de delta_carbs ÷ 0.25 grammes
   - Banane: réduire de delta_carbs ÷ 0.23 grammes
   - Pain: réduire de delta_carbs ÷ 0.49 grammes

   ▼ LIPIDES TROP ÉLEVÉS (delta_fat > 0):
   - Huile d'olive: réduire de delta_fat ÷ 1.0 grammes
   - Avocat: réduire de delta_fat ÷ 0.15 grammes
   - Amandes: réduire de delta_fat ÷ 0.49 grammes

   ÉTAPE C: RECALCULER après ajustement
   - Refaire les calculs de macros pour TOUS les repas
   - Vérifier que les totaux sont maintenant dans ±5%
   - Si toujours hors cible, répéter l'ajustement

   ÉTAPE D: METTRE À JOUR les quantités dans l'output
   - Les quantités finales dans "ingredients" doivent refléter les ajustements
   - Ex: si poulet réduit de 120g à 78g, écrire "78g chicken breast"

   EXEMPLE CONCRET:
   - Cible protein: 153g
   - Calcul initial: 166g protein (+13g = +8.5%)
   - Trouvé: "120g chicken breast" au Lunch
   - Calcul: 13g ÷ 0.31 = 42g à retirer
   - Nouvelle quantité: 120g - 42g = 78g chicken breast
   - Nouveau total: 166g - 13g = 153g ✓

3. ADJUST PORTIONS TO MEET TARGETS (MANDATORY if outside tolerance)
   Use these conversion rates to fix deficits:

   TO ADD CARBS:
   - +30g carbs → add 80g dry pasta (becomes ~200g cooked)
   - +30g carbs → add 77g dry rice (becomes ~200g cooked)
   - +25g carbs → add 100g bread

   TO ADD PROTEIN:
   - +20g protein → add 65g chicken breast
   - +20g protein → add 100g salmon
   - +13g protein → add 2 eggs

   TO ADD CALORIES (when carbs/protein are OK):
   - +100 kcal → add 11g olive oil (1 tbsp)
   - +100 kcal → add 15g nuts

   TO REDUCE CALORIES/FAT:
   - Remove or reduce oil/butter portions
   - Use cooking spray instead of oil
   - Reduce cheese portions

   ALWAYS document your adjustments in the "notes" field!

4. ASSEMBLE FINAL OUTPUT
   Create DailyMealPlan JSON with:
   - Calculated (not estimated) macro values
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
    // ... 3 more meals (Lunch, Afternoon Snack, Dinner)
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

IMPORTANT REMINDERS:
- Use CALCULATED macros, not just copy Supervisor's estimates
- Include validated_ingredients from Executor (with passio_food_id)
- Ensure daily_totals match sum of all meals
- Add notes if any adjustments were made
- Return ONLY valid JSON without markdown fences

Return ONLY valid JSON without markdown fences or commentary.
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON conforming to DailyMealPlan schema with calculated macros and validated_ingredients",
    )
