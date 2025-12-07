"""
Pre-calculate macros in Python to reduce LLM processing time.

This module handles the calculation of nutritional macros from validated
ingredient data, which was previously done by the MEAL_RECIPE_REVIEWER LLM.

By doing these calculations in Python:
- Reduces tokens in the Reviewer prompt by ~50%
- Ensures accurate calculations (no LLM math errors)
- Speeds up the overall meal planning process
"""

from typing import Any, Dict, List, Optional
import sys


def calculate_ingredient_macros(ingredient: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate actual macros for an ingredient based on quantity and per-100g values.

    Formula: macros = (quantity_g / 100) × macros_per_100g

    Args:
        ingredient: Dict with quantity_g and *_per_100g fields

    Returns:
        Same dict with added protein_g, carbs_g, fat_g, calories fields
    """
    quantity = ingredient.get("quantity_g", 0) or 0

    # Handle both direct fields and nested adjusted_quantity_g
    if ingredient.get("adjusted_quantity_g"):
        quantity = ingredient["adjusted_quantity_g"]

    # Get per-100g values (default to 0 if missing)
    protein_per_100g = ingredient.get("protein_per_100g", 0) or 0
    carbs_per_100g = ingredient.get("carbs_per_100g", 0) or 0
    fat_per_100g = ingredient.get("fat_per_100g", 0) or 0
    calories_per_100g = ingredient.get("calories_per_100g", 0) or 0

    # Calculate actual values
    factor = quantity / 100.0 if quantity > 0 else 0
    ingredient["protein_g"] = round(protein_per_100g * factor, 1)
    ingredient["carbs_g"] = round(carbs_per_100g * factor, 1)
    ingredient["fat_g"] = round(fat_per_100g * factor, 1)
    ingredient["calories"] = round(calories_per_100g * factor)

    return ingredient


def calculate_meal_totals(validated_ingredients: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate total macros for a meal from its validated ingredients.

    Args:
        validated_ingredients: List of ingredient dicts with calculated macros

    Returns:
        Dict with total protein_g, carbs_g, fat_g, calories
    """
    totals = {
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
        "calories": 0,
    }

    for ing in validated_ingredients:
        totals["protein_g"] += ing.get("protein_g", 0) or 0
        totals["carbs_g"] += ing.get("carbs_g", 0) or 0
        totals["fat_g"] += ing.get("fat_g", 0) or 0
        totals["calories"] += ing.get("calories", 0) or 0

    # Round the totals
    totals["protein_g"] = round(totals["protein_g"], 1)
    totals["carbs_g"] = round(totals["carbs_g"], 1)
    totals["fat_g"] = round(totals["fat_g"], 1)
    totals["calories"] = round(totals["calories"])

    return totals


def calculate_daily_totals(meals: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate total macros for a day from all meals.

    Args:
        meals: List of meal dicts, each with protein_g, carbs_g, fat_g, calories

    Returns:
        Dict with daily totals
    """
    totals = {
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
        "calories": 0,
    }

    for meal in meals:
        totals["protein_g"] += meal.get("protein_g", 0) or 0
        totals["carbs_g"] += meal.get("carbs_g", 0) or 0
        totals["fat_g"] += meal.get("fat_g", 0) or 0
        totals["calories"] += meal.get("calories", 0) or 0

    # Round the totals
    totals["protein_g"] = round(totals["protein_g"], 1)
    totals["carbs_g"] = round(totals["carbs_g"], 1)
    totals["fat_g"] = round(totals["fat_g"], 1)
    totals["calories"] = round(totals["calories"])

    return totals


def enrich_validated_ingredients(
    validated_ingredients_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Enrich ValidatedIngredientsList with pre-calculated macros.

    Processes the Executor output and adds calculated macros to each ingredient,
    plus totals for each meal.

    Args:
        validated_ingredients_data: ValidatedIngredientsList dict from Executor

    Returns:
        Same dict with added macro calculations
    """
    validated_meals = validated_ingredients_data.get("validated_meals", [])

    for meal in validated_meals:
        ingredients = meal.get("validated_ingredients", [])

        # Calculate macros for each ingredient
        for ing in ingredients:
            calculate_ingredient_macros(ing)

        # Calculate meal totals
        meal_totals = calculate_meal_totals(ingredients)
        meal["calculated_totals"] = meal_totals

    # Calculate overall totals
    all_meal_totals = [m.get("calculated_totals", {}) for m in validated_meals]
    validated_ingredients_data["calculated_daily_totals"] = calculate_daily_totals(
        all_meal_totals
    )

    return validated_ingredients_data


def check_macro_compliance(
    calculated_totals: Dict[str, float],
    target_totals: Dict[str, float],
    tolerance_percent: float = 10.0,
) -> Dict[str, Any]:
    """Check if calculated macros are within tolerance of targets.

    Args:
        calculated_totals: Dict with calculated protein_g, carbs_g, fat_g, calories
        target_totals: Dict with target values
        tolerance_percent: Acceptable deviation (default 10%)

    Returns:
        Dict with compliance status and deltas
    """
    results = {
        "compliant": True,
        "deltas": {},
        "out_of_range": [],
    }

    macro_names = ["protein_g", "carbs_g", "fat_g", "calories"]

    for macro in macro_names:
        calculated = calculated_totals.get(macro, 0) or 0
        target = target_totals.get(macro, 0) or 0

        if target == 0:
            continue

        delta = calculated - target
        delta_percent = (delta / target) * 100 if target > 0 else 0

        results["deltas"][macro] = {
            "calculated": calculated,
            "target": target,
            "delta": round(delta, 1),
            "delta_percent": round(delta_percent, 1),
        }

        if abs(delta_percent) > tolerance_percent:
            results["compliant"] = False
            results["out_of_range"].append(macro)

    return results


def suggest_adjustments(
    compliance_result: Dict[str, Any],
    validated_meals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Suggest ingredient adjustments to meet macro targets.

    Uses standard conversion rates to recommend portion changes.

    Args:
        compliance_result: Output from check_macro_compliance
        validated_meals: List of meals with validated_ingredients

    Returns:
        List of suggested adjustments
    """
    suggestions = []

    for macro in compliance_result.get("out_of_range", []):
        delta_info = compliance_result["deltas"].get(macro, {})
        delta = delta_info.get("delta", 0)

        if macro == "protein_g":
            if delta > 0:
                # Too much protein - suggest reduction
                suggestions.append({
                    "macro": "protein",
                    "action": "reduce",
                    "delta_g": abs(delta),
                    "suggestion": f"Reduce chicken by {abs(delta) / 0.31:.0f}g or eggs by {abs(delta) / 6.5:.1f}",
                })
            else:
                # Not enough protein - suggest addition
                suggestions.append({
                    "macro": "protein",
                    "action": "increase",
                    "delta_g": abs(delta),
                    "suggestion": f"Add {abs(delta) / 0.31:.0f}g chicken breast or {abs(delta) / 0.20:.0f}g salmon",
                })

        elif macro == "carbs_g":
            if delta > 0:
                suggestions.append({
                    "macro": "carbs",
                    "action": "reduce",
                    "delta_g": abs(delta),
                    "suggestion": f"Reduce rice by {abs(delta) / 0.23:.0f}g or pasta by {abs(delta) / 0.25:.0f}g",
                })
            else:
                suggestions.append({
                    "macro": "carbs",
                    "action": "increase",
                    "delta_g": abs(delta),
                    "suggestion": f"Add {abs(delta) / 0.23:.0f}g cooked rice or {abs(delta) / 0.49:.0f}g bread",
                })

        elif macro == "fat_g":
            if delta > 0:
                suggestions.append({
                    "macro": "fat",
                    "action": "reduce",
                    "delta_g": abs(delta),
                    "suggestion": f"Reduce olive oil by {abs(delta):.0f}g or avocado by {abs(delta) / 0.15:.0f}g",
                })
            else:
                suggestions.append({
                    "macro": "fat",
                    "action": "increase",
                    "delta_g": abs(delta),
                    "suggestion": f"Add {abs(delta):.0f}g olive oil or {abs(delta) / 0.49:.0f}g almonds",
                })

        elif macro == "calories":
            if delta > 0:
                suggestions.append({
                    "macro": "calories",
                    "action": "reduce",
                    "delta_g": abs(delta),
                    "suggestion": f"Reduce portion sizes or remove {abs(delta) / 9:.0f}g of fats",
                })
            else:
                suggestions.append({
                    "macro": "calories",
                    "action": "increase",
                    "delta_g": abs(delta),
                    "suggestion": f"Add {abs(delta) / 9:.0f}g olive oil or {abs(delta) / 6:.0f}g nuts",
                })

    return suggestions


def format_pre_calculated_summary(
    validated_ingredients_data: Dict[str, Any],
    target_totals: Dict[str, float],
) -> str:
    """Format a summary of pre-calculated macros for the Reviewer prompt.

    This creates a concise summary that the Reviewer can use directly,
    reducing the amount of calculation needed.

    Args:
        validated_ingredients_data: Enriched ValidatedIngredientsList
        target_totals: Daily macro targets

    Returns:
        Formatted string summary
    """
    calculated = validated_ingredients_data.get("calculated_daily_totals", {})

    lines = [
        "=== PRE-CALCULATED MACRO SUMMARY ===",
        "",
        "CALCULATED DAILY TOTALS (from validated ingredients):",
        f"  - Calories: {calculated.get('calories', 0)} kcal",
        f"  - Protein:  {calculated.get('protein_g', 0):.1f}g",
        f"  - Carbs:    {calculated.get('carbs_g', 0):.1f}g",
        f"  - Fat:      {calculated.get('fat_g', 0):.1f}g",
        "",
        "TARGET TOTALS:",
        f"  - Calories: {target_totals.get('calories', 0)} kcal",
        f"  - Protein:  {target_totals.get('protein_g', 0):.1f}g",
        f"  - Carbs:    {target_totals.get('carbs_g', 0):.1f}g",
        f"  - Fat:      {target_totals.get('fat_g', 0):.1f}g",
        "",
    ]

    # Check compliance
    compliance = check_macro_compliance(calculated, target_totals)

    if compliance["compliant"]:
        lines.append("STATUS: ✅ All macros within ±10% tolerance")
    else:
        lines.append(f"STATUS: ⚠️ Out of range: {', '.join(compliance['out_of_range'])}")
        lines.append("")
        lines.append("DELTAS:")
        for macro, info in compliance["deltas"].items():
            if macro in compliance["out_of_range"]:
                lines.append(
                    f"  - {macro}: {info['delta']:+.1f} ({info['delta_percent']:+.1f}%)"
                )

    lines.append("")
    lines.append("MEAL BREAKDOWN:")

    for meal in validated_ingredients_data.get("validated_meals", []):
        meal_type = meal.get("meal_type", "Unknown")
        totals = meal.get("calculated_totals", {})
        lines.append(
            f"  {meal_type}: {totals.get('calories', 0)} kcal | "
            f"P: {totals.get('protein_g', 0):.1f}g | "
            f"C: {totals.get('carbs_g', 0):.1f}g | "
            f"F: {totals.get('fat_g', 0):.1f}g"
        )

    lines.append("")
    lines.append("=== END SUMMARY ===")

    return "\n".join(lines)
