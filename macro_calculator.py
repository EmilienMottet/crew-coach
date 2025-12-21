"""
Pre-calculate macros in Python to reduce LLM processing time.

This module handles the calculation of nutritional macros from validated
ingredient data, which was previously done by the MEAL_RECIPE_REVIEWER LLM.

By doing these calculations in Python:
- Reduces tokens in the Reviewer prompt by ~50%
- Ensures accurate calculations (no LLM math errors)
- Speeds up the overall meal planning process
"""

from typing import Any, Callable, Dict, List, Optional
import sys
import time

from mcp_tool_wrapper import get_trusted_nutrition, TRUSTED_INGREDIENTS
from retry_utils import (
    exponential_backoff_delay,
    get_passio_circuit_breaker,
    is_retriable_error,
    DEFAULT_MAX_RETRIES,
)


# Fallback nutrition estimates by food category (per 100g)
# Used when Passio API fails to provide data
FALLBACK_NUTRITION_ESTIMATES = {
    # Proteins
    "chicken": {"protein": 31, "carbs": 0, "fat": 3.6, "calories": 165},
    "beef": {"protein": 26, "carbs": 0, "fat": 15, "calories": 250},
    "pork": {"protein": 25, "carbs": 0, "fat": 14, "calories": 242},
    "fish": {"protein": 20, "carbs": 0, "fat": 5, "calories": 130},
    "salmon": {"protein": 20, "carbs": 0, "fat": 13, "calories": 208},
    "tuna": {"protein": 29, "carbs": 0, "fat": 1, "calories": 130},
    "shrimp": {"protein": 24, "carbs": 0, "fat": 0.3, "calories": 99},
    "egg": {"protein": 13, "carbs": 1, "fat": 11, "calories": 155},
    "tofu": {"protein": 8, "carbs": 2, "fat": 4, "calories": 76},
    # Dairy
    "milk": {"protein": 3.4, "carbs": 5, "fat": 3.3, "calories": 61},
    "yogurt": {"protein": 10, "carbs": 4, "fat": 0.7, "calories": 59},
    "cheese": {"protein": 25, "carbs": 1, "fat": 33, "calories": 402},
    "butter": {"protein": 0.9, "carbs": 0, "fat": 81, "calories": 717},
    "cream": {"protein": 2.1, "carbs": 2.8, "fat": 36, "calories": 340},
    # Grains
    "rice": {"protein": 2.7, "carbs": 28, "fat": 0.3, "calories": 130},
    "pasta": {"protein": 5, "carbs": 31, "fat": 0.9, "calories": 157},
    "bread": {"protein": 9, "carbs": 49, "fat": 3.2, "calories": 265},
    "oat": {"protein": 13, "carbs": 66, "fat": 7, "calories": 389},
    "quinoa": {"protein": 4.4, "carbs": 21, "fat": 1.9, "calories": 120},
    # Vegetables
    "potato": {"protein": 2, "carbs": 17, "fat": 0.1, "calories": 77},
    "sweet potato": {"protein": 1.6, "carbs": 20, "fat": 0.1, "calories": 86},
    "broccoli": {"protein": 2.8, "carbs": 7, "fat": 0.4, "calories": 34},
    "spinach": {"protein": 2.9, "carbs": 3.6, "fat": 0.4, "calories": 23},
    "carrot": {"protein": 0.9, "carbs": 10, "fat": 0.2, "calories": 41},
    "tomato": {"protein": 0.9, "carbs": 3.9, "fat": 0.2, "calories": 18},
    "onion": {"protein": 1.1, "carbs": 9, "fat": 0.1, "calories": 40},
    "pepper": {"protein": 1, "carbs": 6, "fat": 0.3, "calories": 31},
    "zucchini": {"protein": 1.2, "carbs": 3.1, "fat": 0.3, "calories": 17},
    "cucumber": {"protein": 0.7, "carbs": 3.6, "fat": 0.1, "calories": 16},
    "lettuce": {"protein": 1.4, "carbs": 2.9, "fat": 0.2, "calories": 15},
    # Fruits
    "banana": {"protein": 1.1, "carbs": 23, "fat": 0.3, "calories": 89},
    "apple": {"protein": 0.3, "carbs": 14, "fat": 0.2, "calories": 52},
    "orange": {"protein": 0.9, "carbs": 12, "fat": 0.1, "calories": 47},
    "berry": {"protein": 0.7, "carbs": 14, "fat": 0.3, "calories": 57},
    "mango": {"protein": 0.8, "carbs": 15, "fat": 0.4, "calories": 60},
    "avocado": {"protein": 2, "carbs": 9, "fat": 15, "calories": 160},
    # Nuts & Seeds
    "almond": {"protein": 21, "carbs": 22, "fat": 49, "calories": 579},
    "walnut": {"protein": 15, "carbs": 14, "fat": 65, "calories": 654},
    "peanut": {"protein": 26, "carbs": 16, "fat": 49, "calories": 567},
    "cashew": {"protein": 18, "carbs": 30, "fat": 44, "calories": 553},
    "seed": {"protein": 18, "carbs": 28, "fat": 42, "calories": 534},
    # Legumes
    "lentil": {"protein": 9, "carbs": 20, "fat": 0.4, "calories": 116},
    "bean": {"protein": 8.7, "carbs": 22, "fat": 0.5, "calories": 127},
    "chickpea": {"protein": 8.9, "carbs": 27, "fat": 2.6, "calories": 164},
    # Oils & Fats
    "oil": {"protein": 0, "carbs": 0, "fat": 100, "calories": 884},
    "olive": {"protein": 0, "carbs": 0, "fat": 100, "calories": 884},
    # Sweeteners
    "honey": {"protein": 0.3, "carbs": 82, "fat": 0, "calories": 304},
    "sugar": {"protein": 0, "carbs": 100, "fat": 0, "calories": 387},
    "syrup": {"protein": 0, "carbs": 67, "fat": 0.1, "calories": 260},
    # Default for unrecognized items
    "default": {"protein": 5, "carbs": 15, "fat": 5, "calories": 125},
}


def estimate_nutrition_from_name(ingredient_name: str) -> Dict[str, float]:
    """Estimate nutrition values based on ingredient name when API fails.

    Args:
        ingredient_name: Name of the ingredient

    Returns:
        Dict with protein_per_100g, carbs_per_100g, fat_per_100g, calories_per_100g
    """
    name_lower = ingredient_name.lower()

    # Find best matching category
    best_match = None
    best_match_len = 0

    for category in FALLBACK_NUTRITION_ESTIMATES:
        if category in name_lower and len(category) > best_match_len:
            best_match = category
            best_match_len = len(category)

    # Use matched category or default
    estimates = FALLBACK_NUTRITION_ESTIMATES.get(
        best_match or "default",
        FALLBACK_NUTRITION_ESTIMATES["default"]
    )

    return {
        "protein_per_100g": estimates["protein"],
        "carbs_per_100g": estimates["carbs"],
        "fat_per_100g": estimates["fat"],
        "calories_per_100g": estimates["calories"],
        "estimated": True,
        "estimate_source": best_match or "default",
    }


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


def _is_valid_passio_ref_code(ref_code: str) -> bool:
    """Check if a ref_code looks like a valid Passio refCode.

    Real Passio refCodes decode to JSON with keys like:
    - {"labelid": "...", "type": "synonym", "resultid": "...", "metadata": null}

    LLM-hallucinated refCodes decode to fake structures like:
    - {"food_id": "eggs_001"}

    Args:
        ref_code: Base64-encoded string to validate

    Returns:
        True if it looks like a real Passio refCode
    """
    import base64
    import json

    if not ref_code or not isinstance(ref_code, str):
        return False

    try:
        decoded = base64.b64decode(ref_code).decode("utf-8")
        parsed = json.loads(decoded)

        # Real Passio refCodes have "labelid" or "resultid" keys
        if isinstance(parsed, dict):
            has_real_keys = "labelid" in parsed or "resultid" in parsed or "type" in parsed
            # Hallucinated ones have "food_id" key
            has_fake_keys = "food_id" in parsed
            return has_real_keys and not has_fake_keys
    except Exception:
        pass

    return False


def _extract_ingredient_query(ingredient_name: str) -> str:
    """Extract a simple search query from an ingredient name.

    Removes quantities and units to get just the food name.
    Example: "200g chicken breast" → "chicken breast"

    Args:
        ingredient_name: Full ingredient string with quantity

    Returns:
        Simplified search query
    """
    import re

    # Remove common quantity patterns
    # "200g X", "200 g X", "2 cups X", "1/2 cup X", "100ml X"
    cleaned = re.sub(r"^\d+(?:\.\d+)?(?:\s*(?:g|kg|ml|l|oz|cup|cups|tbsp|tsp|lb|lbs))?\s*", "", ingredient_name, flags=re.IGNORECASE)

    # Also remove trailing "cooked", "raw", "fresh", etc. modifiers
    cleaned = re.sub(r"\s*,?\s*(?:cooked|raw|fresh|frozen|dried|canned)\s*$", "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip().lower()


def fetch_nutrition_for_ingredients(
    validated_ingredients_data: Dict[str, Any],
    tool_caller: Callable[[str], Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Fetch nutritional data for ingredients missing protein_per_100g.

    This function automatically fetches nutrition data from the Passio API
    for ingredients that have a passio_ref_code but are missing nutrition values.
    This removes the dependency on the LLM calling hexis_get_passio_food_details.

    IMPORTANT: LLMs often hallucinate fake ref_codes like {"food_id": "eggs_001"}.
    This function validates ref_codes and falls back to the search cache to get
    real Passio ref_codes when the LLM provides invalid ones.

    Uses PassioNutritionCache to avoid redundant API calls.

    Args:
        validated_ingredients_data: ValidatedIngredientsList dict from Executor
        tool_caller: Function that takes ref_code and returns nutrition dict
                    Expected signature: (ref_code: str) -> Optional[Dict]
                    Should return: {"protein": float, "carbs": float, "fat": float, "calories": float}

    Returns:
        Same dict with nutrition data populated
    """
    from passio_nutrition_cache import get_passio_cache, get_passio_search_cache

    nutrition_cache = get_passio_cache()
    search_cache = get_passio_search_cache()
    fetch_count = 0
    cache_hit_count = 0
    search_cache_fallback_count = 0
    estimate_fallback_count = 0

    validated_meals = validated_ingredients_data.get("validated_meals", [])

    trusted_hit_count = 0

    for meal in validated_meals:
        for ing in meal.get("validated_ingredients", []):
            # Skip if already has valid nutrition data
            if ing.get("protein_per_100g") is not None and ing.get("protein_per_100g") > 0:
                continue

            ingredient_name = ing.get("name", "")

            # PRIORITY 1: Check TRUSTED_INGREDIENTS first (verified data, no API needed)
            query = _extract_ingredient_query(ingredient_name)
            if query:
                trusted = get_trusted_nutrition(query)
                if trusted:
                    ing["protein_per_100g"] = trusted.get("protein_per_100g", 0)
                    ing["carbs_per_100g"] = trusted.get("carbs_per_100g", 0)
                    ing["fat_per_100g"] = trusted.get("fat_per_100g", 0)
                    ing["calories_per_100g"] = trusted.get("calories_per_100g", 0)
                    ing["data_source"] = "TRUSTED"
                    trusted_hit_count += 1
                    print(f"   🔒 Using TRUSTED data for '{ingredient_name}'", file=sys.stderr)
                    continue

            # Get ref_code from LLM output
            ref_code = ing.get("passio_ref_code")

            # Validate the ref_code - LLMs often hallucinate fake ones
            if ref_code and not _is_valid_passio_ref_code(ref_code):
                print(f"   ⚠️ Invalid ref_code for '{ingredient_name}', checking search cache...", file=sys.stderr)
                ref_code = None  # Force fallback to search cache

            # Fallback: Look up search cache by ingredient name to get real ref_code
            if not ref_code and ingredient_name:
                query = _extract_ingredient_query(ingredient_name)
                if query:
                    cached_search = search_cache.get(query)
                    if cached_search and cached_search.get("passio_ref_code"):
                        ref_code = cached_search["passio_ref_code"]
                        # Also get nutrition directly from search cache if available
                        if cached_search.get("protein_per_100g") is not None:
                            ing["protein_per_100g"] = cached_search.get("protein_per_100g", 0)
                            ing["carbs_per_100g"] = cached_search.get("carbs_per_100g", 0)
                            ing["fat_per_100g"] = cached_search.get("fat_per_100g", 0)
                            ing["calories_per_100g"] = cached_search.get("calories_per_100g", 0)
                            # Update the ingredient with the correct ref_code
                            ing["passio_ref_code"] = ref_code
                            search_cache_fallback_count += 1
                            continue
                        # Update the ingredient with the correct ref_code
                        ing["passio_ref_code"] = ref_code
                        print(f"   ✅ Found real ref_code for '{query}' in search cache", file=sys.stderr)
                        search_cache_fallback_count += 1

            # If no valid ref_code, use fallback estimation
            if not ref_code:
                fallback = estimate_nutrition_from_name(ingredient_name)
                ing["protein_per_100g"] = fallback["protein_per_100g"]
                ing["carbs_per_100g"] = fallback["carbs_per_100g"]
                ing["fat_per_100g"] = fallback["fat_per_100g"]
                ing["calories_per_100g"] = fallback["calories_per_100g"]
                ing["nutrition_estimated"] = True
                ing["estimate_source"] = fallback.get("estimate_source", "default")
                estimate_fallback_count += 1
                print(f"   📊 No ref_code for '{ingredient_name}', using fallback (source: {ing['estimate_source']})", file=sys.stderr)
                continue

            # Try nutrition cache first
            cached = nutrition_cache.get(ref_code)
            if cached:
                # Check if cache has real values (not 0s from failed fetches)
                has_valid_nutrition = (
                    cached.get("protein_per_100g", 0) > 0 or
                    cached.get("carbs_per_100g", 0) > 0 or
                    cached.get("fat_per_100g", 0) > 0
                )
                if has_valid_nutrition:
                    ing["protein_per_100g"] = cached.get("protein_per_100g", 0)
                    ing["carbs_per_100g"] = cached.get("carbs_per_100g", 0)
                    ing["fat_per_100g"] = cached.get("fat_per_100g", 0)
                    ing["calories_per_100g"] = cached.get("calories_per_100g", 0)
                    cache_hit_count += 1
                    continue
                else:
                    # Invalid cache entry (all zeros) - invalidate and refetch
                    print(f"   ⚠️ Invalidating zero-value cache for '{ingredient_name}'", file=sys.stderr)
                    nutrition_cache.delete(ref_code)

            # Fetch from API with exponential backoff
            # Add small delay between API calls to avoid rate limiting
            # Passio API seems to have undocumented rate limits (~10-15 req/min)
            if fetch_count > 0:
                time.sleep(0.3)  # 300ms delay between calls

            circuit_breaker = get_passio_circuit_breaker()
            max_retries = DEFAULT_MAX_RETRIES

            for attempt in range(max_retries + 1):
                # Check circuit breaker
                if not circuit_breaker.can_execute():
                    print(f"   ⚡ Circuit breaker open, skipping nutrition fetch for '{ingredient_name}'", file=sys.stderr)
                    break

                try:
                    nutrition = tool_caller(ref_code)

                    # Check for API error in response
                    if isinstance(nutrition, dict) and nutrition.get("error"):
                        error_msg = nutrition.get("error", "")
                        if "500" in str(error_msg) or "error while searching" in str(error_msg).lower():
                            raise RuntimeError(f"Passio API error: {error_msg}")

                    if nutrition:
                        # Extract nutrition values (API may return "protein" or "protein_per_100g")
                        protein = nutrition.get("protein") or nutrition.get("protein_per_100g", 0)
                        carbs = nutrition.get("carbs") or nutrition.get("carbs_per_100g", 0)
                        fat = nutrition.get("fat") or nutrition.get("fat_per_100g", 0)
                        calories = nutrition.get("calories") or nutrition.get("calories_per_100g", 0)

                        ing["protein_per_100g"] = float(protein) if protein else 0.0
                        ing["carbs_per_100g"] = float(carbs) if carbs else 0.0
                        ing["fat_per_100g"] = float(fat) if fat else 0.0
                        ing["calories_per_100g"] = float(calories) if calories else 0.0

                        # Only cache if we got real values (avoid caching 0s from failed API calls)
                        if ing["protein_per_100g"] > 0 or ing["carbs_per_100g"] > 0 or ing["fat_per_100g"] > 0:
                            nutrition_cache.set(ref_code, {
                                "protein_per_100g": ing["protein_per_100g"],
                                "carbs_per_100g": ing["carbs_per_100g"],
                                "fat_per_100g": ing["fat_per_100g"],
                                "calories_per_100g": ing["calories_per_100g"],
                            })
                            # Also update search cache with nutrition data
                            # This allows future runs to skip the API call entirely
                            query = _extract_ingredient_query(ingredient_name)
                            if query:
                                search_cache.set(query, {
                                    "passio_food_id": ing.get("passio_food_id"),
                                    "passio_ref_code": ref_code,
                                    "passio_food_name": ing.get("passio_food_name"),
                                    "protein_per_100g": ing["protein_per_100g"],
                                    "carbs_per_100g": ing["carbs_per_100g"],
                                    "fat_per_100g": ing["fat_per_100g"],
                                    "calories_per_100g": ing["calories_per_100g"],
                                })
                        fetch_count += 1

                    # Success - reset circuit breaker
                    circuit_breaker.record_success()
                    break

                except Exception as e:
                    circuit_breaker.record_failure()

                    if attempt >= max_retries or not is_retriable_error(e):
                        print(f"   ⚠️ Failed to fetch nutrition for {ingredient_name} after {attempt + 1} attempts: {e}", file=sys.stderr)
                        # Use fallback estimation when API fails
                        fallback = estimate_nutrition_from_name(ingredient_name)
                        ing["protein_per_100g"] = fallback["protein_per_100g"]
                        ing["carbs_per_100g"] = fallback["carbs_per_100g"]
                        ing["fat_per_100g"] = fallback["fat_per_100g"]
                        ing["calories_per_100g"] = fallback["calories_per_100g"]
                        ing["nutrition_estimated"] = True
                        ing["estimate_source"] = fallback.get("estimate_source", "default")
                        estimate_fallback_count += 1
                        print(f"   📊 Using fallback estimate for '{ingredient_name}' (source: {ing['estimate_source']})", file=sys.stderr)
                        break

                    delay = exponential_backoff_delay(attempt)
                    print(f"   ⏳ Nutrition fetch retry {attempt + 1}/{max_retries} for '{ingredient_name}' after {delay:.1f}s", file=sys.stderr)
                    time.sleep(delay)

    if fetch_count > 0 or cache_hit_count > 0 or search_cache_fallback_count > 0 or estimate_fallback_count > 0 or trusted_hit_count > 0:
        print(
            f"   🥗 Nutrition fetch: {trusted_hit_count} trusted, {fetch_count} API calls, {cache_hit_count} cache hits, "
            f"{search_cache_fallback_count} search cache fallbacks, {estimate_fallback_count} estimated",
            file=sys.stderr,
        )

    return validated_ingredients_data
