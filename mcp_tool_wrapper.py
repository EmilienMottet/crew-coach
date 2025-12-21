"""
Wrapper for MCP tools to validate and fix malformed inputs.

This module provides a defensive layer that intercepts MCP tool calls
and ensures the input format is correct before passing to the actual tool.

The main challenge is that CrewAI validates tool inputs BEFORE calling the tool's
_run() method. To work around this, we create a custom BaseTool wrapper that:
1. Accepts any input type (bypasses CrewAI's schema validation)
2. Validates and fixes the input manually in _run()
3. Delegates to the original MCP tool

Additional optimizations for Passio tools:
- Caches search results to avoid redundant API calls
- Filters response data to reduce token usage
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from passio_nutrition_cache import get_passio_search_cache
from retry_utils import (
    exponential_backoff_delay,
    get_passio_circuit_breaker,
    is_retriable_error,
    DEFAULT_MAX_RETRIES,
)

# =============================================================================
# Passio API Rate Limiter
# =============================================================================
# Passio API has undocumented rate limits that cause 500 "Error while searching"
# when too many requests are made in quick succession.
# This rate limiter adds a minimum delay between consecutive API calls.
PASSIO_MIN_DELAY_SECONDS = float(os.getenv("PASSIO_MIN_DELAY_SECONDS", "0.3"))
_passio_last_call_time: float = 0.0


def _passio_rate_limit():
    """Enforce minimum delay between Passio API calls to avoid rate limiting."""
    global _passio_last_call_time
    now = time.time()
    elapsed = now - _passio_last_call_time
    if elapsed < PASSIO_MIN_DELAY_SECONDS:
        delay = PASSIO_MIN_DELAY_SECONDS - elapsed
        time.sleep(delay)
    _passio_last_call_time = time.time()

# =============================================================================
# Trusted Ingredients Database (Fallback for known-bad Passio entries)
# =============================================================================
# Values per 100g - sourced from USDA and verified nutritional databases
TRUSTED_INGREDIENTS: Dict[str, Dict[str, float]] = {
    # Protein powders (common source of Passio errors)
    "whey protein": {"protein": 80.0, "carbs": 4.0, "fat": 2.0, "calories": 370.0},
    "whey protein powder": {"protein": 80.0, "carbs": 4.0, "fat": 2.0, "calories": 370.0},
    "whey isolate": {"protein": 90.0, "carbs": 2.0, "fat": 1.0, "calories": 380.0},
    "casein protein": {"protein": 80.0, "carbs": 3.0, "fat": 1.0, "calories": 350.0},
    "protein powder": {"protein": 75.0, "carbs": 8.0, "fat": 3.0, "calories": 360.0},
    # Common proteins
    "chicken breast": {"protein": 31.0, "carbs": 0.0, "fat": 3.6, "calories": 165.0},
    "turkey breast": {"protein": 29.0, "carbs": 0.0, "fat": 1.0, "calories": 135.0},
    "salmon": {"protein": 20.0, "carbs": 0.0, "fat": 13.0, "calories": 208.0},
    "tuna": {"protein": 29.0, "carbs": 0.0, "fat": 1.0, "calories": 130.0},
    "eggs": {"protein": 13.0, "carbs": 1.1, "fat": 11.0, "calories": 155.0},
    "egg whites": {"protein": 11.0, "carbs": 0.7, "fat": 0.2, "calories": 52.0},
    # Dairy
    "greek yogurt": {"protein": 10.0, "carbs": 3.6, "fat": 0.7, "calories": 59.0},
    "cottage cheese": {"protein": 11.0, "carbs": 3.4, "fat": 4.3, "calories": 98.0},
    # Grains
    "oats": {"protein": 13.0, "carbs": 66.0, "fat": 7.0, "calories": 389.0},
    "brown rice": {"protein": 2.6, "carbs": 23.0, "fat": 0.9, "calories": 111.0},
    "quinoa": {"protein": 4.4, "carbs": 21.0, "fat": 1.9, "calories": 120.0},
    "basmati rice": {"protein": 3.5, "carbs": 25.0, "fat": 0.4, "calories": 121.0},
    # Fats
    "olive oil": {"protein": 0.0, "carbs": 0.0, "fat": 100.0, "calories": 884.0},
    "avocado": {"protein": 2.0, "carbs": 9.0, "fat": 15.0, "calories": 160.0},
    "almonds": {"protein": 21.0, "carbs": 22.0, "fat": 49.0, "calories": 579.0},
    "walnuts": {"protein": 15.0, "carbs": 14.0, "fat": 65.0, "calories": 654.0},
    # Vegetables (common source of Passio errors - often return inflated carbs)
    "zucchini": {"protein": 1.2, "carbs": 3.1, "fat": 0.3, "calories": 17.0},
    "courgette": {"protein": 1.2, "carbs": 3.1, "fat": 0.3, "calories": 17.0},
    "bell pepper": {"protein": 1.0, "carbs": 6.0, "fat": 0.3, "calories": 31.0},
    "red bell pepper": {"protein": 1.0, "carbs": 6.0, "fat": 0.3, "calories": 31.0},
    "green bell pepper": {"protein": 0.9, "carbs": 4.6, "fat": 0.2, "calories": 20.0},
    "yellow bell pepper": {"protein": 1.0, "carbs": 6.3, "fat": 0.2, "calories": 27.0},
    "broccoli": {"protein": 2.8, "carbs": 7.0, "fat": 0.4, "calories": 34.0},
    "cauliflower": {"protein": 1.9, "carbs": 5.0, "fat": 0.3, "calories": 25.0},
    "spinach": {"protein": 2.9, "carbs": 3.6, "fat": 0.4, "calories": 23.0},
    "kale": {"protein": 4.3, "carbs": 8.8, "fat": 0.9, "calories": 49.0},
    "cucumber": {"protein": 0.7, "carbs": 3.6, "fat": 0.1, "calories": 16.0},
    "tomato": {"protein": 0.9, "carbs": 3.9, "fat": 0.2, "calories": 18.0},
    "cherry tomatoes": {"protein": 0.9, "carbs": 3.9, "fat": 0.2, "calories": 18.0},
    "carrot": {"protein": 0.9, "carbs": 10.0, "fat": 0.2, "calories": 41.0},
    "carrots": {"protein": 0.9, "carbs": 10.0, "fat": 0.2, "calories": 41.0},
    "onion": {"protein": 1.1, "carbs": 9.0, "fat": 0.1, "calories": 40.0},
    "red onion": {"protein": 1.1, "carbs": 9.0, "fat": 0.1, "calories": 40.0},
    "garlic": {"protein": 6.4, "carbs": 33.0, "fat": 0.5, "calories": 149.0},
    "mushroom": {"protein": 3.1, "carbs": 3.3, "fat": 0.3, "calories": 22.0},
    "mushrooms": {"protein": 3.1, "carbs": 3.3, "fat": 0.3, "calories": 22.0},
    "asparagus": {"protein": 2.2, "carbs": 3.9, "fat": 0.1, "calories": 20.0},
    "green beans": {"protein": 1.8, "carbs": 7.0, "fat": 0.2, "calories": 31.0},
    "eggplant": {"protein": 1.0, "carbs": 6.0, "fat": 0.2, "calories": 25.0},
    "aubergine": {"protein": 1.0, "carbs": 6.0, "fat": 0.2, "calories": 25.0},
    "celery": {"protein": 0.7, "carbs": 3.0, "fat": 0.2, "calories": 16.0},
    "lettuce": {"protein": 1.4, "carbs": 2.9, "fat": 0.2, "calories": 15.0},
    "cabbage": {"protein": 1.3, "carbs": 5.8, "fat": 0.1, "calories": 25.0},
    "sweet potato": {"protein": 1.6, "carbs": 20.0, "fat": 0.1, "calories": 86.0},
    # Fruits (common items)
    "banana": {"protein": 1.1, "carbs": 23.0, "fat": 0.3, "calories": 89.0},
    "apple": {"protein": 0.3, "carbs": 14.0, "fat": 0.2, "calories": 52.0},
    "orange": {"protein": 0.9, "carbs": 12.0, "fat": 0.1, "calories": 47.0},
    "blueberries": {"protein": 0.7, "carbs": 14.0, "fat": 0.3, "calories": 57.0},
    "strawberries": {"protein": 0.7, "carbs": 8.0, "fat": 0.3, "calories": 32.0},
    "mixed berries": {"protein": 0.7, "carbs": 12.0, "fat": 0.3, "calories": 50.0},
    # Additional proteins
    "beef": {"protein": 26.0, "carbs": 0.0, "fat": 15.0, "calories": 250.0},
    "ground beef": {"protein": 26.0, "carbs": 0.0, "fat": 15.0, "calories": 250.0},
    "pork": {"protein": 27.0, "carbs": 0.0, "fat": 14.0, "calories": 242.0},
    "shrimp": {"protein": 24.0, "carbs": 0.2, "fat": 0.3, "calories": 99.0},
    "tofu": {"protein": 8.0, "carbs": 1.9, "fat": 4.8, "calories": 76.0},
    # Additional grains/starches
    "white rice": {"protein": 2.7, "carbs": 28.0, "fat": 0.3, "calories": 130.0},
    "pasta": {"protein": 5.0, "carbs": 25.0, "fat": 0.9, "calories": 131.0},
    "bread": {"protein": 9.0, "carbs": 49.0, "fat": 3.2, "calories": 265.0},
    # Legumes
    "chickpeas": {"protein": 8.9, "carbs": 27.0, "fat": 2.6, "calories": 164.0},
    "lentils": {"protein": 9.0, "carbs": 20.0, "fat": 0.4, "calories": 116.0},
    "black beans": {"protein": 8.9, "carbs": 24.0, "fat": 0.5, "calories": 132.0},
    # Additional dairy
    "milk": {"protein": 3.4, "carbs": 5.0, "fat": 1.0, "calories": 42.0},
    "almond milk": {"protein": 0.6, "carbs": 1.4, "fat": 1.1, "calories": 17.0},
    "butter": {"protein": 0.9, "carbs": 0.1, "fat": 81.0, "calories": 717.0},
    "honey": {"protein": 0.3, "carbs": 82.0, "fat": 0.0, "calories": 304.0},
}


# =============================================================================
# Food Category Thresholds (for sanity checks)
# =============================================================================
# Maximum expected carbs per 100g by category - values above these are likely errors
CATEGORY_CARB_LIMITS: Dict[str, float] = {
    "vegetable": 15.0,         # Vegetables rarely exceed 15g carbs/100g (except starchy ones)
    "leafy_green": 8.0,        # Leafy greens are very low carb
    "fruit": 25.0,             # Fresh fruits can have more natural sugars
    "protein": 5.0,            # Pure proteins have minimal carbs
    "dairy": 16.0,             # Most dairy is low carb (slightly increased for tolerance)
    "fat": 2.0,                # Pure fats have no carbs
    # NEW: Categories for processed/plant-based foods that were causing false positives
    "processed_fruit": 80.0,   # Jams, compotes, fruit preserves (high sugar content is normal)
    "plant_based_dairy": 20.0, # Soy/oat/almond yogurt (slightly higher carbs than cow dairy)
    "condiment": 50.0,         # Sauces, spreads, condiments (varied carb content)
}

# Keyword-based category detection
FOOD_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "vegetable": [
        "zucchini", "courgette", "bell pepper", "pepper", "broccoli", "cauliflower",
        "spinach", "kale", "cabbage", "carrot", "celery", "cucumber", "eggplant",
        "aubergine", "asparagus", "green bean", "mushroom", "onion", "tomato",
        "lettuce", "arugula", "chard", "beet", "radish", "turnip", "leek",
        "peas", "artichoke", "fennel", "brussels sprout",
    ],
    "leafy_green": [
        "spinach", "kale", "lettuce", "arugula", "chard", "collard", "watercress",
        "rocket", "endive", "radicchio",
    ],
    "fruit": [
        "apple", "banana", "orange", "grape", "berry", "strawberry", "blueberry",
        "raspberry", "blackberry", "mango", "pineapple", "watermelon", "melon",
        "peach", "pear", "plum", "cherry", "kiwi", "papaya", "fig",
    ],
    "protein": [
        "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "egg",
        "turkey", "lamb", "duck", "tofu", "tempeh", "seitan", "cod", "tilapia",
        "mackerel", "sardine", "anchovy", "crab", "lobster", "scallop",
    ],
    "dairy": [
        "milk", "yogurt", "cheese", "cream", "butter", "ghee", "kefir",
        "cottage cheese", "ricotta", "mascarpone",
    ],
    "fat": [
        "olive oil", "coconut oil", "avocado oil", "vegetable oil", "canola oil",
        "sunflower oil", "sesame oil", "ghee", "lard",
    ],
    # NEW: Categories for processed/plant-based foods
    "processed_fruit": [
        "jam", "jelly", "compote", "preserve", "marmalade", "confiture",
        "fruit spread", "fruit butter", "coulis",
    ],
    "plant_based_dairy": [
        "soy yogurt", "soya yogurt", "oat yogurt", "almond yogurt", "coconut yogurt",
        "plant yogurt", "vegan yogurt", "dairy-free yogurt", "non-dairy yogurt",
        "soy milk", "oat milk", "almond milk", "coconut milk", "rice milk",
    ],
    "condiment": [
        "ketchup", "mustard", "mayonnaise", "sauce", "dressing", "relish",
        "chutney", "salsa", "pesto", "hummus", "tahini", "sriracha",
    ],
}


def detect_food_category(food_name: str, query: str = "") -> Optional[str]:
    """Detect the food category based on name keywords.

    Args:
        food_name: The food's display name
        query: The search query used

    Returns:
        The most specific matching category or None if no match.
        Priority order (most specific first):
        1. processed_fruit (jams, preserves - before fruit)
        2. plant_based_dairy (soy yogurt, oat milk - before dairy)
        3. condiment (sauces, spreads)
        4. leafy_green (more specific than vegetable)
        5. vegetable, fruit, protein, dairy, fat
    """
    combined = f"{food_name} {query}".lower()

    # PRIORITY 1: Check processed_fruit BEFORE fruit
    # This prevents "strawberry jam" from being categorized as "fruit"
    for keyword in FOOD_CATEGORY_KEYWORDS.get("processed_fruit", []):
        if keyword in combined:
            return "processed_fruit"

    # PRIORITY 2: Check plant_based_dairy BEFORE dairy
    # This prevents "soy yogurt" from being categorized as "dairy"
    for keyword in FOOD_CATEGORY_KEYWORDS.get("plant_based_dairy", []):
        if keyword in combined:
            return "plant_based_dairy"

    # PRIORITY 3: Check condiments (varied carb content)
    for keyword in FOOD_CATEGORY_KEYWORDS.get("condiment", []):
        if keyword in combined:
            return "condiment"

    # PRIORITY 4: Check leafy_green (more specific than vegetable)
    for keyword in FOOD_CATEGORY_KEYWORDS.get("leafy_green", []):
        if keyword in combined:
            return "leafy_green"

    # Check other categories in priority order
    for category in ["vegetable", "fruit", "protein", "dairy", "fat"]:
        for keyword in FOOD_CATEGORY_KEYWORDS.get(category, []):
            if keyword in combined:
                return category

    return None


def is_nutrition_sane(food: Dict[str, Any], query: str = "") -> bool:
    """Check if nutritional data is plausible.

    Detects obviously wrong data like:
    - All nutrition values are 0 or missing (no data)
    - Carbs > 100g/100g for non-pure-carb foods
    - Category-specific limits (vegetables > 15g carbs, proteins > 5g carbs, etc.)
    - Protein powder with > 30g carbs/100g
    - Total macros don't roughly match calories
    """
    protein = food.get("protein_per_100g") or food.get("nutrients", {}).get("protein", 0) or 0
    carbs = food.get("carbs_per_100g") or food.get("nutrients", {}).get("carbs", 0) or 0
    fat = food.get("fat_per_100g") or food.get("nutrients", {}).get("fat", 0) or 0
    calories = food.get("calories_per_100g") or food.get("nutrients", {}).get("calories", 0) or 0

    food_name = (food.get("displayName") or food.get("shortName") or food.get("passio_food_name") or "").lower()
    query_lower = query.lower()

    # Rule 0: All values are 0 or missing = no real nutrition data
    # This is a critical check to prevent caching entries without actual data
    if protein == 0 and carbs == 0 and fat == 0 and calories == 0:
        print(f"   ⚠️  SANITY FAIL: {food_name or query} has all-zero nutrition (no data)", file=sys.stderr)
        return False

    # Rule 1: Carbs > 100g/100g is impossible (except pure sugar/starch)
    if carbs > 100 and "sugar" not in food_name and "starch" not in food_name:
        print(f"   ⚠️  SANITY FAIL: {food_name} has {carbs}g carbs/100g (impossible)", file=sys.stderr)
        return False

    # Rule 1.5: Category-specific carb limits (e.g., vegetables shouldn't have 76g carbs)
    category = detect_food_category(food_name, query_lower)
    if category and category in CATEGORY_CARB_LIMITS:
        max_carbs = CATEGORY_CARB_LIMITS[category]
        if carbs > max_carbs:
            print(f"   ⚠️  SANITY FAIL: {food_name} ({category}) has {carbs}g carbs/100g (max for {category}: {max_carbs}g)", file=sys.stderr)
            return False

    # Rule 2: Protein powder/whey with > 30g carbs is wrong
    is_protein_powder = any(kw in food_name or kw in query_lower
                            for kw in ["whey", "protein powder", "casein", "isolate"])
    if is_protein_powder and carbs > 30:
        print(f"   ⚠️  SANITY FAIL: {food_name} (protein powder) has {carbs}g carbs/100g (should be <30g)", file=sys.stderr)
        return False

    # Rule 3: Calories should roughly match macro calculation (±30%)
    # Formula: protein*4 + carbs*4 + fat*9 ≈ calories
    if calories > 50:  # Only check if meaningful calories
        calculated = protein * 4 + carbs * 4 + fat * 9
        if calculated > 0:
            ratio = calories / calculated
            if ratio < 0.5 or ratio > 2.0:
                print(f"   ⚠️  SANITY FAIL: {food_name} calorie mismatch (stated: {calories}, calculated: {calculated:.0f})", file=sys.stderr)
                return False

    return True


def is_search_result_entry(food: Dict[str, Any]) -> bool:
    """Check if this is a search result (no nutrition) vs detail result (has nutrition).

    Search results from hexis_search_passio_foods have refCode but no nutrients.
    Detail results from hexis_get_passio_food_details have nutrients array.

    This is used to skip nutrition validation for search results, since they
    don't include nutrition data by design - it's fetched separately via
    hexis_get_passio_food_details.
    """
    has_ref_code = bool(food.get("refCode"))
    has_nutrients = "nutrients" in food and food["nutrients"]
    # Also check flat nutrition fields (from cache or enriched results)
    has_flat_nutrition = any(
        food.get(f) for f in ["protein_per_100g", "carbs_per_100g", "fat_per_100g", "calories_per_100g"]
    )
    return has_ref_code and not has_nutrients and not has_flat_nutrition


def validate_against_trusted(food: Dict[str, Any], query: str = "") -> Tuple[bool, Optional[str]]:
    """Validate nutrition against TRUSTED_INGREDIENTS if a match exists.

    This catches cases where the Passio API returns incorrect data (e.g., Greek yogurt
    with 633 kcal/100g instead of ~59 kcal/100g) that would pass generic sanity checks.

    Args:
        food: Dict containing nutrition data (protein_per_100g, carbs_per_100g, etc.)
        query: The search query used to find this food

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if no trusted match exists or values are within tolerance
        - (False, "error message") if values deviate significantly from trusted data
    """
    query_lower = query.lower().strip()
    trusted = None
    matched_key = None

    # Try to find a matching trusted ingredient
    if query_lower in TRUSTED_INGREDIENTS:
        trusted = TRUSTED_INGREDIENTS[query_lower]
        matched_key = query_lower
    else:
        for ingredient, nutrients in TRUSTED_INGREDIENTS.items():
            if ingredient in query_lower or query_lower in ingredient:
                trusted = nutrients
                matched_key = ingredient
                break

    if trusted is None:
        return (True, None)  # No trusted reference, can't validate

    # Extract nutrition from food dict (handle different key formats)
    protein = food.get("protein_per_100g") or food.get("nutrients", {}).get("protein", 0) or 0
    carbs = food.get("carbs_per_100g") or food.get("nutrients", {}).get("carbs", 0) or 0
    fat = food.get("fat_per_100g") or food.get("nutrients", {}).get("fat", 0) or 0
    calories = food.get("calories_per_100g") or food.get("nutrients", {}).get("calories", 0) or 0

    # Tolerance: 50% deviation from trusted values (generous to allow for brand variations)
    TOLERANCE = 0.5
    issues = []

    for macro, api_val, trusted_val in [
        ("fat", fat, trusted["fat"]),
        ("calories", calories, trusted["calories"]),
    ]:
        if trusted_val > 0:
            deviation = abs(api_val - trusted_val) / trusted_val
            if deviation > TOLERANCE:
                issues.append(f"{macro}: API={api_val:.1f}, trusted={trusted_val:.1f} ({deviation*100:.0f}% off)")

    if issues:
        food_name = food.get("displayName") or food.get("shortName") or food.get("passio_food_name") or query
        error_msg = f"TRUSTED MISMATCH for '{matched_key}' ({food_name}): {'; '.join(issues)}"
        return (False, error_msg)

    return (True, None)


def get_trusted_nutrition(query: str) -> Optional[Dict[str, Any]]:
    """Get nutrition from trusted database if available.

    Returns a Passio-like food item with verified nutritional data.
    """
    query_lower = query.lower().strip()

    # Try exact match first
    if query_lower in TRUSTED_INGREDIENTS:
        nutrients = TRUSTED_INGREDIENTS[query_lower]
        return _build_trusted_food_item(query, nutrients)

    # Try partial match
    for ingredient, nutrients in TRUSTED_INGREDIENTS.items():
        if ingredient in query_lower or query_lower in ingredient:
            print(f"   🔒 Using TRUSTED data for '{query}' → matched '{ingredient}'", file=sys.stderr)
            return _build_trusted_food_item(query, nutrients)

    return None


def _build_trusted_food_item(name: str, nutrients: Dict[str, float]) -> Dict[str, Any]:
    """Build a Passio-like food item from trusted nutrients."""
    return {
        "id": f"trusted_{name.lower().replace(' ', '_')}",
        "resultId": f"trusted_{name.lower().replace(' ', '_')}",
        "refCode": None,  # No Passio ref code for trusted items
        "displayName": name.title(),
        "shortName": name.title(),
        "type": "trusted",
        "dataOrigin": "TRUSTED_DATABASE",
        "nutrients": nutrients,
        "protein_per_100g": nutrients["protein"],
        "carbs_per_100g": nutrients["carbs"],
        "fat_per_100g": nutrients["fat"],
        "calories_per_100g": nutrients["calories"],
    }


def prefer_verified_entries(foods: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
    """Reorder foods list to prefer verified entries over OpenFood (crowdsourced).

    OpenFood entries (id starting with 'openfood') are user-submitted and
    often contain errors. Prefer entries from Passio's verified database.

    For search results (which don't include nutrition data), we skip the sanity
    check since nutrition is fetched separately via hexis_get_passio_food_details.
    For detail results (which should have nutrition), we filter out invalid entries.
    """
    if not foods:
        return foods

    verified = []
    openfood = []
    search_results_count = 0

    for food in foods:
        food_id = str(food.get("id", "") or food.get("resultId", ""))

        # Don't filter search results by nutrition - they don't have it!
        # Nutrition is fetched separately via hexis_get_passio_food_details
        if is_search_result_entry(food):
            search_results_count += 1
            # Keep search result, just categorize by source
        elif not is_nutrition_sane(food, query):
            # Filter out detail results with invalid nutrition
            continue

        if food_id.startswith("openfood"):
            openfood.append(food)
        else:
            verified.append(food)

    # Return verified first, then openfood as fallback
    result = verified + openfood

    # Only log if we filtered detail results (not search results)
    detail_filtered = len(foods) - len(result) - search_results_count
    if detail_filtered > 0:
        print(f"   🛡️  Filtered out {detail_filtered} entries with invalid nutrition data", file=sys.stderr)

    if search_results_count > 0:
        print(f"   📋 Kept {len(result)} search result(s) (nutrition fetched separately)", file=sys.stderr)

    if verified and openfood:
        print(f"   ✅ Preferring {len(verified)} verified entries over {len(openfood)} OpenFood entries", file=sys.stderr)

    return result


# Global counter for tracking tool calls
_tool_call_counter: Dict[str, int] = {}

# Global storage for captured Hexis tool results
# This allows Python to access tool results directly without relying on LLM output
_hexis_tool_results: List[Dict[str, Any]] = []


def reset_tool_call_counter() -> None:
    """Reset the global tool call counter."""
    global _tool_call_counter
    _tool_call_counter = {}


def reset_hexis_tool_results() -> None:
    """Reset the captured Hexis tool results."""
    global _hexis_tool_results
    _hexis_tool_results = []


def get_hexis_tool_results() -> List[Dict[str, Any]]:
    """Get the captured Hexis tool results.

    Returns:
        List of tool result dicts with keys:
        - tool_name: str
        - parameters: Dict (start_date, end_date)
        - success: bool
        - result: Dict (raw API response)
        - error_message: Optional[str]
    """
    return _hexis_tool_results


def capture_hexis_result(
    tool_name: str,
    parameters: Dict[str, Any],
    result: Any,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    """Capture a Hexis tool result for later retrieval.

    Args:
        tool_name: Name of the tool called
        parameters: Parameters passed to the tool
        result: Raw result from the tool
        success: Whether the call succeeded
        error_message: Error message if failed
    """
    global _hexis_tool_results
    _hexis_tool_results.append({
        "tool_name": tool_name,
        "parameters": parameters,
        "success": success,
        "result": result,
        "error_message": error_message,
    })


def get_tool_call_count(tool_name: Optional[str] = None) -> int:
    """
    Get the number of times a tool was called.

    Args:
        tool_name: Optional specific tool name. If None, returns total calls.

    Returns:
        Number of calls for the tool (or total if tool_name is None)
    """
    if tool_name is None:
        return sum(_tool_call_counter.values())
    return _tool_call_counter.get(tool_name, 0)


def get_tool_call_summary() -> Dict[str, int]:
    """Get a summary of all tool calls."""
    return dict(_tool_call_counter)


def validate_tool_input(tool_input: Any) -> Dict[str, Any]:
    """
    Validate and fix malformed tool inputs from LLM.

    Common issues:
    - LLM passes a list instead of a dict
    - LLM includes multiple parameter sets
    - LLM includes mocked responses in the input

    Args:
        tool_input: Raw input from LLM (could be dict, list, str, etc.)

    Returns:
        Valid dictionary of parameters

    Raises:
        ValueError: If input cannot be salvaged
    """
    # If it's already a dict, return as-is
    if isinstance(tool_input, dict):
        # Check if it looks like a mocked response (has "status" and "data")
        if "status" in tool_input and "data" in tool_input:
            raise ValueError(
                "Tool input appears to be a mocked response. "
                "Pass only the parameters, not the expected response."
            )
        return tool_input

    # If it's a list, extract all valid parameter dicts
    if isinstance(tool_input, list):
        print(
            f"⚠️  Warning: Tool received a list instead of dict. "
            f"Extracting valid parameter dicts...",
            file=sys.stderr
        )

        # Collect all dicts that look like parameters (not responses)
        valid_items = []
        for item in tool_input:
            if isinstance(item, dict):
                # Skip items that look like mocked responses
                if "status" in item and "data" in item:
                    continue
                # Skip items that look like Hexis results (hallucinated output)
                if "days" in item and isinstance(item["days"], list):
                    continue
                valid_items.append(item)

        if len(valid_items) == 0:
            raise ValueError(
                f"Could not extract valid parameters from list: {tool_input}. "
                f"Please pass a single dictionary with parameters only."
            )
        elif len(valid_items) == 1:
            # Single item - return as dict
            print(f"✅ Extracted single parameter dict: {valid_items[0]}", file=sys.stderr)
            return valid_items[0]
        else:
            # Multiple items - return batch marker for sequential processing
            print(f"✅ Extracted {len(valid_items)} parameter dicts for batch processing", file=sys.stderr)
            return {"__batch_items__": valid_items}

    # If it's a string, try to parse as JSON
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
            return validate_tool_input(parsed)  # Recursive call
        except json.JSONDecodeError:
            raise ValueError(f"Tool input is a string but not valid JSON: {tool_input}")

    # Unknown type
    raise ValueError(
        f"Tool input must be a dictionary, got {type(tool_input).__name__}: {tool_input}"
    )


class PermissiveToolWrapper(BaseTool):
    """
    A permissive tool wrapper that accepts any input and validates it manually.

    This bypasses CrewAI's strict schema validation which rejects list inputs.
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    original_tool: Any = Field(description="Original MCP tool to delegate to")

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> Any:
        """
        Run the tool with manual input validation.

        This method receives kwargs from CrewAI. We need to handle cases where:
        - The LLM passes a list instead of a dict
        - The list contains multiple parameter sets
        - The list contains mocked responses
        """
        tool_name = getattr(self.original_tool, 'name', self.name)

        # If we received multiple kwargs, try to extract the real parameters
        # Sometimes CrewAI passes the entire input as a single kwarg
        if len(kwargs) == 1:
            key, value = list(kwargs.items())[0]
            # If the single kwarg looks like it contains the actual input, use it
            if isinstance(value, (dict, list)):
                try:
                    validated = validate_tool_input(value)
                    print(f"✅ Successfully validated input for {tool_name}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️  Validation warning for {tool_name}: {e}", file=sys.stderr)
                    print(f"   Attempting to use original input...", file=sys.stderr)
                    validated = kwargs
            else:
                validated = kwargs
        else:
            # Multiple kwargs - try to validate as-is
            try:
                validated = validate_tool_input(kwargs)
            except Exception as e:
                print(f"⚠️  Could not validate kwargs for {tool_name}: {e}", file=sys.stderr)
                validated = kwargs

        # Call the original tool with validated input
        original_run = getattr(self.original_tool, '_run', None) or getattr(self.original_tool, 'run', None)
        if original_run:
            if isinstance(validated, dict):
                return original_run(**validated)
            return original_run(validated)
        else:
            raise RuntimeError(f"Tool {tool_name} has no _run or run method")

    async def _arun(self, **kwargs: Any) -> Any:
        """
        Asynchronously run the tool.
        
        Since the underlying validation and original tool might be sync,
        we offload the execution to a thread to avoid blocking the loop.
        """
        return await asyncio.to_thread(self._run, **kwargs)


# =============================================================================
# Passio Search Cache and Response Filtering
# =============================================================================

# Essential fields to keep from Passio search results
PASSIO_ESSENTIAL_FIELDS = {
    "id", "resultId", "refCode", "displayName", "shortName", "longName",
    "type", "dataOrigin", "scoredName",
    # Nutrition fields (from get_passio_food_details)
    "protein_per_100g", "carbs_per_100g", "fat_per_100g", "calories_per_100g",
    "protein", "carbs", "fat", "calories",
    "nutrients",
}


def filter_passio_food(food: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a single Passio food item to keep only essential fields.

    Reduces token usage by ~70% while preserving all necessary data for
    ingredient validation and Hexis integration.
    """
    if not isinstance(food, dict):
        return food

    filtered = {}
    for key in PASSIO_ESSENTIAL_FIELDS:
        if key in food:
            filtered[key] = food[key]

    # Keep nested nutrients if present
    if "nutrients" in food and isinstance(food["nutrients"], dict):
        filtered["nutrients"] = food["nutrients"]

    return filtered


def filter_passio_response(result: Any, query: str = "") -> Any:
    """Filter Passio API response to reduce token usage and improve data quality.

    Handles both search results (list of foods) and detail results (single food).
    Also applies:
    - Sanity checks to filter out entries with invalid nutritional data
    - Preference for verified entries over crowdsourced OpenFood entries
    """
    if result is None:
        return result

    # Handle string results (already serialized JSON)
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            filtered = filter_passio_response(parsed, query)
            return json.dumps(filtered)
        except (json.JSONDecodeError, TypeError):
            return result

    # Handle dict with "foods" key (search result)
    if isinstance(result, dict):
        if "foods" in result and isinstance(result["foods"], list):
            # Step 1: Filter fields
            filtered_foods = [filter_passio_food(f) for f in result["foods"]]
            # Step 2: Apply sanity checks and prefer verified entries
            result["foods"] = prefer_verified_entries(filtered_foods, query)
        else:
            # Single food item
            result = filter_passio_food(result)
        return result

    # Handle list of foods
    if isinstance(result, list):
        filtered = [filter_passio_food(f) for f in result]
        return prefer_verified_entries(filtered, query)

    return result


def extract_passio_data_for_cache(result: Any, query: str) -> Optional[Dict[str, Any]]:
    """Extract data from Passio search result for caching.

    Returns the first food item with essential fields for the search cache.
    Returns None if no nutrition data is available (prevents caching incomplete data).
    """
    foods = None

    # Parse string result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "foods" in parsed:
                foods = parsed["foods"]
            elif isinstance(parsed, list):
                foods = parsed
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(result, dict) and "foods" in result:
        foods = result["foods"]
    elif isinstance(result, list):
        foods = result

    if not foods or len(foods) == 0:
        return None

    # Get the first (best match) food item
    first_food = foods[0]
    if not isinstance(first_food, dict):
        return None

    # Extract essential fields for cache
    cache_data = {
        "passio_food_id": first_food.get("id") or first_food.get("resultId"),
        "passio_ref_code": first_food.get("refCode"),
        "passio_food_name": first_food.get("displayName") or first_food.get("shortName"),
    }

    # Extract nutrition if available (from enriched results or get_passio_food_details)
    has_nutrition = False
    if "nutrients" in first_food:
        nutrients = first_food["nutrients"]
        cache_data["protein_per_100g"] = nutrients.get("protein", 0)
        cache_data["carbs_per_100g"] = nutrients.get("carbs", 0)
        cache_data["fat_per_100g"] = nutrients.get("fat", 0)
        cache_data["calories_per_100g"] = nutrients.get("calories", 0)
        has_nutrition = True
    elif "protein_per_100g" in first_food:
        cache_data["protein_per_100g"] = first_food.get("protein_per_100g", 0)
        cache_data["carbs_per_100g"] = first_food.get("carbs_per_100g", 0)
        cache_data["fat_per_100g"] = first_food.get("fat_per_100g", 0)
        cache_data["calories_per_100g"] = first_food.get("calories_per_100g", 0)
        has_nutrition = True

    # CRITICAL: Only return cache_data if nutrition was actually extracted
    # This prevents caching entries with missing/unknown nutrition values
    if not has_nutrition:
        # Cache negative result to avoid repeated API calls
        search_cache = get_passio_search_cache()
        search_cache.set_negative(query, "no_nutrition_data")
        print(f"   ⚠️  Cached NEGATIVE for '{query}' - no nutrition data in response", file=sys.stderr)
        return None

    return cache_data


def build_cached_passio_response(cached_data: Dict[str, Any]) -> str:
    """Build a Passio-like response from cached data.

    Creates a minimal response that looks like a Passio search result
    but contains only the cached food item.
    """
    food_item = {
        "id": cached_data.get("passio_food_id"),
        "resultId": cached_data.get("passio_food_id"),
        "refCode": cached_data.get("passio_ref_code"),
        "displayName": cached_data.get("passio_food_name"),
        "shortName": cached_data.get("passio_food_name"),
        "type": "cached",
        "dataOrigin": "PASSIO_CACHE",
    }

    # Add nutrition if available
    if cached_data.get("protein_per_100g") is not None:
        food_item["nutrients"] = {
            "protein": cached_data.get("protein_per_100g", 0),
            "carbs": cached_data.get("carbs_per_100g", 0),
            "fat": cached_data.get("fat_per_100g", 0),
            "calories": cached_data.get("calories_per_100g", 0),
        }
        food_item["protein_per_100g"] = cached_data.get("protein_per_100g", 0)
        food_item["carbs_per_100g"] = cached_data.get("carbs_per_100g", 0)
        food_item["fat_per_100g"] = cached_data.get("fat_per_100g", 0)
        food_item["calories_per_100g"] = cached_data.get("calories_per_100g", 0)

    response = {"foods": [food_item]}
    return json.dumps(response)


def wrap_mcp_tool(tool: Any) -> Any:
    """
    Wrap an MCP tool to add input validation and logging.

    This wrapper intercepts the tool execution and:
    1. Logs tool calls for debugging
    2. Handles list inputs that some LLMs incorrectly send
    3. Validates inputs before passing to the original tool

    Note: We NO LONGER replace the args_schema because that was causing
    the tool arguments to be lost when Pydantic validated against the
    permissive schema. The original tool's schema is preserved.

    Args:
        tool: MCP tool object with a `_run` or `run` method

    Returns:
        Wrapped tool with validation
    """
    # Store original run method
    original_run = getattr(tool, '_run', None) or getattr(tool, 'run', None)

    if not original_run:
        # Tool doesn't have a run method, return as-is
        return tool

    tool_name = getattr(tool, 'name', 'unknown')

    # Store original schema for reference (but do NOT replace it!)
    original_schema = getattr(tool, 'args_schema', None)
    tool._original_args_schema = original_schema
    # NOTE: We no longer set tool.args_schema = PermissiveSchema
    # This was causing the actual arguments to be lost when CrewAI
    # validated {"start_date": "...", "end_date": "..."} against a schema
    # that expected {"tool_input": {...}}, resulting in tool_input={}

    def validated_run(*args, **kwargs) -> Any:
        """Run the tool with input validation."""
        tool_name = tool.name if hasattr(tool, 'name') else 'unknown'

        # Increment tool call counter
        global _tool_call_counter
        _tool_call_counter[tool_name] = _tool_call_counter.get(tool_name, 0) + 1

        # Log that the tool is being called
        print(f"\n🔧 MCP Tool Call: {tool_name}", file=sys.stderr)
        print(f"   Args: {args[:1] if args else 'none'}", file=sys.stderr)
        print(f"   Kwargs keys: {list(kwargs.keys()) if kwargs else 'none'}", file=sys.stderr)

        # Remove security_context if present (CrewAI internal, not for the tool)
        kwargs.pop('security_context', None)

        # Determine the input to validate
        raw_input = None

        # Case 1: Legacy permissive schema input (tool_input key)
        # This was from the old approach - kept for backwards compatibility
        if 'tool_input' in kwargs and kwargs['tool_input']:
            raw_input = kwargs.pop('tool_input')
            print(f"   Legacy tool_input detected: {type(raw_input).__name__}", file=sys.stderr)
        # Case 2: Direct kwargs (the normal case now that we preserve original schema)
        elif kwargs:
            raw_input = kwargs
            kwargs = {}  # Clear so we don't double-pass
            print(f"   Direct kwargs: {list(raw_input.keys())}", file=sys.stderr)
        # Case 3: Positional args (fallback)
        elif args:
            raw_input = args[0]
            args = args[1:]
            print(f"   Positional arg: {type(raw_input).__name__}", file=sys.stderr)

        if raw_input is not None:
            try:
                validated_input = validate_tool_input(raw_input)

                # =============================================================
                # Passio Search Optimization: Cache + Filtering
                # =============================================================
                is_passio_search = "passio" in tool_name.lower() and "search" in tool_name.lower()

                # Inject default limit for Passio search tools to avoid context overflow
                # The Passio API can return 100+ results (150K+ chars) without a limit
                if is_passio_search:
                    if isinstance(validated_input, dict) and "limit" not in validated_input:
                        default_limit = int(os.getenv("PASSIO_DEFAULT_LIMIT", "5"))
                        validated_input["limit"] = default_limit
                        print(f"   📉 Added default limit={default_limit} to Passio search", file=sys.stderr)

                    query = validated_input.get("query", "") if isinstance(validated_input, dict) else ""
                    if query:
                        # Priority 1: Check TRUSTED ingredients database (verified data)
                        trusted_food = get_trusted_nutrition(query)
                        if trusted_food:
                            result = json.dumps({"foods": [trusted_food]})
                            print(f"   🔒 TRUSTED HIT for '{query}' - using verified nutritional data", file=sys.stderr)
                            return result

                        # Priority 2: Check cache for Passio search queries
                        search_cache = get_passio_search_cache()
                        cached_result = search_cache.get(query)
                        if cached_result:
                            # Check if this is a negative cache entry
                            if cached_result.get("negative"):
                                reason = cached_result.get("reason", "no_data")
                                print(f"   ⛔ NEGATIVE CACHE HIT for '{query}' - {reason}, skipping API call", file=sys.stderr)
                                # Return empty result to avoid repeated API calls
                                return json.dumps({"foods": [], "negative_cache": True, "reason": reason})
                            # Validate cached data passes sanity check
                            if is_nutrition_sane(cached_result, query):
                                print(f"   🎯 CACHE HIT for '{query}' - skipping API call", file=sys.stderr)
                                result = build_cached_passio_response(cached_result)
                                result_summary = str(result)[:200]
                                print(f"   ✅ Cached result: {result_summary}...\n", file=sys.stderr)
                                return result
                            else:
                                print(f"   ⚠️  CACHE INVALIDATED for '{query}' - failed sanity check", file=sys.stderr)
                                search_cache.delete(query)

                # Check for batch marker - process multiple items sequentially
                if isinstance(validated_input, dict) and "__batch_items__" in validated_input:
                    batch_items = validated_input["__batch_items__"]
                    print(f"   🔄 Batch processing {len(batch_items)} items...", file=sys.stderr)
                    results = []
                    for i, item in enumerate(batch_items, 1):
                        print(f"   📝 Processing item {i}/{len(batch_items)}: {list(item.keys())}", file=sys.stderr)
                        item_result = original_run(**item)
                        results.append(item_result)
                    result = results
                    print(f"   ✅ Batch complete: {len(results)} results", file=sys.stderr)
                else:
                    print(f"   ✅ Validated input: {list(validated_input.keys()) if isinstance(validated_input, dict) else validated_input}", file=sys.stderr)

                    # =============================================================
                    # Passio API Retry with Exponential Backoff
                    # =============================================================
                    is_passio_tool = "passio" in tool_name.lower() or "hexis" in tool_name.lower()

                    if is_passio_tool:
                        # Use exponential backoff for Passio/Hexis API calls
                        max_retries = DEFAULT_MAX_RETRIES
                        circuit_breaker = get_passio_circuit_breaker()

                        # Rate limit Passio API calls to avoid 500 errors
                        _passio_rate_limit()

                        for attempt in range(max_retries + 1):
                            # Check circuit breaker
                            if not circuit_breaker.can_execute():
                                print(f"   ⚡ Circuit breaker open, skipping Passio API call", file=sys.stderr)
                                result = json.dumps({"error": "Circuit breaker open - Passio API temporarily unavailable"})
                                break

                            try:
                                if isinstance(validated_input, dict):
                                    result = original_run(**validated_input)
                                else:
                                    result = original_run(validated_input, *args, **kwargs)

                                # Check if result contains an error
                                result_str = str(result) if result else ""
                                if "error" in result_str.lower() and ("500" in result_str or "error while searching" in result_str.lower()):
                                    raise RuntimeError(f"Passio API error: {result_str[:200]}")

                                # Success - reset circuit breaker
                                circuit_breaker.record_success()
                                break

                            except Exception as e:
                                circuit_breaker.record_failure()

                                if attempt >= max_retries or not is_retriable_error(e):
                                    print(f"   ❌ Passio API failed after {attempt + 1} attempts: {e}", file=sys.stderr)
                                    result = json.dumps({"error": str(e)})
                                    break

                                delay = exponential_backoff_delay(attempt)
                                print(
                                    f"   ⏳ Passio retry {attempt + 1}/{max_retries} after {delay:.1f}s: {str(e)[:100]}",
                                    file=sys.stderr,
                                )
                                time.sleep(delay)
                    else:
                        # Non-Passio tools: call directly without retry
                        if isinstance(validated_input, dict):
                            result = original_run(**validated_input)
                        else:
                            result = original_run(validated_input, *args, **kwargs)

                    # =============================================================
                    # Hexis Tool Result Capture
                    # =============================================================
                    # Capture Hexis weekly plan results in Python for reliable access
                    # This avoids relying on LLM to copy large raw data in its output
                    is_hexis_weekly_plan = "hexis" in tool_name.lower() and "weekly_plan" in tool_name.lower()
                    if is_hexis_weekly_plan and result:
                        try:
                            # Parse result if it's a string
                            result_data = result
                            if isinstance(result, str):
                                result_data = json.loads(result)

                            # Capture the result for later Python access
                            params = validated_input if isinstance(validated_input, dict) else {}
                            capture_hexis_result(
                                tool_name=tool_name,
                                parameters=params,
                                result=result_data,
                                success=True,
                            )
                            print(f"   📦 Captured Hexis result for Python access", file=sys.stderr)
                        except Exception as e:
                            print(f"   ⚠️ Failed to capture Hexis result: {e}", file=sys.stderr)

                    # =============================================================
                    # Passio Post-Processing: Filter + Cache
                    # =============================================================
                    if is_passio_search and result:
                        # Get query for filtering and caching
                        query = validated_input.get("query", "") if isinstance(validated_input, dict) else ""

                        # Filter the response (includes sanity checks and OpenFood deprioritization)
                        original_len = len(str(result))
                        result = filter_passio_response(result, query)
                        filtered_len = len(str(result))
                        reduction = ((original_len - filtered_len) / original_len * 100) if original_len > 0 else 0
                        print(f"   📉 Filtered Passio response: {original_len} → {filtered_len} chars ({reduction:.0f}% reduction)", file=sys.stderr)

                        # Cache the first result for future queries (only if it passed sanity checks)
                        if query:
                            cache_data = extract_passio_data_for_cache(result, query)
                            if cache_data and cache_data.get("passio_ref_code"):
                                # PRIORITY: Check TRUSTED and override if deviation > 50%
                                trusted_food = get_trusted_nutrition(query)
                                if trusted_food:
                                    passio_carbs = cache_data.get("carbs_per_100g", 0) or 0
                                    trusted_carbs = trusted_food.get("carbs_per_100g", 0)
                                    passio_protein = cache_data.get("protein_per_100g", 0) or 0
                                    trusted_protein = trusted_food.get("protein_per_100g", 0)

                                    # Check for >50% deviation in carbs or protein
                                    carb_deviation = abs(passio_carbs - trusted_carbs) / max(trusted_carbs, 1) if trusted_carbs > 0 else 0
                                    protein_deviation = abs(passio_protein - trusted_protein) / max(trusted_protein, 1) if trusted_protein > 0 else 0

                                    if carb_deviation > 0.5 or protein_deviation > 0.5:
                                        print(f"   🔒 OVERRIDING Passio with TRUSTED for '{query}':", file=sys.stderr)
                                        print(f"      Passio: P={passio_protein}g C={passio_carbs}g", file=sys.stderr)
                                        print(f"      Trusted: P={trusted_protein}g C={trusted_carbs}g", file=sys.stderr)
                                        # Merge trusted nutrition with Passio metadata
                                        cache_data["protein_per_100g"] = trusted_food.get("protein_per_100g", 0)
                                        cache_data["carbs_per_100g"] = trusted_food.get("carbs_per_100g", 0)
                                        cache_data["fat_per_100g"] = trusted_food.get("fat_per_100g", 0)
                                        cache_data["calories_per_100g"] = trusted_food.get("calories_per_100g", 0)
                                        cache_data["data_source"] = "TRUSTED_OVERRIDE"

                                # Validate against trusted ingredients (catches Passio API errors)
                                trusted_valid, trusted_error = validate_against_trusted(cache_data, query)
                                if not trusted_valid:
                                    print(f"   ⚠️  NOT CACHING '{query}' - {trusted_error}", file=sys.stderr)
                                    # Replace with trusted data if available and not already overridden
                                    if trusted_food and cache_data.get("data_source") != "TRUSTED_OVERRIDE":
                                        cache_data["protein_per_100g"] = trusted_food.get("protein_per_100g", 0)
                                        cache_data["carbs_per_100g"] = trusted_food.get("carbs_per_100g", 0)
                                        cache_data["fat_per_100g"] = trusted_food.get("fat_per_100g", 0)
                                        cache_data["calories_per_100g"] = trusted_food.get("calories_per_100g", 0)
                                        cache_data["data_source"] = "TRUSTED_OVERRIDE"
                                        print(f"   🔒 Using TRUSTED data for '{query}'", file=sys.stderr)

                                # Only cache if it passes sanity check
                                if is_nutrition_sane(cache_data, query):
                                    search_cache = get_passio_search_cache()
                                    search_cache.set(query, cache_data)
                                    source_note = " (TRUSTED override)" if cache_data.get("data_source") == "TRUSTED_OVERRIDE" else ""
                                    print(f"   💾 Cached result for '{query}': {cache_data.get('passio_food_name')}{source_note}", file=sys.stderr)
                                else:
                                    print(f"   ⚠️  NOT CACHING '{query}' - failed sanity check", file=sys.stderr)

            except Exception as e:
                print(
                    f"❌ Tool input validation failed: {e}\n"
                    f"   Tool: {tool_name}\n"
                    f"   Raw input: {raw_input}",
                    file=sys.stderr
                )
                raise
        else:
            # No input at all, call with original args/kwargs
            print(f"   ⚠️ No input detected, calling with empty args", file=sys.stderr)
            result = original_run(*args, **kwargs)

        # Log the result summary
        result_str = str(result)
        result_len = len(result_str)
        
        # Truncate if too large (100KB limit to avoid Nginx 413 errors)
        max_len = 100000
        if result_len > max_len:
            print(f"⚠️  Warning: Tool output too large ({result_len} chars), truncating to {max_len}...", file=sys.stderr)
            truncated_msg = f"... [TRUNCATED due to size: {result_len} > {max_len} chars]"
            
            if isinstance(result, str):
                result = result[:max_len] + truncated_msg
            elif isinstance(result, dict) or isinstance(result, list):
                # If it's a structured object, we can't easily truncate it while keeping it valid
                # So we convert to string, truncate, and return that
                # This might break tools expecting strict types, but it's better than a crash
                result = str(result)[:max_len] + truncated_msg
                
            result_str = str(result)

        result_summary = result_str[:200] if result else "None"
        print(f"   ✅ Result: {result_summary}{'...' if len(result_str) > 200 else ''}\n", file=sys.stderr)

        return result

    # Replace run method with validated version
    if hasattr(tool, '_run'):
        tool._run = validated_run
    elif hasattr(tool, 'run'):
        tool.run = validated_run

    # Add async run method if not present, wrapping the sync validated run
    if not hasattr(tool, '_arun'):
        async def validated_arun(*args, **kwargs) -> Any:
            return await asyncio.to_thread(validated_run, *args, **kwargs)
        tool._arun = validated_arun

    # Also wrap the __call__ method if it exists (CrewAI may call this directly)
    if hasattr(tool, '__call__') and callable(tool):
        original_call = tool.__call__

        def validated_call(*args, **kwargs) -> Any:
            """Call the tool with input validation."""
            # If first arg is the tool input, validate it
            if args and len(args) > 0 and not callable(args[0]):
                try:
                    validated_input = validate_tool_input(args[0])
                    args = (validated_input,) + args[1:]
                except Exception as e:
                    print(
                        f"❌ Tool input validation failed in __call__: {e}\n"
                        f"   Tool: {tool.name if hasattr(tool, 'name') else 'unknown'}\n"
                        f"   Raw input: {args[0]}",
                        file=sys.stderr
                    )
                    # Try to proceed with original call anyway
                    pass

            return original_call(*args, **kwargs)

        # Don't replace __call__ on the class, create a wrapper instance
        # This is tricky, so we skip it for now

    return tool


def wrap_mcp_tools(tools: List[Any]) -> List[Any]:
    """
    Wrap a list of MCP tools with input validation.

    Args:
        tools: List of MCP tool objects

    Returns:
        List of wrapped tools
    """
    return [wrap_mcp_tool(tool) for tool in tools]
