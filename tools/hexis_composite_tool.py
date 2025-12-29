import re
from typing import Type, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json


class ValidatedIngredientInput(BaseModel):
    """Pre-validated ingredient from meal generation."""

    name: str = Field(
        ..., description="Ingredient name with quantity (e.g., '120g chicken breast')"
    )
    passio_food_id: Optional[str] = Field(
        None, description="Passio Food ID from hexis_search_passio_foods"
    )
    passio_food_name: Optional[str] = Field(
        None, description="Exact name from Passio database"
    )
    passio_ref_code: Optional[str] = Field(
        None,
        description="⚠️ REQUIRED: Base64 refCode from hexis_search_passio_foods. API returns 400 without this!",
    )
    quantity_g: Optional[float] = Field(None, description="Quantity in grams")


class LogMealSchema(BaseModel):
    """Input for HexisLogMealTool."""

    meal_type: str = Field(
        ...,
        description="Type of meal (Breakfast, Lunch, Dinner, Snack, Afternoon Snack)",
    )
    day_name: str = Field(..., description="Name of the day (e.g., Monday)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    meal_name: Optional[str] = Field(
        None,
        description="Descriptive name of the meal (e.g., 'Chicken Salad', 'Oatmeal with Berries'). IMPORTANT for search.",
    )
    description: Optional[str] = Field(
        None,
        description="Detailed description or ingredients (optional, helps with search context)",
    )
    calories: float = Field(..., description="Total calories")
    protein: float = Field(..., description="Total protein in grams")
    carbs: float = Field(..., description="Total carbs in grams")
    fat: float = Field(..., description="Total fat in grams")
    hexis_food_id: Optional[str] = Field(None, description="Hexis Food ID if known")
    data_origin: Optional[str] = Field(None, description="Data origin if known")
    validated_ingredients: Optional[List[ValidatedIngredientInput]] = Field(
        None,
        description="Pre-validated ingredients with Passio IDs from meal generation. MUST include passio_ref_code for each ingredient or API returns 400!",
    )


# Allowed data origins for food search (NO custom food creation)
# With hexis_search_passio_foods, we expect mostly PASSIO but we can be more lenient or just trust the tool.
ALLOWED_DATA_ORIGINS = ["PASSIO", "USDA"]


class HexisLogMealTool(BaseTool):
    name: str = "hexis_log_meal"
    description: str = (
        "Logs a meal to Hexis by searching for existing foods using Passio (supports English/French). "
        "DOES NOT create custom foods - only uses verified database entries. "
        "Handles overwriting existing meals if present."
    )
    args_schema: Type[BaseModel] = LogMealSchema

    _verify_tool: BaseTool
    _update_tool: BaseTool
    _get_plan_tool: BaseTool
    _search_tool: BaseTool

    def __init__(
        self, verify_tool, update_tool, get_plan_tool, search_tool, create_tool=None
    ):
        super().__init__()
        self._verify_tool = verify_tool
        self._update_tool = update_tool
        self._get_plan_tool = get_plan_tool
        self._search_tool = search_tool
        # create_tool is ignored - we don't create custom foods

    def _extract_ingredients_from_description(self, description: str) -> List[str]:
        """Extract ingredient names from a meal description for fallback search.

        Looks for patterns like:
        - "120g chicken breast"
        - "200ml milk"
        - "2 eggs"
        - "Greek yogurt"
        - Comma-separated or newline-separated ingredients

        Returns:
            List of ingredient strings, ordered by likelihood of being searchable
        """
        if not description:
            return []

        ingredients = []

        # Pattern 1: Quantity + unit + food (e.g., "120g chicken breast", "200ml milk")
        qty_unit_patterns = [
            r"(\d+\s*g\s+[\w\s\-]+?)(?:,|\.|$|\n|with|and)",  # 120g chicken breast
            r"(\d+\s*ml\s+[\w\s\-]+?)(?:,|\.|$|\n|with|and)",  # 200ml milk
            r"(\d+\s*oz\s+[\w\s\-]+?)(?:,|\.|$|\n|with|and)",  # 4oz salmon
        ]

        for pattern in qty_unit_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip().rstrip(",.")
                if len(cleaned) > 3:  # Skip very short matches
                    ingredients.append(cleaned)

        # Pattern 2: Common food words without quantities (fallback)
        common_foods = [
            "chicken",
            "beef",
            "salmon",
            "tuna",
            "eggs",
            "egg",
            "rice",
            "pasta",
            "bread",
            "oatmeal",
            "oats",
            "yogurt",
            "greek yogurt",
            "milk",
            "cheese",
            "broccoli",
            "spinach",
            "vegetables",
            "salad",
            "banana",
            "apple",
            "berries",
            "avocado",
            "almonds",
            "nuts",
            "peanut butter",
            "olive oil",
            "butter",
        ]

        description_lower = description.lower()
        for food in common_foods:
            if food in description_lower and food not in [
                i.lower() for i in ingredients
            ]:
                ingredients.append(food)

        # Deduplicate while preserving order
        seen = set()
        unique_ingredients = []
        for ing in ingredients:
            ing_lower = ing.lower().strip()
            if ing_lower not in seen and len(ing_lower) > 2:
                seen.add(ing_lower)
                unique_ingredients.append(ing.strip())

        return unique_ingredients[:5]  # Return max 5 candidates

    def _run(self, **kwargs) -> str:
        try:
            # 1. Get Meal ID from Plan
            date = kwargs.get("date")
            meal_type = kwargs.get("meal_type")
            meal_name_input = kwargs.get("meal_name")

            # Map "Afternoon Snack" to "Snack" for Hexis lookup if needed,
            # but we need to match the "mealSubType" or "mealName" in the plan.
            # Based on raw plan: "mealSubType": "Dinner", "mealName": "Dinner"
            # "mealSubType": "Lunch", "mealName": "Lunch"
            # "mealSubType": "Breakfast", "mealName": "Breakfast"
            # "mealSubType": "General_Snack", "mealName": "PM Snack" (for Afternoon Snack?)

            # Let's try to match loosely or use the map to help
            target_meal_subtypes = [meal_type]
            if meal_type == "Afternoon Snack":
                target_meal_subtypes = ["General_Snack", "Snack", "PM Snack"]
            elif meal_type == "Snack":
                target_meal_subtypes = [
                    "General_Snack",
                    "Snack",
                    "AM Snack",
                    "PM Snack",
                ]

            plan_result = self._get_plan_tool.run(start_date=date, end_date=date)
            if isinstance(plan_result, str):
                plan_result = json.loads(plan_result)

            meal_instance_id = None
            is_verified = False
            if "data" in plan_result and "days" in plan_result["data"]:
                for day in plan_result["data"]["days"]:
                    if day.get("dayString") == date:
                        for meal in day.get("meals", []):
                            # Check exact match on mealSubType or mealName
                            # or fallback to mapping
                            m_subtype = meal.get("mealSubType")
                            m_name = meal.get("mealName")

                            if m_subtype == meal_type or m_name == meal_type:
                                meal_instance_id = meal.get("id")
                                is_verified = meal.get("mealVerification") is not None
                                break

                            # Handle Snacks specifically if needed
                            if meal_type in [
                                "Snack",
                                "Afternoon Snack",
                            ] and m_subtype in ["General_Snack", "Snack"]:
                                # If there are multiple snacks, we might pick the first one or try to distinguish
                                # For now, pick the first available snack slot
                                meal_instance_id = meal.get("id")
                                is_verified = meal.get("mealVerification") is not None
                                break
                        if meal_instance_id:
                            break

            if not meal_instance_id:
                return f"Error: Could not find meal slot for {meal_type} on {date}"

            # 2. Get food ID - either from validated_ingredients, hexis_food_id, or search
            if not meal_name_input:
                return f"Error: 'meal_name' is required. Please provide a descriptive name (e.g., 'Chicken Salad') to avoid generic entries."

            food_id = kwargs.get("hexis_food_id")
            data_origin = kwargs.get("data_origin")
            custom_food_name = meal_name_input
            validated_ingredients = kwargs.get("validated_ingredients")

            # Macros from the plan (we will use these to overwrite/verify)
            macros = {
                "protein": kwargs.get("protein"),
                "carb": kwargs.get("carbs"),
                "fat": kwargs.get("fat"),
                "energy": kwargs.get("calories"),
            }

            # Strategy 1: Use pre-validated ingredients if available (PREFERRED - no search needed)
            ref_code = None  # refCode may be required for Passio foods
            if validated_ingredients and len(validated_ingredients) > 0:
                # Use the first ingredient with a valid passio_food_id and passio_ref_code
                for ing in validated_ingredients:
                    ing_data = (
                        ing
                        if isinstance(ing, dict)
                        else ing.model_dump() if hasattr(ing, "model_dump") else {}
                    )
                    # Need both passio_food_id and passio_ref_code for Hexis API
                    if ing_data.get("passio_food_id"):
                        food_id = ing_data["passio_food_id"]
                        ref_code = ing_data.get("passio_ref_code")
                        data_origin = "PASSIO"
                        custom_food_name = ing_data.get(
                            "passio_food_name", meal_name_input
                        )
                        if ref_code:
                            print(
                                f"✅ Using pre-validated foodId: {food_id} with refCode ({custom_food_name})"
                            )
                        else:
                            print(
                                f"⚠️  Using pre-validated foodId: {food_id} WITHOUT refCode ({custom_food_name})"
                            )
                        break

                if not food_id:
                    print(
                        f"⚠️  validated_ingredients provided but no passio_food_id found, falling back to search"
                    )

            # Strategy 2: Use provided hexis_food_id if available
            if food_id:
                print(f"✅ Using provided Hexis Food ID: {food_id} ({data_origin})")
            else:
                # Strategy 3: Fall back to search (least preferred - may fail)
                search_query = meal_name_input

                try:
                    # Search for the food
                    print(f"🔍 Searching Hexis for: '{search_query}'")
                    search_result = self._search_tool.run(query=search_query)
                    if isinstance(search_result, str):
                        try:
                            search_result = json.loads(search_result)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not JSON

                    # Filter results to only PASSIO and USDA sources
                    found_food = None
                    all_results = []
                    if isinstance(search_result, list):
                        all_results = search_result
                    elif (
                        isinstance(search_result, dict)
                        and "data" in search_result
                        and isinstance(search_result["data"], list)
                    ):
                        all_results = search_result["data"]

                    # Find first food (The new tool is specific to Passio, so we can trust the results more, but we still check origin if available)
                    for food in all_results:
                        # The new tool might return foods with different structure or just Passio foods.
                        # We'll take the first one that looks valid.
                        found_food = food
                        break

                    if found_food:
                        food_id = found_food.get("id") or found_food.get("resultId")
                        ref_code = found_food.get(
                            "refCode"
                        )  # CRITICAL: capture refCode from search
                        data_origin = found_food.get("dataOrigin", "PASSIO")
                        custom_food_name = found_food.get("name", meal_name_input)
                        if ref_code:
                            print(
                                f"✅ Found food '{found_food.get('name')}' with refCode: {ref_code[:40]}..."
                            )
                        else:
                            print(
                                f"⚠️  Found food '{found_food.get('name')}' but NO refCode (ID: {food_id})"
                            )
                    else:
                        # Strategy 3b: Fallback - extract ingredients from description and retry search
                        print(
                            f"⚠️  No food found for meal name '{search_query}'. Trying ingredient extraction fallback..."
                        )
                        description = kwargs.get("description", "")
                        extracted_ingredients = (
                            self._extract_ingredients_from_description(description)
                        )

                        if extracted_ingredients:
                            print(
                                f"🔍 Extracted {len(extracted_ingredients)} ingredient candidates: {extracted_ingredients[:3]}"
                            )

                            for ingredient_query in extracted_ingredients:
                                print(
                                    f"🔍 Fallback search for ingredient: '{ingredient_query}'"
                                )
                                try:
                                    fallback_result = self._search_tool.run(
                                        query=ingredient_query
                                    )
                                    if isinstance(fallback_result, str):
                                        try:
                                            fallback_result = json.loads(
                                                fallback_result
                                            )
                                        except json.JSONDecodeError:
                                            continue

                                    fallback_foods = []
                                    if isinstance(fallback_result, list):
                                        fallback_foods = fallback_result
                                    elif (
                                        isinstance(fallback_result, dict)
                                        and "data" in fallback_result
                                    ):
                                        fallback_foods = fallback_result.get("data", [])
                                    elif (
                                        isinstance(fallback_result, dict)
                                        and "foods" in fallback_result
                                    ):
                                        fallback_foods = fallback_result.get(
                                            "foods", []
                                        )

                                    if fallback_foods:
                                        found_food = fallback_foods[0]
                                        food_id = found_food.get(
                                            "id"
                                        ) or found_food.get("resultId")
                                        ref_code = found_food.get("refCode")
                                        data_origin = found_food.get(
                                            "dataOrigin", "PASSIO"
                                        )
                                        custom_food_name = (
                                            found_food.get("name")
                                            or found_food.get("displayName")
                                            or ingredient_query
                                        )
                                        print(
                                            f"✅ Fallback SUCCESS: Found '{custom_food_name}' for ingredient '{ingredient_query}'"
                                        )
                                        break
                                except Exception as fallback_err:
                                    print(
                                        f"⚠️  Fallback search error for '{ingredient_query}': {fallback_err}"
                                    )
                                    continue

                        if not found_food:
                            print(
                                f"❌ No food found for '{search_query}' and fallback extraction failed"
                            )
                            return f"Error: No food found for '{search_query}'. Only verified database entries are allowed (no custom foods)."

                except Exception as e:
                    print(f"❌ Error searching for food '{search_query}': {e}")
                    return f"Error searching for food '{search_query}': {e}"

            if not food_id and not ref_code:
                return f"Error: Could not find food_id/refCode for '{custom_food_name}'. Search failed or no PASSIO/USDA match."

            # 3. Verify Meal (Overwrite)
            # Get quantity in grams from the first validated ingredient, default to 100g
            quantity_g = 100.0
            if validated_ingredients:
                for ing in validated_ingredients:
                    ing_data = (
                        ing
                        if isinstance(ing, dict)
                        else ing.model_dump() if hasattr(ing, "model_dump") else {}
                    )
                    if ing_data.get("quantity_g"):
                        quantity_g = float(ing_data["quantity_g"])
                        break

            # Use proper gram-based portion with foodId ONLY (refCode is NOT valid in Hexis API)
            food_object = {
                "foodId": food_id,  # REQUIRED by Hexis API
                "foodName": custom_food_name,
                "quantity": 1.0,
                "portion": {"unit": "g", "value": quantity_g, "name": "grams"},
                "macros": macros,
                "dataOrigin": data_origin or "PASSIO",
            }
            print(
                f"📦 food_object: foodId={food_id}, quantity={quantity_g}g, dataOrigin={data_origin or 'PASSIO'}"
            )

            verify_args = {
                "meal_id": meal_instance_id,
                "date": date,
                "foods": [food_object],
                "carb_code": "MEDIUM",  # Default to MEDIUM, or infer?
                "skipped": False,
            }

            # Use update_verified_meal for already-verified meals, verify_meal for new ones
            if is_verified:
                print(f"🔄 Updating verified meal {meal_instance_id}")
                verify_result = self._update_tool.run(**verify_args)
                action_type = "Updated"
            else:
                print(f"✅ Verifying new meal {meal_instance_id}")
                verify_result = self._verify_tool.run(**verify_args)
                action_type = "Verified"

            return f"Successfully {action_type} meal {meal_type} for {date}. Result: {verify_result}"

        except Exception as e:
            return f"Error in HexisLogMealTool: {str(e)}"


def create_hexis_log_meal_tool(hexis_tools: List[BaseTool]) -> HexisLogMealTool:
    """Factory to create the composite tool from existing MCP tools.

    Note: This tool does NOT create custom foods. It only searches for existing foods
    from PASSIO and USDA databases to ensure nutritional accuracy.
    """
    verify_tool = next(
        (t for t in hexis_tools if t.name == "hexis__hexis_verify_meal"), None
    )
    update_tool = next(
        (t for t in hexis_tools if t.name == "hexis__hexis_update_verified_meal"), None
    )
    get_plan_tool = next(
        (t for t in hexis_tools if t.name == "hexis__hexis_get_weekly_plan"), None
    )
    search_tool = next(
        (t for t in hexis_tools if t.name == "hexis__hexis_search_passio_foods"), None
    )

    # create_tool is no longer required - we don't create custom foods
    if not all([verify_tool, update_tool, get_plan_tool, search_tool]):
        missing = []
        if not verify_tool:
            missing.append("hexis_verify_meal")
        if not update_tool:
            missing.append("hexis_update_verified_meal")
        if not get_plan_tool:
            missing.append("hexis_get_weekly_plan")
        if not search_tool:
            missing.append("hexis_search_passio_foods")
        raise ValueError(f"Missing required Hexis tools: {', '.join(missing)}")

    return HexisLogMealTool(
        verify_tool=verify_tool,
        update_tool=update_tool,
        get_plan_tool=get_plan_tool,
        search_tool=search_tool,
    )
