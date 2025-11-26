
from typing import Type, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json

class LogMealSchema(BaseModel):
    """Input for HexisLogMealTool."""
    meal_type: str = Field(..., description="Type of meal (Breakfast, Lunch, Dinner, Snack, Afternoon Snack)")
    day_name: str = Field(..., description="Name of the day (e.g., Monday)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    meal_name: Optional[str] = Field(None, description="Descriptive name of the meal (e.g., 'Chicken Salad', 'Oatmeal with Berries'). IMPORTANT for search.")
    description: Optional[str] = Field(None, description="Detailed description or ingredients (optional, helps with search context)")
    calories: float = Field(..., description="Total calories")
    protein: float = Field(..., description="Total protein in grams")
    carbs: float = Field(..., description="Total carbs in grams")
    fat: float = Field(..., description="Total fat in grams")

class HexisLogMealTool(BaseTool):
    name: str = "hexis_log_meal"
    description: str = (
        "Logs a meal to Hexis. Creates a custom food and verifies the meal. "
        "Handles overwriting existing meals if present."
    )
    args_schema: Type[BaseModel] = LogMealSchema
    
    _create_tool: BaseTool
    _verify_tool: BaseTool
    _update_tool: BaseTool
    _get_plan_tool: BaseTool
    _search_tool: BaseTool

    def __init__(self, verify_tool, update_tool, get_plan_tool, search_tool, create_tool):
        super().__init__()
        self._verify_tool = verify_tool
        self._update_tool = update_tool
        self._get_plan_tool = get_plan_tool
        self._search_tool = search_tool
        self._create_tool = create_tool

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
                 target_meal_subtypes = ["General_Snack", "Snack", "AM Snack", "PM Snack"]

            plan_result = self._get_plan_tool.run(start_date=date, end_date=date)
            if isinstance(plan_result, str):
                plan_result = json.loads(plan_result)
            
            meal_instance_id = None
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
                                break
                            
                            # Handle Snacks specifically if needed
                            if meal_type in ["Snack", "Afternoon Snack"] and m_subtype in ["General_Snack", "Snack"]:
                                # If there are multiple snacks, we might pick the first one or try to distinguish
                                # For now, pick the first available snack slot
                                meal_instance_id = meal.get("id")
                                break
                        if meal_instance_id:
                            break
            
            if not meal_instance_id:
                return f"Error: Could not find meal slot for {meal_type} on {date}"

            # 2. Search for Existing Food (Strict Mode)
            # Enforce meal_name is provided to avoid generic names like "Dinner - Tuesday"
            if not meal_name_input:
                return f"Error: 'meal_name' is required. Please provide a descriptive name (e.g., 'Chicken Salad') to avoid generic entries."

            search_query = meal_name_input
            custom_food_name = meal_name_input
            
            food_id = None
            data_origin = None
            
            # Macros from the plan (we will use these to overwrite/verify)
            macros = {
                "protein": kwargs.get("protein"),
                "carb": kwargs.get("carbs"),
                "fat": kwargs.get("fat"),
                "energy": kwargs.get("calories")
            }

            try:
                # Search for the food
                print(f"🔍 Searching Hexis for: '{search_query}'")
                search_result = self._search_tool.run(query=search_query)
                if isinstance(search_result, str):
                    try:
                        search_result = json.loads(search_result)
                    except json.JSONDecodeError:
                        pass # Keep as string if not JSON
                
                found_food = None
                if isinstance(search_result, list) and len(search_result) > 0:
                    found_food = search_result[0]
                elif isinstance(search_result, dict) and "data" in search_result and isinstance(search_result["data"], list) and len(search_result["data"]) > 0:
                    found_food = search_result["data"][0]
                
                if found_food:
                    food_id = found_food.get("id")
                    data_origin = found_food.get("dataOrigin", "CUSTOM_FOOD")
                    print(f"✅ Found existing food '{found_food.get('name')}' (ID: {food_id})")
                else:
                    print(f"⚠️  Food '{search_query}' not found in Hexis.")
                    # Fallback to custom food creation

            except Exception as e:
                print(f"❌ Error searching for food '{search_query}': {e}")
                # Fallback to custom food creation

            if not food_id:
                # Fallback: Create Custom Food
                print(f"🍱 Creating custom food: '{custom_food_name}'")
                try:
                    create_result = self._create_tool.run(
                        food_name=custom_food_name,
                        energy=macros["energy"],
                        protein=macros["protein"],
                        carb=macros["carb"],
                        fat=macros["fat"],
                        serving_value=1,
                        serving_unit="serving",
                        serving_name="serving"
                    )
                    
                    if isinstance(create_result, str):
                        try:
                            create_result = json.loads(create_result)
                        except json.JSONDecodeError:
                            pass
                    
                    if isinstance(create_result, dict) and "id" in create_result:
                        food_id = create_result["id"]
                        data_origin = "CUSTOM_FOOD"
                        print(f"✅ Created custom food '{custom_food_name}' (ID: {food_id})")
                    else:
                        return f"Error: Failed to create custom food '{custom_food_name}'. Result: {create_result}"
                        
                except Exception as e:
                    return f"Error creating custom food '{custom_food_name}': {e}"

            # 3. Verify Meal (Overwrite)
            food_object = {
                "foodId": food_id,
                "foodName": custom_food_name, # Use the descriptive name for the log entry
                "quantity": 1.0, # 1 serving
                "portion": {
                    "unit": "serving",
                    "value": 1.0,
                    "name": "serving"
                },
                "macros": macros,
                "dataOrigin": data_origin
            }

            verify_args = {
                "meal_id": meal_instance_id,
                "date": date,
                "foods": [food_object],
                "carb_code": "MEDIUM", # Default to MEDIUM, or infer?
                "skipped": False
            }
            
            # Call verify tool
            verify_result = self._verify_tool.run(**verify_args)
            
            return f"Successfully logged meal {meal_type} for {date}. Verify result: {verify_result}"

        except Exception as e:
            return f"Error in HexisLogMealTool: {str(e)}"

def create_hexis_log_meal_tool(hexis_tools: List[BaseTool]) -> HexisLogMealTool:
    """Factory to create the composite tool from existing MCP tools."""
    verify_tool = next((t for t in hexis_tools if t.name == "hexis__hexis_verify_meal"), None)
    update_tool = next((t for t in hexis_tools if t.name == "hexis__hexis_update_verified_meal"), None)
    get_plan_tool = next((t for t in hexis_tools if t.name == "hexis__hexis_get_weekly_plan"), None)
    search_tool = next((t for t in hexis_tools if t.name == "hexis__hexis_search_foods"), None)
    create_tool = next((t for t in hexis_tools if t.name == "hexis__hexis_create_custom_food"), None)

    if not all([verify_tool, update_tool, get_plan_tool, search_tool, create_tool]):
        missing = []
        if not verify_tool: missing.append("hexis_verify_meal")
        if not update_tool: missing.append("hexis_update_verified_meal")
        if not get_plan_tool: missing.append("hexis_get_weekly_plan")
        if not search_tool: missing.append("hexis_search_foods")
        if not create_tool: missing.append("hexis_create_custom_food")
        raise ValueError(f"Missing required Hexis tools: {', '.join(missing)}")

    return HexisLogMealTool(
        verify_tool=verify_tool,
        update_tool=update_tool,
        get_plan_tool=get_plan_tool,
        search_tool=search_tool,
        create_tool=create_tool
    )
