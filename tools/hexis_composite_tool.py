import json
from typing import Any, Dict, List, Optional, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

class LogMealSchema(BaseModel):
    day_name: str = Field(..., description="Day of the week (e.g., Monday)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    meal_type: str = Field(..., description="Type of meal (Breakfast, Lunch, Dinner, Snack)")
    meal_name: str = Field(..., description="Name of the meal")
    calories: float = Field(..., description="Energy in kcal")
    protein: float = Field(..., description="Protein in g")
    carbs: float = Field(..., description="Carbs in g")
    fat: float = Field(..., description="Fat in g")

class HexisLogMealTool(BaseTool):
    args_schema: Type[BaseModel] = Field(description="Schema for tool arguments")
    _create_tool: Any = PrivateAttr()
    _verify_tool: Any = PrivateAttr()

    def _run(self, **kwargs) -> str:
        try:
            # 1. Create Custom Food
            food_name = f"{kwargs.get('meal_type')} - {kwargs.get('day_name')}"
            create_args = {
                "food_name": food_name,
                "serving_value": 1.0,
                "serving_name": "serving",
                "serving_unit": "serving",
                "protein": kwargs.get("protein"),
                "carb": kwargs.get("carbs"),
                "fat": kwargs.get("fat"),
                "energy": kwargs.get("calories")
            }
            
            # Call create tool
            create_result_raw = self._create_tool._run(**create_args)
            
            # Parse result
            create_result = None
            if isinstance(create_result_raw, str):
                try:
                    create_result = json.loads(create_result_raw)
                except json.JSONDecodeError:
                    return f"Error: Failed to parse create_custom_food output: {create_result_raw}"
            elif isinstance(create_result_raw, dict):
                create_result = create_result_raw
            else:
                return f"Error: Unexpected output type from create_custom_food: {type(create_result_raw)}"
                
            food_id = create_result.get("id")
            if not food_id:
                return f"Error: Failed to get food ID from create_custom_food. Output: {create_result}"

            # 2. Verify Meal
            # Determine carb code
            carbs = kwargs.get("carbs")
            carb_code = "HIGH" if carbs >= 60 else ("MEDIUM" if carbs >= 30 else "LOW")
            
            # Map meal type to ID
            meal_type_map = {
                "Breakfast": "BREAKFAST",
                "Lunch": "LUNCH",
                "Dinner": "DINNER",
                "Snack": "SNACK",
                "Afternoon Snack": "SNACK"
            }
            meal_id = meal_type_map.get(kwargs.get("meal_type"), "SNACK")
            
            verify_args = {
                "meal_id": meal_id,
                "date": kwargs.get("date"),
                "foods": [create_result], # Pass the full object as required by the strategy
                "carb_code": carb_code,
                "skipped": False
            }
            
            verify_result_raw = self._verify_tool._run(**verify_args)
            
            # Parse verify result for cleaner output
            verify_result = verify_result_raw
            if isinstance(verify_result_raw, str):
                try:
                    verify_result = json.loads(verify_result_raw)
                except json.JSONDecodeError:
                     pass

            return json.dumps({
                "status": "success",
                "day_name": kwargs.get("day_name"),
                "meal_type": kwargs.get("meal_type"),
                "hexis_id": food_id,
                "verification_result": verify_result
            })

        except Exception as e:
            return f"Error in hexis_log_meal: {str(e)}"

def create_hexis_log_meal_tool(hexis_tools: List[BaseTool]) -> HexisLogMealTool:
    """Factory to create the composite tool from existing MCP tools."""
    create_tool = next((t for t in hexis_tools if t.name == "hexis__hexis_create_custom_food"), None)
    verify_tool = next((t for t in hexis_tools if t.name == "hexis__hexis_verify_meal"), None)
    
    if not create_tool or not verify_tool:
        raise ValueError("Missing required Hexis tools (hexis_create_custom_food, hexis_verify_meal)")
        
    tool = HexisLogMealTool(
        name="hexis_log_meal",
        description="Log a meal to Hexis by creating a custom food and verifying it in one step.",
        args_schema=LogMealSchema
    )
    tool._create_tool = create_tool
    tool._verify_tool = verify_tool
    return tool
