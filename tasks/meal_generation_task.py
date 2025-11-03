"""Task for generating complete weekly meal plans."""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task

from schemas import WeeklyMealPlan


def create_meal_generation_task(
    agent: Any,
    weekly_nutrition_plan: Dict[str, Any]
) -> Task:
    """
    Create a task for generating a complete weekly meal plan.

    Args:
        agent: The agent responsible for this task
        weekly_nutrition_plan: The structured nutrition plan (WeeklyNutritionPlan as dict)

    Returns:
        Configured Task instance
    """
    nutrition_json = json.dumps(weekly_nutrition_plan, indent=2)

    description = f"""
    Generate a complete weekly meal plan with delicious, balanced meals that hit
    nutritional targets while maximizing variety and culinary appeal.

    WEEKLY NUTRITION PLAN INPUT:
    {nutrition_json}

    YOUR MISSION:

    1. OPTIONALLY USE MEALY MCP TOOLS
        - Call Mealy tools to fetch user preferences (favorite ingredients, cuisines)
        - Check for dietary restrictions or allergies
        - Review past successful meals for inspiration
        (Skip if MCP tools not available - proceed with general meal generation)

    2. GENERATE MEALS FOR EACH DAY
        For each day (Monday through Sunday), create:

        A. BREAKFAST (~25-30% of daily calories)
           - Appropriate for morning consumption
           - Consider workout timing (light if pre-workout, substantial if post-workout)
           - Include protein, complex carbs, healthy fats
           - Quick preparation (<15 min) or meal-prep friendly

        B. LUNCH (~30-35% of daily calories)
           - Balanced meal with all macros
           - Portable/packable if needed for work
           - Satisfying and sustaining

        C. DINNER (~30-35% of daily calories)
           - Often the main meal of the day
           - Recovery-focused if post-workout
           - Time for more elaborate preparation (acceptable)

        D. SNACKS (~10-20% of daily calories, 1-2 snacks)
           - Strategic timing (pre-workout fuel, post-workout recovery, or between meals)
           - Easy to prepare and portable
           - Purpose-driven (energy boost, protein hit, etc.)

    3. ENSURE VARIETY ACROSS THE WEEK
        - Rotate protein sources: chicken, fish, beef, pork, eggs, legumes, tofu, dairy
        - Vary cooking methods: grilled, baked, sautéed, steamed, roasted, raw
        - Mix cuisines: French, Mediterranean, Asian fusion, Latin American, Middle Eastern
        - Use seasonal ingredients
        - Different vegetables and fruits each day
        - Variety in grains: rice, quinoa, pasta, bread, oats, potatoes

    4. NUTRITIONAL PRECISION
        For each meal, calculate and specify:
        - Total calories
        - Protein (g)
        - Carbohydrates (g)
        - Fat (g)
        - Preparation time (minutes)

        Adjust portions to hit daily targets:
        - Sum all meals for the day
        - Verify totals match the daily target (±50 calories acceptable)
        - Adjust final meal portions if needed to balance

    5. PROVIDE COMPLETE MEAL DETAILS
        For each meal, include:
        - meal_type: "Breakfast" / "Lunch" / "Dinner" / "Snack"
        - meal_name: Creative, appetizing name (e.g., "Mediterranean Chicken Bowl")
        - description: Brief description with key ingredients and appeal
        - calories, protein_g, carbs_g, fat_g: Calculated macros
        - preparation_time_min: Realistic time estimate
        - ingredients: Complete list with quantities
          Example: ["200g chicken breast", "1 cup quinoa", "2 cups mixed greens", ...]
        - recipe_notes: Brief cooking instructions or assembly notes
          Example: "Grill chicken 6-8 min per side. Cook quinoa according to package. Assemble bowl with greens, quinoa, sliced chicken, cherry tomatoes, cucumber, feta, and lemon vinaigrette."

    6. CREATE CONSOLIDATED SHOPPING LIST
        - Extract all ingredients from the week
        - Organize by category (proteins, grains, vegetables, fruits, dairy, pantry)
        - Aggregate quantities where possible
        - Make it practical for grocery shopping

    7. PROVIDE MEAL PREP TIPS
        - Identify batch cooking opportunities (cook grains for multiple days)
        - Suggest pre-prep tasks (chop vegetables, marinate proteins)
        - Storage guidance (what keeps well, what to prepare fresh)
        - Time-saving strategies

    IMPORTANT GUIDELINES:

    - NEVER repeat the same meal twice in the week
    - Use common, accessible ingredients (no exotic items)
    - Keep breakfast simple and quick (or prep ahead)
    - Make recipes realistic for home cooking
    - Prioritize whole foods over processed alternatives
    - Include herbs/spices for flavor without calories
    - Ensure adequate hydration-rich foods (fruits, vegetables)
    - Consider athletic performance needs:
      * Pre-workout: Easily digestible, moderate-high carb, low fat
      * Post-workout: Protein + carbs, moderate fat OK
      * Rest days: Balanced, moderate portions

    EXAMPLE MEAL STRUCTURE:
    {{
      "meal_type": "Dinner",
      "meal_name": "Grilled Salmon with Roasted Sweet Potato and Broccoli",
      "description": "Omega-3 rich salmon fillet with caramelized sweet potato wedges and garlic-herb roasted broccoli",
      "calories": 680,
      "protein_g": 45,
      "carbs_g": 55,
      "fat_g": 25,
      "preparation_time_min": 35,
      "ingredients": [
        "180g salmon fillet",
        "250g sweet potato (1 medium)",
        "200g broccoli florets",
        "2 tsp olive oil",
        "2 cloves garlic, minced",
        "1 lemon, juiced",
        "Fresh dill",
        "Salt, pepper, paprika"
      ],
      "recipe_notes": "Preheat oven to 200°C. Cut sweet potato into wedges, toss with 1 tsp olive oil and paprika, roast 25 min. At 15 min mark, add broccoli with garlic. Grill salmon 4-5 min per side. Serve with lemon and dill."
    }}

    OUTPUT CONTRACT:
    - Respond with valid JSON matching the WeeklyMealPlan schema
    - Do not wrap JSON in markdown fences
    - Include 3 main meals + 1-2 snacks per day for all 7 days
    - Ensure daily totals match nutrition targets (±50 cal tolerance)
    - Provide complete shopping list and meal prep tips
    - Make meals DELICIOUS and PRACTICAL
    """

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON adhering to the WeeklyMealPlan schema with complete daily meal plans, shopping list, and prep tips",
        output_json=WeeklyMealPlan,
    )
