"""Test script for the meal planning crew."""
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crew_mealy import MealPlanningCrew


def _preview_text(value: Any, fallback: str = "N/A") -> str:
    """Return a short preview-friendly string for logging."""
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except TypeError:
            pass
    return str(value)


def test_meal_planning():
    """Test the meal planning crew with a sample week."""
    # Use the requested week (user confirmed Hexis has data for this week)
    week_start_date = "2025-11-17"

    days_env = os.getenv("MEAL_PLAN_DAYS")
    try:
        days_to_generate = int(days_env) if days_env else 7
    except ValueError:
        print(f"⚠️  Invalid MEAL_PLAN_DAYS value '{days_env}' - defaulting to 7 days\n")
        days_to_generate = 7

    if days_to_generate < 1:
        print(f"⚠️  Requested days ({days_to_generate}) is less than 1 - defaulting to 1 day\n")
        days_to_generate = 1
    elif days_to_generate > 7:
        print(f"⚠️  Requested days ({days_to_generate}) exceeds 7 - capping to 7 days\n")
        days_to_generate = 7

    print(f"\n{'='*70}")
    print(f"🧪 Testing Meal Planning Crew")
    print(f"{'='*70}")
    print(f"Week start date: {week_start_date}")
    print(f"Days requested: {days_to_generate}")
    print(f"{'='*70}\n")

    try:
        # Initialize crew
        print("⏳ Initializing crew...\n")
        crew = MealPlanningCrew()

        # Generate meal plan
        print("⏳ Generating meal plan (this may take 2-10 minutes)...\n")
        result = crew.generate_meal_plan(week_start_date, days_to_generate=days_to_generate)

        # Check for errors
        if "error" in result:
            print(f"\n❌ ERROR: {result['error']}")
            print(f"Failed at step: {result.get('step', 'unknown')}\n")
            return False

        # Display results
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS: Meal Planning Completed")
        print(f"{'='*70}\n")

        # Hexis Analysis Summary
        hexis = result.get("hexis_analysis", {})
        print(f"📊 HEXIS ANALYSIS:")
        print(f"   Training Load: {_preview_text(hexis.get('training_load_summary'))[:80]}...")
        print(f"   Recovery: {_preview_text(hexis.get('recovery_status'))[:80]}...")
        print()

        # Nutrition Plan Summary
        nutrition = result.get("nutrition_plan", {})
        print(f"📅 NUTRITION PLAN:")
        print(f"   Week: {nutrition.get('week_start_date')} to {nutrition.get('week_end_date')}")
        print(f"   Summary: {_preview_text(nutrition.get('weekly_summary'))[:80]}...")
        print()

        # Meal Plan Summary
        meal_plan = result.get("meal_plan", {})
        daily_plans = meal_plan.get("daily_plans", [])
        print(f"👨‍🍳 MEAL PLAN:")
        print(f"   Days planned: {len(daily_plans)}")
        total_meals = sum(len(day.get("meals", [])) for day in daily_plans)
        print(f"   Total meals: {total_meals}")
        print()

        # Validation Summary
        validation = result.get("validation", {})
        print(f"🔍 VALIDATION:")
        print(f"   Approved: {'✅ Yes' if validation.get('approved') else '❌ No'}")
        print(f"   Variety Score: {validation.get('variety_score', 'N/A')}")
        print(f"   Practicality: {validation.get('practicality_score', 'N/A')}")
        if validation.get("issues_found"):
            print(f"   Issues: {len(validation['issues_found'])}")
        print()

        # Integration Summary
        integration = result.get("integration", {})
        print(f"🔗 MEALY INTEGRATION:")
        print(f"   Meals created: {integration.get('total_meals_created', 0)}")
        if integration.get("mealy_week_url"):
            print(f"   View meals: {integration['mealy_week_url']}")
        print(f"   Summary: {_preview_text(integration.get('summary'))[:80]}...")
        print()

        # Sample meal (first breakfast)
        if daily_plans and len(daily_plans) > 0:
            first_day = daily_plans[0]
            meals = first_day.get("meals", [])
            if meals:
                first_meal = meals[0]
                print(f"📝 SAMPLE MEAL ({first_day['day_name']} Breakfast):")
                print(f"   Name: {first_meal.get('meal_name', 'N/A')}")
                print(f"   Calories: {first_meal.get('calories', 0)} kcal")
                print(f"   Macros: {first_meal.get('protein_g', 0)}P / {first_meal.get('carbs_g', 0)}C / {first_meal.get('fat_g', 0)}F")
                print(f"   Prep time: {first_meal.get('preparation_time_min', 0)} min")
                print()

        print(f"{'='*70}")
        print(f"✅ Test completed successfully!")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\n❌ ERROR during test:")
        print(f"{str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_meal_planning()
    sys.exit(0 if success else 1)
