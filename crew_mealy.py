"""Main Crew definition for weekly meal plan generation."""
from __future__ import annotations

# CRITICAL: Initialize auth BEFORE importing CrewAI to ensure structured outputs work
import llm_auth_init  # noqa: F401

import concurrent.futures
import copy
import threading
import time
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
import pytz
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import json_repair  # Added for robust JSON parsing

from crewai import Crew, LLM, Process
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel, ValidationError

from agents import (
    create_hexis_analysis_agent,
    create_weekly_structure_agent,
    create_meal_generation_agent,
    create_meal_compilation_agent,
    create_nutritional_validation_agent,
    create_mealy_integration_agent,
    # Supervisor/Executor/Reviewer pattern for MEAL_GENERATION
    create_meal_planning_supervisor_agent,
    create_ingredient_validation_executor_agent,
    create_meal_recipe_reviewer_agent,
    # Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS
    create_hexis_data_supervisor_agent,
    create_hexis_data_executor_agent,
    create_hexis_analysis_reviewer_agent,
)
from schemas import (
    HexisWeeklyAnalysis,
    WeeklyNutritionPlan,
    DailyMealPlan,
    WeeklyMealPlan,
    NutritionalValidation,
    MealyIntegrationResult,
    # Supervisor/Executor/Reviewer inter-agent schemas for MEAL_GENERATION
    MealPlanTemplate,
    ValidatedIngredientsList,
    # Supervisor/Executor/Reviewer inter-agent schemas for HEXIS_ANALYSIS
    HexisDataRetrievalPlan,
    RawHexisData,
)
from tasks import (
    create_hexis_analysis_task,
    create_weekly_structure_task,
    create_meal_generation_task,
    create_meal_compilation_task,
    create_nutritional_validation_task,
    create_mealy_integration_task,
    # Supervisor/Executor/Reviewer pattern for MEAL_GENERATION
    create_meal_planning_supervisor_task,
    create_ingredient_validation_executor_task,
    create_meal_recipe_reviewer_task,
    # Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS
    create_hexis_data_supervisor_task,
    create_hexis_data_executor_task,
    create_hexis_analysis_reviewer_task,
    _extract_daily_meal_targets,  # Deterministic extraction for reliability
)

# NOTE: load_catalog_tool_names kept for compatibility but currently unused
from mcp_utils import build_mcp_references, load_catalog_tool_names
from mcp_auth_wrapper import MetaMCPAdapter
from llm_provider_rotation import create_llm_with_rotation, get_model_for_category
from validation_config import check_variety, MIN_UNIQUE_MEALS_RATIO
from mcp_tool_wrapper import (
    wrap_mcp_tools,
    wrap_mcp_tool,
    reset_hexis_tool_results,
    get_hexis_tool_results,
    capture_hexis_result,
)
from tools.hexis_composite_tool import create_hexis_log_meal_tool
from macro_calculator import (
    enrich_validated_ingredients,
    fetch_nutrition_for_ingredients,
    format_pre_calculated_summary,
)


# ==============================================================================
# Utility functions for deterministic day_name handling
# ==============================================================================
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _date_to_day_name(date_str: str) -> str:
    """Convert ISO date string to English day name deterministically.

    Args:
        date_str: Date in YYYY-MM-DD format (e.g., "2025-12-19")

    Returns:
        English day name (e.g., "Friday")

    Raises:
        ValueError: If date_str is not a valid ISO date
    """
    dt = datetime.fromisoformat(date_str)
    return DAY_NAMES[dt.weekday()]  # weekday() returns 0=Monday, 6=Sunday


def _fix_daily_target_day_names(nutrition_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Fix day_name values using deterministic date conversion.

    The LLM sometimes generates wrong day_name/date pairs (e.g., "Friday" for
    a date that is actually Saturday). This function overwrites day_name with
    the correct value computed from the date.

    Args:
        nutrition_plan: WeeklyNutritionPlan dict from LLM with daily_targets

    Returns:
        Same nutrition_plan with corrected day_names (modified in place)
    """
    fixes_applied = 0
    for target in nutrition_plan.get("daily_targets", []):
        date_str = target.get("date", "")
        if not date_str:
            continue
        try:
            expected_day_name = _date_to_day_name(date_str)
            actual_day_name = target.get("day_name", "")

            if actual_day_name != expected_day_name:
                print(
                    f"   🔧 Fix day_name: {date_str} was '{actual_day_name}' -> '{expected_day_name}'",
                    file=sys.stderr,
                )
                target["day_name"] = expected_day_name
                fixes_applied += 1
        except ValueError as e:
            print(
                f"   ⚠️  Could not parse date '{date_str}': {e}",
                file=sys.stderr,
            )

    if fixes_applied > 0:
        print(
            f"\n   ✅ Fixed {fixes_applied} day_name(s) in nutrition plan\n",
            file=sys.stderr,
        )

    return nutrition_plan


class MealPlanningCrew:
    """Crew for generating and validating weekly meal plans."""

    def __init__(self):
        """Initialize the crew with LLM and agents."""
        # Environment and auth already configured at module level via llm_auth_init

        # Get configuration
        base_url = os.getenv("OPENAI_API_BASE", "https://ccproxy.emottet.com/v1")
        base_url = os.getenv("OPENAI_API_BASE", "https://ccproxy.emottet.com/v1")

        # Use category names directly to enable automatic cascade fallback
        # If a specific model is set via env var, use it instead of the category
        # This enables: complex → intermediate → simple → fallback cascade on errors
        complex_model_name = os.getenv(
            "OPENAI_COMPLEX_MODEL_NAME",
            "complex",  # Use category instead of random model for automatic cascade
        )
        intermediate_model_name = os.getenv(
            "OPENAI_INTERMEDIATE_MODEL_NAME",
            "intermediate",  # Use category instead of random model
        )
        simple_model_name = os.getenv(
            "OPENAI_SIMPLE_MODEL_NAME",
            "simple",  # Use category instead of random model
        )

        # Get configured API key (already set by llm_auth_init)
        api_key = os.getenv("OPENAI_API_KEY", "dummy-key")

        # Create per-agent LLMs with optional overrides
        # This allows each agent to use a different model/endpoint for cost optimization
        # EXECUTOR agents (MCP tools) - has_tools=True excludes thinking models
        # Thinking models hallucinate tool calls (ReAct in content) instead of using tool_calls
        self.hexis_analysis_llm = self._create_agent_llm(
            agent_name="HEXIS_ANALYSIS",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=True,  # Agent uses MCP tools
        )

        self.meal_generation_llm = self._create_agent_llm(
            agent_name="MEAL_GENERATION",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=True,  # Agent uses MCP tools
        )

        # =====================================================================
        # Supervisor/Executor/Reviewer pattern for MEAL_GENERATION
        # Separates reasoning from tool execution to avoid thinking model issues
        # =====================================================================

        # MEAL_PLANNING_SUPERVISOR: Pure reasoning (no tools) - thinking models OK
        self.meal_planning_supervisor_llm = self._create_agent_llm(
            agent_name="MEAL_PLANNING_SUPERVISOR",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=False,  # Pure reasoning - thinking models allowed
        )

        # INGREDIENT_VALIDATION_EXECUTOR: Tool execution only - NO thinking models
        self.ingredient_validation_executor_llm = self._create_agent_llm(
            agent_name="INGREDIENT_VALIDATION_EXECUTOR",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=True,  # Uses hexis_search_passio_foods - NO thinking models
        )

        # MEAL_RECIPE_REVIEWER: Pure reasoning (no tools) - thinking models OK
        self.meal_recipe_reviewer_llm = self._create_agent_llm(
            agent_name="MEAL_RECIPE_REVIEWER",
            default_model=intermediate_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=False,  # Pure reasoning - thinking models allowed
        )

        # =====================================================================
        # Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS
        # Separates reasoning from tool execution to avoid thinking model issues
        # =====================================================================

        # HEXIS_DATA_SUPERVISOR: Pure reasoning (no tools) - thinking models OK
        self.hexis_data_supervisor_llm = self._create_agent_llm(
            agent_name="HEXIS_DATA_SUPERVISOR",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=False,  # Pure reasoning - thinking models allowed
        )

        # HEXIS_DATA_EXECUTOR: BYPASSED - Now using direct Python tool calls
        # The executor LLM was unreliable (hallucinated tool calls instead of executing them)
        # self.hexis_data_executor_llm = self._create_agent_llm(
        #     agent_name="HEXIS_DATA_EXECUTOR",
        #     default_model=simple_model_name,
        #     default_base=base_url,
        #     default_key=api_key,
        #     has_tools=True,  # Uses hexis_get_weekly_plan - NO thinking models
        # )

        # HEXIS_ANALYSIS_REVIEWER: Pure reasoning (no tools) - thinking models OK
        self.hexis_analysis_reviewer_llm = self._create_agent_llm(
            agent_name="HEXIS_ANALYSIS_REVIEWER",
            default_model=intermediate_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=False,  # Pure reasoning - thinking models allowed
        )

        # SUPERVISOR/REVIEWER agents (pure reasoning, no tools - thinking models OK)
        self.weekly_structure_llm = self._create_agent_llm(
            agent_name="WEEKLY_STRUCTURE",
            default_model=intermediate_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - Pure reasoning/planning
        )

        self.nutritional_validation_llm = self._create_agent_llm(
            agent_name="NUTRITIONAL_VALIDATION",
            default_model=intermediate_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - Pure reasoning/validation
        )

        # SIMPLE agents (JSON aggregation/formatting, no tools - thinking models OK)
        self.meal_compilation_llm = self._create_agent_llm(
            agent_name="MEAL_COMPILATION",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - Pure formatting
        )

        # MEALY_INTEGRATION - EXECUTOR with tools (CRITICAL)
        # This agent must call hexis_log_meal multiple times (once per meal: 8-28 calls)
        # has_tools=True excludes thinking models that hallucinate tool calls
        # Thinking models return tool_calls=None while simulating ReAct in content
        self.mealy_integration_llm = self._create_agent_llm(
            agent_name="MEALY_INTEGRATION",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=True,  # Agent uses MCP tools - THIS WAS THE BUG
        )

        # Initialize MCP adapters with MetaMCP authentication fix
        self.mcp_adapters = []
        self.mcp_tools = []

        mcp_api_key = os.getenv("MCP_API_KEY", "")
        require_mcp = os.getenv("REQUIRE_MCP", "true").lower() == "true"

        # Define MCP server URLs for Meal Planning Crew
        # Note: Mealy MCP removed - now using Hexis for meal integration
        mcp_servers = {
            "Food": os.getenv("FOOD_MCP_SERVER_URL", ""),
            "Intervals.icu": os.getenv("INTERVALS_MCP_SERVER_URL", ""),
            "Toolbox": os.getenv("TOOLBOX_MCP_SERVER_URL", ""),
        }

        # Filter out empty URLs
        active_servers = {name: url for name, url in mcp_servers.items() if url}

        if active_servers and mcp_api_key:
            print(f"\n🔗 Connecting to {len(active_servers)} MCP servers...\n", file=sys.stderr)

            for server_name, server_url in active_servers.items():
                try:
                    print(f"   Connecting to {server_name}...", file=sys.stderr)
                    adapter = MetaMCPAdapter(server_url, mcp_api_key, connect_timeout=30)
                    adapter.start()
                    self.mcp_adapters.append(adapter)
                    self.mcp_tools.extend(adapter.tools)
                    print(f"   ✅ {server_name}: {len(adapter.tools)} tools discovered", file=sys.stderr)
                except Exception as e:
                    error_msg = f"   ❌ {server_name}: Connection failed - {e}"
                    if require_mcp:
                        print(error_msg, file=sys.stderr)
                        raise ValueError(f"MCP connection failed for {server_name}: {e}")
                    else:
                        print(f"{error_msg} (continuing without this server)", file=sys.stderr)

            print(f"\n✅ MCP connection complete! Total tools discovered: {len(self.mcp_tools)}\n", file=sys.stderr)

            # Wrap MCP tools with input validation to fix malformed inputs from LLM
            print("🛡️  Wrapping MCP tools with input validation...\n", file=sys.stderr)
            self.mcp_tools = wrap_mcp_tools(self.mcp_tools)
        elif require_mcp:
            error_msg = (
                "\n❌ Error: MCP configuration missing. "
                "Please set MCP server URLs and MCP_API_KEY environment variables.\n"
                "Required: FOOD_MCP_SERVER_URL, INTERVALS_MCP_SERVER_URL\n"
            )
            print(error_msg, file=sys.stderr)
            raise ValueError("MCP configuration is required but not provided")
        else:
            print("\n⚠️  Warning: No MCP configuration. Agents will operate without live data.\n", file=sys.stderr)

        # Filter tools by type for different agents
        hexis_tools = [t for t in self.mcp_tools if "hexis" in t.name.lower()]
        food_data_tools = [t for t in self.mcp_tools if "food" in t.name.lower() and "hexis" not in t.name.lower()]
        intervals_tools = [t for t in self.mcp_tools if "intervals" in t.name.lower()]
        toolbox_tools = [t for t in self.mcp_tools if any(keyword in t.name.lower() for keyword in ["fetch", "time", "task"])]

        if hexis_tools:
            print(f"✅ Found {len(hexis_tools)} Hexis tools\n", file=sys.stderr)
        if food_data_tools:
            print(f"🍎 Found {len(food_data_tools)} Food Data Central tools\n", file=sys.stderr)
        if intervals_tools:
            print(f"📊 Found {len(intervals_tools)} Intervals.icu tools\n", file=sys.stderr)
        if toolbox_tools:
            print(f"🛠️  Found {len(toolbox_tools)} Toolbox tools\n", file=sys.stderr)

        # Create agents with MCP tools
        # Hexis Analysis Agent: needs Hexis tools only
        hexis_analysis_tools = hexis_tools
        self.hexis_analysis_agent = create_hexis_analysis_agent(
            self.hexis_analysis_llm, tools=hexis_analysis_tools if hexis_analysis_tools else None
        )

        # Weekly Structure Agent: no MCP tools needed (pure reasoning)
        self.weekly_structure_agent = create_weekly_structure_agent(self.weekly_structure_llm)

        # Meal Generation Agent: needs all food-related tools (LEGACY - kept for backwards compatibility)
        meal_generation_tools = hexis_tools + food_data_tools + toolbox_tools
        self.meal_generation_agent = create_meal_generation_agent(
            self.meal_generation_llm, tools=meal_generation_tools if meal_generation_tools else None
        )

        # =====================================================================
        # Supervisor/Executor/Reviewer agents for MEAL_GENERATION
        # This pattern separates reasoning from tool execution
        # =====================================================================

        # SUPERVISOR: Pure reasoning - designs meals (NO TOOLS)
        self.meal_planning_supervisor_agent = create_meal_planning_supervisor_agent(
            self.meal_planning_supervisor_llm
        )

        # EXECUTOR: Tool execution - validates ingredients (hexis_search_passio_foods)
        # Only needs the search tool, not all hexis tools
        # IMPORTANT: Use exact match to avoid selecting barcode or advanced variants
        self.passio_search_tool = next(
            (t for t in hexis_tools if t.name.lower() == "hexis__hexis_search_passio_foods"),
            None
        )
        executor_tools = [self.passio_search_tool] if self.passio_search_tool else []
        self.ingredient_validation_executor_agent = create_ingredient_validation_executor_agent(
            self.ingredient_validation_executor_llm,
            tools=executor_tools if executor_tools else None
        )
        if self.passio_search_tool:
            print(f"   ✅ Executor agent has {self.passio_search_tool.name} tool\n", file=sys.stderr)
        else:
            print(f"   ⚠️  Executor agent missing hexis_search_passio_foods tool!\n", file=sys.stderr)

        # REVIEWER: Pure reasoning - calculates macros and finalizes (NO TOOLS)
        self.meal_recipe_reviewer_agent = create_meal_recipe_reviewer_agent(
            self.meal_recipe_reviewer_llm
        )

        # =====================================================================
        # Supervisor/Executor/Reviewer agents for HEXIS_ANALYSIS
        # This pattern separates reasoning from tool execution
        # =====================================================================

        # SUPERVISOR: Pure reasoning - plans data retrieval (NO TOOLS)
        self.hexis_data_supervisor_agent = create_hexis_data_supervisor_agent(
            self.hexis_data_supervisor_llm
        )

        # EXECUTOR: Tool execution - retrieves Hexis data (hexis_get_weekly_plan)
        # Only needs the weekly plan tool
        # EXECUTOR: BYPASSED - Now using direct Python tool calls in generate_meal_plan()
        # The LLM executor was unreliable (hallucinated tool calls for batches 2-3)
        # hexis_weekly_plan_tool = next(
        #     (t for t in hexis_tools if "get_weekly_plan" in t.name.lower()),
        #     None
        # )
        # hexis_executor_tools = [hexis_weekly_plan_tool] if hexis_weekly_plan_tool else hexis_tools
        # self.hexis_data_executor_agent = create_hexis_data_executor_agent(
        #     self.hexis_data_executor_llm,
        #     tools=hexis_executor_tools if hexis_executor_tools else None
        # )
        # if hexis_weekly_plan_tool:
        #     print(f"   ✅ HEXIS Executor agent has hexis_get_weekly_plan tool\n", file=sys.stderr)
        # else:
        #     print(f"   ⚠️  HEXIS Executor using all Hexis tools (hexis_get_weekly_plan not found)\n", file=sys.stderr)

        # REVIEWER: Pure reasoning - analyzes data and creates final output (NO TOOLS)
        self.hexis_analysis_reviewer_agent = create_hexis_analysis_reviewer_agent(
            self.hexis_analysis_reviewer_llm
        )

        self.meal_compilation_agent = create_meal_compilation_agent(self.meal_compilation_llm)

        # Nutritional Validation Agent: pure reasoning agent (NO TOOLS)
        # It analyzes the meal plan and nutrition targets provided in the task description
        # without needing to call external APIs
        self.nutritional_validation_agent = create_nutritional_validation_agent(
            self.nutritional_validation_llm, tools=None
        )

        # Hexis Integration Agent: needs Hexis tools for meal verification
        # Create composite tool for integration agent
        # This wraps create_custom_food and verify_meal into a single tool
        # to ensure the 2-step process is atomic and reliable
        hexis_log_meal_tool = create_hexis_log_meal_tool(hexis_tools)

        # Wrap composite tool with input validation to handle list inputs from LLM
        # This is critical: LLMs may pass multiple meals as a list instead of calling once per meal
        print("🛡️  Wrapping hexis_log_meal composite tool with input validation...\n", file=sys.stderr)
        hexis_log_meal_tool = wrap_mcp_tool(hexis_log_meal_tool)

        # Integration agent gets ONLY the composite tool (and maybe get_weekly_plan if needed)
        # But for now, just the composite tool is enough for the logging part
        integration_tools = [hexis_log_meal_tool]

        self.mealy_integration_agent = create_mealy_integration_agent(
            llm=self.mealy_integration_llm,
            tools=integration_tools
        )

    def _create_agent_llm(
        self,
        agent_name: str,
        default_model: str,
        default_base: str,
        default_key: str,
        has_tools: bool = False,
    ) -> LLM:
        """Create an LLM for a specific agent with optional per-agent overrides.

        Supports environment variables:
        - {AGENT_NAME}_AGENT_MODEL: Model name for this agent
        - {AGENT_NAME}_AGENT_API_BASE: API base URL for this agent
        - {AGENT_NAME}_BLACKLISTED_MODELS: Comma-separated list of models to blacklist

        Note: Blacklist handling is now centralized in llm_provider_rotation.py
        via DEFAULT_AGENT_BLACKLISTS and build_category_cascade().

        Args:
            agent_name: Name prefix for env vars (e.g., "HEXIS_ANALYSIS", "MEAL_GENERATION")
            default_model: Fallback model name (or category like "complex")
            default_base: Fallback API base URL
            default_key: API key to use
            has_tools: If True, exclude thinking models that hallucinate tool calls

        Returns:
            Configured LLM instance
        """
        # Check for agent-specific overrides
        model_key = f"{agent_name}_AGENT_MODEL"
        base_key = f"{agent_name}_AGENT_API_BASE"

        agent_model = os.getenv(model_key, default_model)
        agent_base = os.getenv(base_key, default_base)

        # Log configuration for transparency
        print(
            f"🤖 {agent_name} Agent:\n"
            f"   Model: {agent_model}\n"
            f"   Endpoint: {agent_base}\n",
            file=sys.stderr,
        )

        # Blacklist handling is now centralized in llm_provider_rotation.py
        # via DEFAULT_AGENT_BLACKLISTS and build_category_cascade()
        return self._create_llm(
            agent_model,
            agent_base,
            default_key,
            agent_name=agent_name,
            has_tools=has_tools,
        )

    @staticmethod
    def _create_llm(
        model_name: str,
        api_base: str,
        api_key: str,
        agent_name: str,
        has_tools: bool = False,
    ) -> LLM:
        """Instantiate an LLM with provider rotation support when enabled."""

        return create_llm_with_rotation(
            agent_name=agent_name,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            has_tools=has_tools,
        )

    def _validate_ingredients_parallel(
        self,
        meal_template: Dict[str, Any],
        max_workers: int = 10,
    ) -> Dict[str, Any]:
        """Validate all ingredients in parallel using ThreadPoolExecutor.

        Bypasses the slow sequential CrewAI agent loop (1 tool call per iteration)
        by executing all ingredient searches in parallel via Python.

        Args:
            meal_template: Dict with 'meals' list containing MealTemplate dicts
            max_workers: Maximum concurrent search threads (default 10)

        Returns:
            ValidatedIngredientsList dict ready for the Reviewer
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import re

        if not self.passio_search_tool:
            print("   ⚠️ No passio_search_tool available for parallel validation", file=sys.stderr)
            return self._empty_validation_result(meal_template)

        # Extract unique ingredients from all meals
        # Key: normalized ingredient name, Value: (original_name, quantity_g)
        unique_ingredients: Dict[str, tuple] = {}
        meals = meal_template.get("meals", [])

        for meal in meals:
            ingredients_list = meal.get("ingredients_to_validate", [])
            for ingredient_str in ingredients_list:
                # Parse "120g chicken breast" -> (120.0, "chicken breast")
                quantity_g, clean_name = self._parse_ingredient_string(ingredient_str)
                key = clean_name.lower().strip()
                if key and key not in unique_ingredients:
                    unique_ingredients[key] = (ingredient_str, quantity_g, clean_name)

        if not unique_ingredients:
            print("   ⚠️ No ingredients to validate", file=sys.stderr)
            return self._empty_validation_result(meal_template)

        print(f"   🔍 Validating {len(unique_ingredients)} unique ingredients in parallel (max_workers={max_workers})...", file=sys.stderr)

        # Search all ingredients in parallel
        validated_cache: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        def search_single_ingredient(name: str, original: str, quantity: float) -> tuple:
            """Search a single ingredient (runs in thread)."""
            try:
                result = self.passio_search_tool._run(query=name, limit=5)

                # Parse result if string
                if isinstance(result, str):
                    result = json.loads(result)

                # Extract first food match
                foods = result.get("foods", [])
                if foods:
                    first_match = foods[0]
                    return (name, {
                        "name": original,
                        "passio_food_id": first_match.get("id") or first_match.get("resultId"),
                        "passio_food_name": first_match.get("displayName") or first_match.get("shortName") or first_match.get("longName"),
                        "passio_ref_code": first_match.get("refCode"),
                        "quantity_g": quantity,
                        "validation_status": "found",
                    })
                else:
                    return (name, {
                        "name": original,
                        "validation_status": "not_found",
                        "quantity_g": quantity,
                    })
            except Exception as e:
                return (name, {
                    "name": original,
                    "validation_status": "error",
                    "substitution_note": str(e)[:100],
                    "quantity_g": quantity,
                })

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(search_single_ingredient, name, orig, qty): name
                for name, (orig, qty, _) in unique_ingredients.items()
            }

            for future in as_completed(futures):
                name, result = future.result()
                validated_cache[name] = result

        # Count successes
        successful = sum(1 for v in validated_cache.values() if v.get("validation_status") == "found")
        print(f"   ✅ Validated {successful}/{len(unique_ingredients)} ingredients in parallel", file=sys.stderr)

        # Build ValidatedIngredientsList structure
        validated_meals = []
        total_validations = 0
        successful_validations = 0

        for meal in meals:
            meal_ingredients = []
            ingredients_list = meal.get("ingredients_to_validate", [])

            for ingredient_str in ingredients_list:
                _, clean_name = self._parse_ingredient_string(ingredient_str)
                key = clean_name.lower().strip()
                total_validations += 1

                if key in validated_cache:
                    cached = validated_cache[key]
                    meal_ingredients.append(cached)
                    if cached.get("validation_status") == "found":
                        successful_validations += 1
                else:
                    # Fallback for missing cache entry
                    meal_ingredients.append({
                        "name": ingredient_str,
                        "validation_status": "not_found",
                    })

            validated_meals.append({
                "meal_type": meal.get("meal_type", "Unknown"),
                "meal_name": meal.get("meal_name", "Unknown"),
                "validated_ingredients": meal_ingredients,
                "validation_success": all(
                    ing.get("validation_status") == "found" for ing in meal_ingredients
                ),
            })

        return {
            "day_name": meal_template.get("day_name", "Unknown"),
            "date": meal_template.get("date", ""),
            "validated_meals": validated_meals,
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "substitutions_made": 0,
        }

    # Unit conversions for ingredient parsing (grams per unit)
    UNIT_CONVERSIONS = {
        # Eggs by size
        "egg": 50, "eggs": 50,
        "large egg": 60, "large eggs": 60,
        "medium egg": 50, "medium eggs": 50,
        "small egg": 40, "small eggs": 40,
        # Bread slices
        "slice": 30, "slices": 30,
        "large slice": 40, "large slices": 40,
        "medium slice": 30, "medium slices": 30,
        "small slice": 20, "small slices": 20,
        # Volume measures (approximate gram equivalents)
        "cup": 240, "cups": 240,
        "tbsp": 15, "tsp": 5,
        # Generic pieces
        "piece": 50, "pieces": 50,
        "large piece": 75, "large pieces": 75,
        "medium piece": 50, "medium pieces": 50,
        "small piece": 30, "small pieces": 30,
        # Common items
        "banana": 120, "bananas": 120,
        "apple": 180, "apples": 180,
        "tortilla": 50, "tortillas": 50,
        "whole wheat tortilla": 50, "whole wheat tortillas": 50,
    }

    def _parse_ingredient_string(self, ingredient_str: str) -> tuple:
        """Parse ingredient string like '120g chicken breast' into (quantity_g, name).

        Handles various formats:
        - "120g chicken breast" -> (120, "chicken breast")
        - "2 large eggs" -> (120, "eggs")  # 2 × 60g
        - "3 eggs" -> (150, "eggs")  # 3 × 50g
        - "2 slices bread" -> (60, "bread")  # 2 × 30g
        - "1.5kg rice" -> (1500, "rice")
        - "olive oil" -> (None, "olive oil")

        Returns:
            (quantity_g: float or None, clean_name: str)
        """
        import re

        ingredient_str = ingredient_str.strip()

        # Pattern 1: "2 large eggs", "3 medium slices", "4 small pieces"
        match = re.match(
            r'^(\d+(?:\.\d+)?)\s+(large|medium|small)\s+(eggs?|slices?|pieces?|bananas?|apples?|tortillas?)(.*)$',
            ingredient_str, re.IGNORECASE
        )
        if match:
            qty = float(match.group(1))
            size = match.group(2).lower()
            unit = match.group(3).lower()
            rest = match.group(4).strip()

            # Build conversion key: "large eggs" or "large egg"
            key = f"{size} {unit}"
            conversion = self.UNIT_CONVERSIONS.get(key, self.UNIT_CONVERSIONS.get(unit, 50))

            # Name is either the rest or the unit itself
            name = rest if rest else unit.rstrip('s')
            return (qty * conversion, name)

        # Pattern 2: "2 eggs", "3 slices bread", "1 banana"
        match = re.match(
            r'^(\d+(?:\.\d+)?)\s+(eggs?|slices?|pieces?|cups?|tbsp|tsp|bananas?|apples?|tortillas?)\s*(.*)$',
            ingredient_str, re.IGNORECASE
        )
        if match:
            qty = float(match.group(1))
            unit = match.group(2).lower()
            rest = match.group(3).strip()

            conversion = self.UNIT_CONVERSIONS.get(unit, 50)
            name = rest if rest else unit.rstrip('s')
            return (qty * conversion, name)

        # Pattern 3: "120g chicken", "1.5kg rice", "200ml milk", "2l water"
        match = re.match(
            r'^(\d+(?:\.\d+)?)\s*(g|kg|ml|l)\s+(.+)$',
            ingredient_str, re.IGNORECASE
        )
        if match:
            qty = float(match.group(1))
            unit = match.group(2).lower()
            name = match.group(3).strip()

            if unit == "kg":
                qty *= 1000
            elif unit == "l":
                qty *= 1000
            # ml and g are already in the right unit
            return (qty, name)

        # Pattern 4: "120g" at start without space (e.g., "120g chicken breast")
        match = re.match(
            r'^(\d+(?:\.\d+)?)(g|kg|ml|l)(.+)$',
            ingredient_str, re.IGNORECASE
        )
        if match:
            qty = float(match.group(1))
            unit = match.group(2).lower()
            name = match.group(3).strip()

            if unit == "kg":
                qty *= 1000
            elif unit == "l":
                qty *= 1000
            return (qty, name)

        # Pattern 5: No quantity found, return the whole string as name
        return (None, ingredient_str)

    def _empty_validation_result(self, meal_template: Dict[str, Any]) -> Dict[str, Any]:
        """Return an empty validation result for fallback cases."""
        return {
            "day_name": meal_template.get("day_name", "Unknown"),
            "date": meal_template.get("date", ""),
            "validated_meals": [],
            "total_validations": 0,
            "successful_validations": 0,
            "substitutions_made": 0,
        }

    def _create_passio_nutrition_fetcher(self) -> Callable[[str], Optional[Dict[str, Any]]]:
        """Create a function that fetches nutrition data from Passio API.

        Returns a callable that takes a ref_code and returns nutrition data.
        This is used by fetch_nutrition_for_ingredients to get per-100g values.
        """
        # Find the hexis_get_passio_food_details tool
        details_tool = None
        for tool in self.mcp_tools:
            if "get_passio_food_details" in tool.name.lower():
                details_tool = tool
                break

        if details_tool is None:
            print("   ⚠️ hexis_get_passio_food_details tool not found in MCP tools", file=sys.stderr)
            return lambda ref_code: None

        def fetch_nutrition(ref_code: str) -> Optional[Dict[str, Any]]:
            """Fetch nutrition data for a single ingredient.

            Parses the nested Passio API response and extracts macro nutrients.
            Returns flat dict with protein, carbs, fat, calories (per 100g).
            """
            try:
                # Call the MCP tool
                result = details_tool._run(ref_code=ref_code)

                # Parse result if it's a string
                if isinstance(result, str):
                    result = json.loads(result)

                # Extract nutrients from nested structure:
                # result.results[0].ingredients[0].nutrients[]
                nutrients_list = []
                results = result.get("results", [])
                if results:
                    ingredients = results[0].get("ingredients", [])
                    if ingredients:
                        nutrients_list = ingredients[0].get("nutrients", [])

                if not nutrients_list:
                    return None

                # Build lookup map: nutrient_name -> amount
                nutrient_map = {}
                for n in nutrients_list:
                    name = n.get("nutrient", {}).get("name", "")
                    unit = n.get("nutrient", {}).get("unit", "")
                    amount = n.get("amount", 0)
                    # Store with unit suffix for energy disambiguation
                    nutrient_map[f"{name}|{unit}"] = amount
                    nutrient_map[name] = amount  # Also store without unit

                # Extract macros using Passio's nutrient names
                protein = nutrient_map.get("Protein", 0)
                fat = nutrient_map.get("Total lipid (fat)", 0)
                carbs = nutrient_map.get("Carbohydrate, by difference", 0)
                # Energy in KCAL (not kJ)
                calories = nutrient_map.get("Energy|KCAL", 0)
                if not calories:
                    calories = nutrient_map.get("Energy", 0)
                    # If we got kJ, convert to kcal (rough: kJ / 4.184)
                    if calories > 1000:  # Likely kJ
                        calories = round(calories / 4.184)

                return {
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat,
                    "calories": calories,
                }
            except Exception as e:
                print(f"      ⚠️ Passio fetch error: {e}", file=sys.stderr)
                return None

        return fetch_nutrition

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Remove markdown fences, thought blocks, reasoning prefixes from JSON text."""
        import re
        text = text.strip()

        # STRATEGY: Strip everything before the first '{' if it looks like reasoning
        # This handles multiline reasoning from thinking models (e.g., "Thought: ...")
        first_brace = text.find('{')
        if first_brace > 0:
            prefix = text[:first_brace]
            # Check if the prefix looks like LLM reasoning (common patterns)
            reasoning_indicators = [
                'Thought:', 'I need to', 'Let me', 'Looking at',
                'I should', 'First,', 'To ', 'Based on', 'Given ',
                'I will', "I'll", 'My approach', 'The task',
            ]
            prefix_lower = prefix.lower()
            if any(indicator.lower() in prefix_lower for indicator in reasoning_indicators):
                # Strip everything before the JSON
                text = text[first_brace:]

        text = text.strip()

        # Remove markdown code fences (anywhere in text)
        text = re.sub(r'```json\s*\n?', '', text)  # Remove opening fence
        text = re.sub(r'\n?\s*```', '', text)  # Remove closing fence

        # Remove <thought>...</thought> blocks (common in thinking models)
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)

        return text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """Extract a complete JSON object using brace balancing.

        More robust than regex for large/complex JSON with nested structures.
        Handles escaped quotes and nested objects correctly.
        """
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

        return None  # Unbalanced braces

    @staticmethod
    def _heal_hexis_analysis_structure(data: Dict[str, Any]) -> Dict[str, Any]:
        """Heal common structural errors in HexisWeeklyAnalysis JSON.

        Common error: LLM forgets closing brace for daily_energy_needs, causing
        daily_macro_targets and nutritional_priorities to be incorrectly nested.
        """
        if not isinstance(data, dict):
            return data

        # Check if daily_macro_targets is incorrectly nested inside daily_energy_needs
        daily_energy = data.get("daily_energy_needs", {})
        if isinstance(daily_energy, dict):
            nested_macros = daily_energy.get("daily_macro_targets")
            nested_priorities = daily_energy.get("nutritional_priorities")

            if nested_macros and "daily_macro_targets" not in data:
                print("   🔧 Structure healing: Moving daily_macro_targets to root", file=sys.stderr)
                data["daily_macro_targets"] = nested_macros
                del daily_energy["daily_macro_targets"]

            if nested_priorities and "nutritional_priorities" not in data:
                print("   🔧 Structure healing: Moving nutritional_priorities to root", file=sys.stderr)
                data["nutritional_priorities"] = nested_priorities
                del daily_energy["nutritional_priorities"]

        return data

    @staticmethod
    def _payloads_from_task_output(task_output: TaskOutput) -> List[Dict[str, Any]]:
        """Extract potential JSON payloads from a task output."""
        payloads: List[Dict[str, Any]] = []

        # Debug: log available sources
        print(f"   📥 Parsing task output:", file=sys.stderr)
        if task_output.json_dict:
            print(f"      - json_dict keys: {list(task_output.json_dict.keys())}", file=sys.stderr)
            payloads.append(task_output.json_dict)

        if task_output.pydantic:
            print(f"      - pydantic: present", file=sys.stderr)
            payloads.append(task_output.pydantic.model_dump())

        raw_value = task_output.raw
        if raw_value:
            print(f"      - raw length: {len(raw_value)}", file=sys.stderr)

            # Clean markdown fences before parsing
            cleaned = MealPlanningCrew._clean_json_text(raw_value)

            # Try brace-balanced extraction first (most reliable for large JSON)
            extracted = MealPlanningCrew._extract_json_object(cleaned)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    if isinstance(parsed, dict):
                        print(f"      - Brace-balanced parse SUCCESS, keys: {list(parsed.keys())}", file=sys.stderr)
                        # Apply structure healing for HexisWeeklyAnalysis
                        parsed = MealPlanningCrew._heal_hexis_analysis_structure(parsed)
                        payloads.append(parsed)
                        return payloads  # Success with complete JSON - return early
                except json.JSONDecodeError as e:
                    print(f"      - Brace-balanced parse failed: {e}", file=sys.stderr)

            # Fallback to existing logic
            try:
                candidate_texts = []

                if cleaned:
                    candidate_texts.append(cleaned)
                    first_brace = cleaned.find("{")
                    if first_brace > 0:
                        candidate_texts.append(cleaned[first_brace:])

                decoder = json.JSONDecoder()

                for candidate in candidate_texts:
                    normalized = candidate.strip()
                    if not normalized:
                        continue

                    parsed_raw: Dict[str, Any] | None = None

                    try:
                        potential = json.loads(normalized)
                        if isinstance(potential, dict):
                            parsed_raw = potential
                            print(f"      - json.loads SUCCESS, keys: {list(parsed_raw.keys())}", file=sys.stderr)
                    except json.JSONDecodeError:
                        try:
                            potential, _ = decoder.raw_decode(normalized)
                            if isinstance(potential, dict):
                                parsed_raw = potential
                                print(f"      - raw_decode SUCCESS, keys: {list(parsed_raw.keys())}", file=sys.stderr)
                        except json.JSONDecodeError:
                            # Fallback: Try json_repair for malformed JSON (e.g. unescaped newlines)
                            # 150KB limit to handle 7 days of Hexis data (~110KB)
                            if len(normalized) < 150000:
                                try:
                                    potential = json_repair.repair_json(normalized, return_objects=True)
                                    if isinstance(potential, dict):
                                        parsed_raw = potential
                                        print(f"      - json_repair SUCCESS, keys: {list(parsed_raw.keys())}", file=sys.stderr)
                                except Exception:
                                    continue
                            else:
                                print(f"      - Skipping json_repair (input too large: {len(normalized)} chars)", file=sys.stderr)
                                continue

                    if parsed_raw is not None:
                        # Apply structure healing for HexisWeeklyAnalysis
                        parsed_raw = MealPlanningCrew._heal_hexis_analysis_structure(parsed_raw)
                        payloads.append(parsed_raw)
                    else:
                        # Log failure to parse this candidate
                        print(f"   ⚠️  Failed to parse candidate JSON: {normalized[:100]}...", file=sys.stderr)

            except json.JSONDecodeError:
                pass

        return payloads

    def _collect_payload_candidates(
        self, crew_output: CrewOutput
    ) -> List[Dict[str, Any]]:
        """Collect payload candidates across crew and individual task outputs."""
        candidates: List[Dict[str, Any]] = []

        if crew_output.json_dict:
            # Apply structure healing for HexisWeeklyAnalysis
            healed = MealPlanningCrew._heal_hexis_analysis_structure(crew_output.json_dict)
            candidates.append(healed)

        if crew_output.pydantic:
            # Apply structure healing for HexisWeeklyAnalysis
            healed = MealPlanningCrew._heal_hexis_analysis_structure(crew_output.pydantic.model_dump())
            candidates.append(healed)

        for task_output in reversed(crew_output.tasks_output or []):
            candidates.extend(self._payloads_from_task_output(task_output))

        return candidates

    def _extract_model_from_output(
        self,
        crew_output: CrewOutput,
        model_type: Type[BaseModel],
    ) -> Optional[BaseModel]:
        """Safely parse the final task output into a typed model."""
        validation_errors = []
        for idx, candidate in enumerate(self._collect_payload_candidates(crew_output)):
            try:
                return model_type.model_validate(candidate)
            except ValidationError as e:
                # Log validation error for debugging
                error_summary = {
                    "candidate_index": idx,
                    "error_count": len(e.errors()),
                    "errors": [
                        {
                            "field": ".".join(str(loc) for loc in err["loc"]),
                            "type": err["type"],
                            "message": err["msg"],
                        }
                        for err in e.errors()[:5]  # Show first 5 errors
                    ],
                }
                validation_errors.append(error_summary)
                continue

        # Log all validation errors if parsing failed
        if validation_errors:
            print(
                f"\n❌ DEBUG: All validation attempts failed:\n{json.dumps(validation_errors, indent=2)}\n",
                file=sys.stderr,
            )

        return None

    def _extract_compliance_failure(
        self, crew_output: CrewOutput
    ) -> Optional[Dict[str, Any]]:
        """Extract compliance failure from reviewer output before DailyMealPlan validation.

        When the Reviewer detects macro compliance issues, it returns:
        {"compliance_failed": true, "compliance_issues": [...]}

        This method checks for this pattern BEFORE attempting DailyMealPlan validation,
        allowing us to capture the feedback and pass it to the next Supervisor attempt.

        Returns:
            Dict with compliance_issues if compliance failed, None otherwise
        """
        for candidate in self._collect_payload_candidates(crew_output):
            if isinstance(candidate, dict) and candidate.get("compliance_failed") is True:
                return {
                    "compliance_failed": True,
                    "compliance_issues": candidate.get("compliance_issues", []),
                    "calculated_totals": candidate.get("calculated_totals"),
                    "target_totals": candidate.get("target_totals"),
                }
        return None

    def _enrich_meal_template_with_targets(
        self,
        candidate: Dict[str, Any],
        daily_target: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject target_* fields from daily_target if missing from LLM response.

        This is a fallback mechanism to handle models that omit required fields
        even when explicitly instructed to include them.
        """
        enriched = candidate.copy()

        # Only inject if missing - don't override LLM values
        if "target_calories" not in enriched:
            enriched["target_calories"] = int(daily_target.get("calories", 0))
            print(
                f"   ℹ️  Injected missing target_calories: {enriched['target_calories']}",
                file=sys.stderr,
            )
        if "target_protein" not in enriched:
            enriched["target_protein"] = float(
                daily_target.get("macros", {}).get("protein_g", 0)
            )
            print(
                f"   ℹ️  Injected missing target_protein: {enriched['target_protein']}",
                file=sys.stderr,
            )
        if "target_carbs" not in enriched:
            enriched["target_carbs"] = float(
                daily_target.get("macros", {}).get("carbs_g", 0)
            )
            print(
                f"   ℹ️  Injected missing target_carbs: {enriched['target_carbs']}",
                file=sys.stderr,
            )
        if "target_fat" not in enriched:
            enriched["target_fat"] = float(
                daily_target.get("macros", {}).get("fat_g", 0)
            )
            print(
                f"   ℹ️  Injected missing target_fat: {enriched['target_fat']}",
                file=sys.stderr,
            )

        return enriched

    def _extract_model_with_enrichment(
        self,
        crew_output: CrewOutput,
        model_type: Type[BaseModel],
        enrichment_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[BaseModel]:
        """Extract model from output, optionally enriching MealPlanTemplate with targets.

        Args:
            crew_output: The CrewAI output to parse
            model_type: The Pydantic model type to validate against
            enrichment_data: If provided and model_type is MealPlanTemplate,
                            inject missing target_* fields from this data
        """
        validation_errors = []
        for idx, candidate in enumerate(self._collect_payload_candidates(crew_output)):
            try:
                # Apply enrichment for MealPlanTemplate if data provided
                if enrichment_data is not None and model_type.__name__ == "MealPlanTemplate":
                    candidate = self._enrich_meal_template_with_targets(
                        candidate, enrichment_data
                    )
                return model_type.model_validate(candidate)
            except ValidationError as e:
                error_summary = {
                    "candidate_index": idx,
                    "error_count": len(e.errors()),
                    "errors": [
                        {
                            "field": ".".join(str(loc) for loc in err["loc"]),
                            "type": err["type"],
                            "message": err["msg"],
                        }
                        for err in e.errors()[:5]
                    ],
                }
                validation_errors.append(error_summary)
                continue

        if validation_errors:
            print(
                f"\n❌ DEBUG: All validation attempts failed:\n{json.dumps(validation_errors, indent=2)}\n",
                file=sys.stderr,
            )

        return None

    def _generate_daily_meal_plans(
        self,
        nutrition_plan: Dict[str, Any],
        max_days: Optional[int] = None,
        validation_feedback: Optional[Dict[str, Any]] = None,
        previous_successful_plans: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate daily meal plans using Supervisor/Executor/Reviewer pattern.

        The 3-agent pattern separates:
        1. SUPERVISOR: Pure reasoning - designs meals creatively (thinking models OK)
        2. EXECUTOR: Tool execution - validates ingredients via Passio API (no thinking models)
        3. REVIEWER: Pure reasoning - calculates macros and finalizes (thinking models OK)

        This prevents thinking models from hallucinating tool calls.

        If previous_successful_plans is provided, days that already exist in that list
        will be reused instead of regenerated. Only missing days will be generated.
        """
        daily_targets = nutrition_plan.get("daily_targets", [])

        if not daily_targets:
            print("\n❌ No daily targets provided in nutrition plan\n", file=sys.stderr)
            return None

        total_targets = len(daily_targets)

        if max_days is not None:
            if max_days <= 0:
                print("\n❌ Cannot generate meal plans for fewer than one day\n", file=sys.stderr)
                return None

            if max_days < total_targets:
                limited_days = ", ".join(
                    target.get("day_name", "?") for target in daily_targets[:max_days]
                )
                print(
                    f"\nℹ️  Limiting meal generation to first {max_days} day(s): {limited_days}\n",
                    file=sys.stderr,
                )
            elif max_days > total_targets:
                print(
                    f"\nℹ️  Requested {max_days} day(s) but only {total_targets} target(s) available; generating all available days\n",
                    file=sys.stderr,
                )

            daily_targets = daily_targets[:max_days]

        # Filter out days that were already successfully generated
        reused_plans: List[Dict[str, Any]] = []
        targets_to_generate = daily_targets

        if previous_successful_plans:
            # Build set of day_names that already succeeded
            successful_day_names = {
                plan.get("day_name") for plan in previous_successful_plans
            }
            # Keep plans from previous round
            reused_plans = [
                plan for plan in previous_successful_plans
                if plan.get("day_name") in {t.get("day_name") for t in daily_targets}
            ]
            # Only generate days that are missing
            targets_to_generate = [
                target for target in daily_targets
                if target.get("day_name") not in successful_day_names
            ]

            if reused_plans:
                reused_names = ", ".join(p.get("day_name", "?") for p in reused_plans)
                print(
                    f"\n♻️  Reusing {len(reused_plans)} successful day(s) from previous round: {reused_names}\n",
                    file=sys.stderr,
                )

            if not targets_to_generate:
                print(
                    f"\n✅ All {len(reused_plans)} day(s) already generated successfully, nothing to regenerate\n",
                    file=sys.stderr,
                )
                # Sort by date and return
                reused_plans.sort(key=lambda x: x.get("date", ""))
                return reused_plans

        # Check if parallel generation is enabled (default: True for 2+ days)
        parallel_enabled = os.getenv("PARALLEL_DAY_GENERATION", "true").lower() == "true"
        num_days = len(targets_to_generate)

        if parallel_enabled and num_days >= 2:
            print(
                f"\n🚀 Parallel mode: Generating {num_days} days concurrently (max 4 workers)...\n",
                file=sys.stderr,
            )
            new_plans = self._generate_days_parallel(
                targets_to_generate, nutrition_plan, validation_feedback
            )
        else:
            print(
                f"\n📋 Sequential mode: Generating {num_days} day(s) one at a time...\n",
                file=sys.stderr,
            )
            new_plans = self._generate_days_sequential(
                targets_to_generate, nutrition_plan, validation_feedback
            )

        # Merge reused plans with newly generated plans
        if new_plans is None:
            # Generation failed - return reused plans if any
            if reused_plans:
                print(
                    f"\n⚠️  Generation failed, returning {len(reused_plans)} reused day(s)\n",
                    file=sys.stderr,
                )
                reused_plans.sort(key=lambda x: x.get("date", ""))
                return reused_plans
            return None

        # === FIX: Programmatic variety check ===
        # Check for duplicate meals across days and regenerate if needed
        if len(new_plans) >= 2:
            uniqueness_check = self._check_meal_uniqueness(new_plans)
            variety_result = check_variety(
                unique_meals=uniqueness_check["unique_meals"],
                expected_meals=uniqueness_check["total_meals"],
            )

            print(
                f"\n🔍 Variety check: {variety_result['message']}\n",
                file=sys.stderr,
            )

            if uniqueness_check["duplicates"]:
                print(
                    f"   ⚠️  Duplicates found: {list(uniqueness_check['duplicates'].keys())[:5]}...\n",
                    file=sys.stderr,
                )

                # Regenerate duplicate days (only in parallel mode, sequential already has context)
                if parallel_enabled:
                    new_plans = self._regenerate_duplicate_days(
                        new_plans,
                        uniqueness_check["duplicates"],
                        targets_to_generate,
                        nutrition_plan,
                    )

                    # Re-check after regeneration
                    final_check = self._check_meal_uniqueness(new_plans)
                    print(
                        f"\n   ✅ After regeneration: {final_check['unique_meals']}/{final_check['total_meals']} unique ({final_check['ratio']:.0%})\n",
                        file=sys.stderr,
                    )

        merged_plans = reused_plans + new_plans
        merged_plans.sort(key=lambda x: x.get("date", ""))
        return merged_plans

    def _generate_variety_seed(
        self, daily_targets: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, str]]:
        """Pre-assign unique meal themes per day to ensure variety in parallel mode.

        This ensures that even when days are generated concurrently without context
        from previous days, each day will have unique meal themes.

        IMPORTANT: Uses DATE as key (not day_name) to avoid misalignment when LLM
        generates incorrect day_name/date pairs.

        Args:
            daily_targets: List of daily targets with date and day_name

        Returns:
            Dict mapping DATE to theme assignments (breakfast, lunch, dinner)
        """
        breakfast_themes = [
            "Oeufs brouillés aux légumes grillés",
            "Porridge d'avoine aux fruits et noix",
            "Smoothie bowl protéiné aux baies",
            "Tartines complètes avocat-oeuf poché",
            "Pancakes protéinés aux fruits rouges",
            "Yaourt grec aux graines de chia et miel",
            "Omelette aux champignons et fines herbes",
        ]

        lunch_themes = [
            "Salade composée poulet-quinoa",
            "Bowl méditerranéen au thon",
            "Wrap complet dinde-crudités",
            "Poke bowl saumon-edamame",
            "Salade César au poulet grillé",
            "Buddha bowl lentilles-légumes rôtis",
            "Taboulé de quinoa au poulet",
        ]

        dinner_themes = [
            "Saumon grillé et légumes vapeur",
            "Poulet rôti aux herbes et patate douce",
            "Steak de thon et riz complet",
            "Filet de cabillaud en papillote",
            "Escalope de dinde et quinoa aux légumes",
            "Curry de crevettes et riz basmati",
            "Pavé de boeuf et légumes grillés",
        ]

        variety_seed = {}
        for i, target in enumerate(daily_targets):
            # Use DATE as key (not day_name) to ensure correct lookup even if LLM
            # generated wrong day_name/date pairs
            date_key = target.get("date", f"day_{i+1}")
            variety_seed[date_key] = {
                "breakfast_theme": breakfast_themes[i % len(breakfast_themes)],
                "lunch_theme": lunch_themes[i % len(lunch_themes)],
                "dinner_theme": dinner_themes[i % len(dinner_themes)],
            }

        return variety_seed

    def _is_valid_executor_result(
        self, result: Optional[ValidatedIngredientsList]
    ) -> bool:
        """Check if the Executor result is valid (has validated ingredients).

        Args:
            result: The parsed ValidatedIngredientsList model or None

        Returns:
            True if result has successful validations, False otherwise
        """
        if result is None:
            return False

        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        if not isinstance(result_dict, dict):
            return False

        # Check for validated_meals
        validated_meals = result_dict.get("validated_meals", [])
        if not validated_meals:
            return False

        # Check for successful_validations > 0
        return result_dict.get("successful_validations", 0) > 0

    def _check_meal_uniqueness(
        self, daily_plans: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check meal uniqueness across all generated days.

        Args:
            daily_plans: List of generated daily meal plans

        Returns:
            Dict with uniqueness analysis:
            - unique_meals: count of unique meal names
            - total_meals: total meals checked
            - ratio: uniqueness ratio
            - passes_threshold: True if ratio >= 85%
            - duplicates: dict mapping meal_name to list of days where it appears
        """
        all_meal_names: List[str] = []
        meal_to_day: Dict[str, List[str]] = {}

        for day_plan in daily_plans:
            day_name = day_plan.get("day_name", "Unknown")
            meals = day_plan.get("meals", [])

            for meal in meals:
                meal_name = meal.get("meal_name", "").lower().strip()
                if meal_name:
                    all_meal_names.append(meal_name)
                    if meal_name in meal_to_day:
                        meal_to_day[meal_name].append(day_name)
                    else:
                        meal_to_day[meal_name] = [day_name]

        unique_meals = len(set(all_meal_names))
        total_meals = len(all_meal_names)
        ratio = unique_meals / total_meals if total_meals > 0 else 1.0

        # Find duplicates (meals appearing in multiple days)
        duplicates = {
            name: days for name, days in meal_to_day.items() if len(days) > 1
        }

        return {
            "unique_meals": unique_meals,
            "total_meals": total_meals,
            "ratio": ratio,
            "passes_threshold": ratio >= MIN_UNIQUE_MEALS_RATIO,
            "duplicates": duplicates,
        }

    def _regenerate_duplicate_days(
        self,
        daily_plans: List[Dict[str, Any]],
        duplicates: Dict[str, List[str]],
        daily_targets: List[Dict[str, Any]],
        nutrition_plan: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Regenerate days that have duplicate meals.

        Keeps the first occurrence of each meal and regenerates later days
        that contain duplicates.

        Args:
            daily_plans: Current list of daily meal plans
            duplicates: Dict mapping meal_name to list of days
            daily_targets: Original daily targets for regeneration
            nutrition_plan: Weekly nutrition plan context

        Returns:
            Updated list of daily plans with duplicates resolved
        """
        # Find which days need regeneration (keep first occurrence, regenerate later ones)
        days_to_regenerate: set = set()
        for meal_name, day_list in duplicates.items():
            # Keep the first day, regenerate others
            for day in day_list[1:]:
                days_to_regenerate.add(day)

        if not days_to_regenerate:
            return daily_plans

        print(
            f"\n🔄 Regenerating {len(days_to_regenerate)} day(s) with duplicate meals: {', '.join(days_to_regenerate)}\n",
            file=sys.stderr,
        )

        # Create a mutable copy
        updated_plans = list(daily_plans)

        # Regenerate each duplicate day with full context from other days
        for day_name in days_to_regenerate:
            # Find the target for this day
            day_target = None
            for t in daily_targets:
                if t.get("day_name") == day_name:
                    day_target = t
                    break

            if day_target is None:
                print(
                    f"   ⚠️  Could not find target for {day_name}, skipping\n",
                    file=sys.stderr,
                )
                continue

            # Pass ALL other days as previous_days context to avoid re-duplication
            other_days = [p for p in updated_plans if p.get("day_name") != day_name]

            # Generate with variety feedback
            new_plan = self._generate_single_day(
                day_target,
                nutrition_plan,
                other_days,  # Full context from all other days
                validation_feedback={"variety_issue": f"Regenerating {day_name} due to duplicate meals"},
            )

            if new_plan:
                # Replace in list
                for i, p in enumerate(updated_plans):
                    if p.get("day_name") == day_name:
                        updated_plans[i] = new_plan
                        print(
                            f"   ✅ {day_name}: Regenerated successfully\n",
                            file=sys.stderr,
                        )
                        break
            else:
                print(
                    f"   ❌ {day_name}: Regeneration failed, keeping original\n",
                    file=sys.stderr,
                )

        return updated_plans

    def _generate_days_parallel(
        self,
        daily_targets: List[Dict[str, Any]],
        nutrition_plan: Dict[str, Any],
        validation_feedback: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate daily meal plans in parallel using ThreadPoolExecutor.

        Uses shared context with staggered starts so each worker can see
        completed results from earlier workers, reducing duplicates.
        """
        max_workers = min(4, len(daily_targets))  # Limit to 4 concurrent days

        # Generate variety seed BEFORE parallel execution
        variety_seed = self._generate_variety_seed(daily_targets)
        print(
            f"\n🎲 Variety seed generated for {len(variety_seed)} days\n",
            file=sys.stderr,
        )

        # Shared context: completed plans accessible to all workers
        generated_plans: Dict[str, Dict[str, Any]] = {}
        lock = threading.Lock()
        failed_days: List[str] = []

        def generate_with_context(target: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            """Generate a single day with access to shared completed context."""
            day_name = target.get("day_name", "Unknown")
            date_key = target.get("date", "")  # Use date for variety_seed lookup

            try:
                # Get already-completed days as context (thread-safe read)
                with lock:
                    previous_days = list(generated_plans.values())

                # Generate with context from completed days
                # NOTE: variety_seed uses DATE as key (not day_name) to ensure correct
                # lookup even if LLM generated wrong day_name/date pairs
                result = self._generate_single_day(
                    target,
                    nutrition_plan,
                    previous_days,  # Pass completed days as context
                    validation_feedback,
                    variety_seed.get(date_key, {}),
                )

                if result:
                    # Store result for other workers to see
                    with lock:
                        generated_plans[day_name] = result
                    print(f"\n✅ {day_name}: Completed successfully\n", file=sys.stderr)
                    return result
                else:
                    with lock:
                        failed_days.append(day_name)
                    print(f"\n❌ {day_name}: Failed to generate\n", file=sys.stderr)
                    return None

            except Exception as e:
                with lock:
                    failed_days.append(day_name)
                print(f"\n❌ {day_name}: Error - {e}\n", file=sys.stderr)
                return None

        # Submit with staggered starts (0.5s delay per day) to allow context sharing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, target in enumerate(daily_targets):
                if idx > 0:
                    time.sleep(0.5)  # Stagger starts to allow earlier days to complete first
                futures.append(executor.submit(generate_with_context, target, idx))

            # Wait for all to complete
            concurrent.futures.wait(futures)

        if failed_days:
            print(
                f"\n⚠️  {len(failed_days)} day(s) failed: {', '.join(failed_days)}\n",
                file=sys.stderr,
            )
            if len(generated_plans) == 0:
                return None

        # Convert to list and sort by date
        generated_daily_plans = list(generated_plans.values())
        generated_daily_plans.sort(key=lambda x: x.get("date", ""))

        print(
            f"\n✅ Generated {len(generated_daily_plans)} daily plans in parallel (with shared context)\n",
            file=sys.stderr,
        )
        return generated_daily_plans

    def _generate_days_sequential(
        self,
        daily_targets: List[Dict[str, Any]],
        nutrition_plan: Dict[str, Any],
        validation_feedback: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate daily meal plans sequentially (original behavior).

        Each day can use context from previous days for variety.
        """
        generated_daily_plans: List[Dict[str, Any]] = []

        for target in daily_targets:
            daily_plan = self._generate_single_day(
                target,
                nutrition_plan,
                generated_daily_plans,  # Pass previous days for context
                validation_feedback,
            )

            if daily_plan is None:
                day_name = target.get("day_name", "Unknown")
                print(
                    f"\n❌ {day_name}: Failed to generate, aborting...\n",
                    file=sys.stderr,
                )
                return None

            generated_daily_plans.append(daily_plan)

        print(
            f"\n✅ Generated {len(generated_daily_plans)} daily plans successfully\n",
            file=sys.stderr,
        )
        return generated_daily_plans

    def _generate_single_day(
        self,
        target: Dict[str, Any],
        nutrition_plan: Dict[str, Any],
        previous_days: List[Dict[str, Any]],
        validation_feedback: Optional[Dict[str, Any]] = None,
        variety_seed: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a single day's meal plan using Supervisor/Executor/Reviewer pattern.

        Args:
            target: Daily nutritional targets with meal_targets
            nutrition_plan: Weekly context
            previous_days: Previously generated days (for variety, may be empty in parallel mode)
            validation_feedback: Feedback from failed attempts
            variety_seed: Pre-assigned meal themes for this day (used in parallel mode)

        Returns:
            DailyMealPlan dict or None if failed
        """
        day_name = target.get("day_name", "Unknown day")
        day_date = target.get("date", "?")
        print(
            f"\n📆 Generating meals for {day_name} ({day_date}) using Supervisor/Executor/Reviewer pattern...\n",
            file=sys.stderr,
        )

        max_attempts = 4  # 4th attempt uses relaxed thresholds (+50%)
        daily_plan: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_attempts + 1):
            # 4th attempt uses relaxed thresholds (+50%) as a fallback
            use_relaxed = (attempt == max_attempts)
            threshold_note = " (RELAXED thresholds +50%)" if use_relaxed else ""
            print(
                f"\n🔄 {day_name} - Attempt {attempt}/{max_attempts}{threshold_note}...\n",
                file=sys.stderr,
            )

            # =================================================================
            # STEP 1: SUPERVISOR - Design meals (pure reasoning, no tools)
            # =================================================================
            print(f"   👨‍🍳 Step 1: Supervisor designing meal plan...", file=sys.stderr)

            # Extract meal_targets from the enriched daily target
            meal_targets = target.get("meal_targets", [])
            if not meal_targets:
                raise ValueError(
                    f"Missing meal_targets for {target.get('date')}. "
                    "Hexis data is required for meal planning."
                )

            supervisor_task = create_meal_planning_supervisor_task(
                self.meal_planning_supervisor_agent,
                daily_target=target,
                weekly_context=nutrition_plan,
                previous_days=previous_days,
                validation_feedback=validation_feedback,
                meal_targets=meal_targets,
                variety_seed=variety_seed,
            )
            supervisor_crew = Crew(
                agents=[self.meal_planning_supervisor_agent],
                tasks=[supervisor_task],
                process=Process.sequential,
                verbose=True,
                max_iter=10,
                memory=False,
            )

            supervisor_result = supervisor_crew.kickoff()
            # Use enrichment to inject target_* fields if LLM omits them
            meal_template_model = self._extract_model_with_enrichment(
                supervisor_result, MealPlanTemplate, enrichment_data=target
            )

            if meal_template_model is None:
                print(
                    f"\n   ⚠️  Supervisor failed to produce valid MealPlanTemplate, retrying...\n",
                    file=sys.stderr,
                )
                continue

            meal_template = meal_template_model.model_dump()
            print(
                f"   ✅ Supervisor created template with {len(meal_template.get('meals', []))} meals\n",
                file=sys.stderr,
            )

            # =================================================================
            # STEP 2: PARALLEL INGREDIENT VALIDATION (Python ThreadPoolExecutor)
            # =================================================================
            # Bypasses slow CrewAI agent loop (1 tool call per iteration)
            # by validating all ingredients in parallel via Python
            print(f"   🔍 Step 2: Parallel ingredient validation...", file=sys.stderr)

            validated_ingredients = self._validate_ingredients_parallel(
                meal_template, max_workers=10
            )

            # =================================================================
            # STEP 2a-bis: FETCH MISSING NUTRITION DATA FROM PASSIO API
            # =================================================================
            # The LLM Executor only calls hexis_search_passio_foods (step 1).
            # We now auto-fetch nutrition using hexis_get_passio_food_details.
            nutrition_fetcher = self._create_passio_nutrition_fetcher()
            validated_ingredients = fetch_nutrition_for_ingredients(
                validated_ingredients, nutrition_fetcher
            )

            # =================================================================
            # STEP 2b: PRE-CALCULATE MACROS IN PYTHON (optimization)
            # =================================================================
            # This reduces the Reviewer's work by ~50% - no more LLM math
            validated_ingredients = enrich_validated_ingredients(validated_ingredients)

            # Extract target totals for summary (macros are nested under "macros" key)
            macros = target.get("macros", {})
            target_totals = {
                "calories": target.get("calories", 0),
                "protein_g": macros.get("protein_g", 0),
                "carbs_g": macros.get("carbs_g", 0),
                "fat_g": macros.get("fat_g", 0),
            }
            pre_calc_summary = format_pre_calculated_summary(validated_ingredients, target_totals)
            print(f"   📊 Pre-calculated macros:\n{pre_calc_summary}\n", file=sys.stderr)

            # =================================================================
            # STEP 3: REVIEWER - Calculate macros and finalize (pure reasoning)
            # =================================================================
            print(f"   📊 Step 3: Reviewer finalizing meal plan...", file=sys.stderr)

            reviewer_task = create_meal_recipe_reviewer_task(
                self.meal_recipe_reviewer_agent,
                meal_plan_template=meal_template,
                validated_ingredients=validated_ingredients,
                daily_target=target,
                pre_calc_summary=pre_calc_summary,  # Pass pre-calculated summary
            )
            reviewer_crew = Crew(
                agents=[self.meal_recipe_reviewer_agent],
                tasks=[reviewer_task],
                process=Process.sequential,
                verbose=True,
                max_iter=10,
                memory=False,
            )

            reviewer_result = reviewer_crew.kickoff()

            # Check for compliance failure BEFORE DailyMealPlan validation
            # This captures Reviewer feedback when macros are out of range
            compliance_failure = self._extract_compliance_failure(reviewer_result)
            if compliance_failure:
                issues = compliance_failure.get("compliance_issues", [])
                print(
                    f"\n   ⚠️  Reviewer REJECTED plan due to compliance issues:",
                    file=sys.stderr,
                )
                for issue in issues[:3]:
                    print(f"      - {issue}", file=sys.stderr)

                if attempt < max_attempts:
                    # Build feedback for next Supervisor attempt from compliance failure
                    # Convert issues to strings if they are dicts (LLM sometimes outputs structured data)
                    issues_str = []
                    for issue in issues[:2]:
                        if isinstance(issue, dict):
                            issues_str.append(str(issue.get("message", issue.get("issue", str(issue)))))
                        else:
                            issues_str.append(str(issue))
                    validation_feedback = {
                        **(validation_feedback or {}),
                        "previous_attempt": attempt,
                        "compliance_failed": True,
                        "macro_issues": issues,
                        "critical_instruction": (
                            "The Reviewer REJECTED the previous plan. Adjust portions: "
                            + "; ".join(issues_str)
                        ),
                    }
                    print(
                        f"\n   🔄 Retrying with Reviewer compliance feedback...\n",
                        file=sys.stderr,
                    )
                    continue
                else:
                    print(
                        f"\n   ⚠️  Max attempts reached with compliance failure\n",
                        file=sys.stderr,
                    )

            # Only try DailyMealPlan parsing if not a compliance failure
            meal_model = self._extract_model_from_output(reviewer_result, DailyMealPlan)

            if meal_model is None:
                print(
                    f"\n   ⚠️  Reviewer failed to produce valid DailyMealPlan, retrying...\n",
                    file=sys.stderr,
                )
                continue

            daily_plan = meal_model.model_dump()
            print(
                f"\n✅ {day_name} - Attempt {attempt}: Generated {len(daily_plan.get('meals', []))} meals via Supervisor/Executor/Reviewer\n",
                file=sys.stderr,
            )

            # =================================================================
            # STEP 4a: Python quantity adjustment (if needed)
            # =================================================================
            daily_plan, was_adjusted = self._adjust_ingredient_quantities(daily_plan, target)
            if was_adjusted:
                print(
                    f"   ✅ Python fine-tuned ingredient quantities",
                    file=sys.stderr,
                )

            # =================================================================
            # STEP 4b: Per-day macro validation
            # =================================================================
            # Pass use_relaxed=True on 4th attempt for +50% tolerance fallback
            macro_validation = self._validate_daily_macros(
                daily_plan, target, meal_targets, use_relaxed_thresholds=use_relaxed
            )
            if not macro_validation["passed"]:
                print(
                    f"\n   ⚠️  Macro validation failed: {macro_validation['reason']}\n",
                    file=sys.stderr,
                )
                for issue in macro_validation["issues"][:3]:
                    print(f"      - {issue}", file=sys.stderr)

                if attempt < max_attempts:
                    # Build feedback for next Supervisor attempt
                    local_feedback = validation_feedback or {}
                    python_note = (
                        "Python already attempted quantity adjustment but macros are still off. "
                        "The Supervisor MUST design meals with significantly different ingredient choices or portions."
                    ) if was_adjusted else ""

                    # Prioritize issues: focus on the biggest deviation first
                    issues = macro_validation["issues"]
                    priority_issue = issues[0] if issues else None
                    other_issues = issues[1:] if len(issues) > 1 else []

                    # Improved feedback: single priority fix + constraint to avoid breaking other macros
                    validation_feedback = {
                        **local_feedback,
                        "previous_attempt": attempt,
                        "priority_fix": priority_issue,
                        "other_issues": other_issues,
                        "macro_issues": issues,  # Keep full list for reference
                        "adjustments_needed": macro_validation["adjustments"],
                        "missing_meals": macro_validation["missing_meals"],
                        "python_adjustment_attempted": was_adjusted,
                        "constraint": (
                            "IMPORTANT: Fix the priority_fix issue FIRST without breaking other macros. "
                            "Make targeted adjustments (±10-20g ingredient changes), not complete meal redesigns."
                        ),
                        "critical_instruction": python_note or f"Priority: {priority_issue}",
                    }
                    print(
                        f"\n   🔄 Retrying with priority feedback: {priority_issue} (Python adjusted: {was_adjusted})...\n",
                        file=sys.stderr,
                    )
                    continue
                else:
                    print(
                        f"\n   ⚠️  Max attempts reached, accepting plan with issues\n",
                        file=sys.stderr,
                    )

            print(
                f"\n   ✅ Macro validation passed for {day_name}\n",
                file=sys.stderr,
            )
            break

        if daily_plan is None:
            print(
                f"\n❌ {day_name}: Failed to produce a valid daily plan after {max_attempts} attempts\n",
                file=sys.stderr,
            )

        return daily_plan

    def _adjust_ingredient_quantities(
        self,
        daily_plan: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """Post-process meal plan to fine-tune ingredient quantities for macro targets.

        This function adjusts ingredient quantities when the LLM-generated plan
        exceeds the 10% tolerance threshold. It uses Passio nutritional data
        (protein_per_100g, carbs_per_100g, fat_per_100g, calories_per_100g)
        directly from each ingredient's validated data.

        Args:
            daily_plan: The generated daily meal plan with meals and daily_totals
            target: The nutritional target for the day with calories and macros

        Returns:
            Tuple of (adjusted_plan, was_adjusted)
            - adjusted_plan: Copy of daily_plan with adjusted quantities
            - was_adjusted: True if any adjustments were made
        """
        import copy

        # 1. Extract current and target values
        daily_totals = daily_plan.get("daily_totals", {})
        if not daily_totals:
            return daily_plan, False

        current_protein = daily_totals.get("protein_g", 0)
        current_carbs = daily_totals.get("carbs_g", 0)
        current_fat = daily_totals.get("fat_g", 0)
        current_calories = daily_totals.get("calories", 0)

        target_macros = target.get("macros", {})
        target_protein = target_macros.get("protein_g", 0)
        target_carbs = target_macros.get("carbs_g", 0)
        target_fat = target_macros.get("fat_g", 0)
        target_calories = target.get("calories", 0)

        # 2. Calculate deltas
        protein_delta = current_protein - target_protein
        carbs_delta = current_carbs - target_carbs
        fat_delta = current_fat - target_fat

        # 3. Check if within 10% tolerance (souple threshold)
        TOLERANCE = 0.10  # 10% threshold
        protein_pct = abs(protein_delta) / target_protein if target_protein > 0 else 0
        carbs_pct = abs(carbs_delta) / target_carbs if target_carbs > 0 else 0
        fat_pct = abs(fat_delta) / target_fat if target_fat > 0 else 0

        # Guard: Don't adjust if current values are 0 (indicates missing data, not actual deficit)
        if current_calories == 0 or (current_protein == 0 and current_carbs == 0 and current_fat == 0):
            return daily_plan, False  # Skip adjustment - data is incomplete

        if protein_pct <= TOLERANCE and carbs_pct <= TOLERANCE and fat_pct <= TOLERANCE:
            return daily_plan, False

        # 4. Create a deep copy and adjust ingredients
        adjusted_plan = copy.deepcopy(daily_plan)
        was_adjusted = False
        adjustment_log = []

        # Find which macro needs the most adjustment
        max_issue = max(
            [("protein", protein_delta, protein_pct),
             ("carbs", carbs_delta, carbs_pct),
             ("fat", fat_delta, fat_pct)],
            key=lambda x: x[2]
        )
        priority_macro = max_issue[0]

        for meal in adjusted_plan.get("meals", []):
            if was_adjusted:
                break  # Only adjust one ingredient per validation cycle

            for ing in meal.get("validated_ingredients", []):
                if was_adjusted:
                    break

                ing_name = ing.get("name", "")
                quantity = ing.get("quantity_g")

                if quantity is None or quantity < 20:  # Skip very small quantities
                    continue

                # Read Passio macros directly from ingredient schema
                protein_per_100g = ing.get("protein_per_100g")
                carbs_per_100g = ing.get("carbs_per_100g")
                fat_per_100g = ing.get("fat_per_100g")
                calories_per_100g = ing.get("calories_per_100g")

                # Skip ingredients without Passio nutritional data
                if protein_per_100g is None and carbs_per_100g is None and fat_per_100g is None:
                    continue

                # Default to 0 if any macro is missing
                protein_per_100g = protein_per_100g or 0
                carbs_per_100g = carbs_per_100g or 0
                fat_per_100g = fat_per_100g or 0
                calories_per_100g = calories_per_100g or 0

                # Adjust based on priority macro using Passio data
                # Cap adjustments: min 30% of original (prevent over-reduction), max 3x (prevent over-increase)
                MIN_SCALE = 0.3
                MAX_SCALE = 3.0

                if priority_macro == "protein" and protein_pct > TOLERANCE:
                    if protein_per_100g >= 10:  # Protein-rich food (>=10g per 100g)
                        # Calculate how much to adjust
                        protein_per_g = protein_per_100g / 100
                        adjustment_g = protein_delta / protein_per_g

                        # Skip if adjustment would change quantity by more than 200%
                        if abs(adjustment_g) > quantity * 2:
                            continue

                        # Apply adjustment with caps (min 30% of original, max 3x original)
                        min_qty = max(20, quantity * MIN_SCALE)
                        max_qty = quantity * MAX_SCALE
                        new_qty = max(min_qty, min(max_qty, quantity - adjustment_g))
                        if abs(new_qty - quantity) >= 10:  # Only if significant change
                            ing["adjusted_quantity_g"] = round(new_qty)
                            was_adjusted = True
                            direction = "reduced" if protein_delta > 0 else "increased"
                            adjustment_log.append(
                                f"{ing_name}: {quantity:.0f}g → {new_qty:.0f}g ({direction} for protein)"
                            )

                elif priority_macro == "carbs" and carbs_pct > TOLERANCE:
                    if carbs_per_100g >= 15:  # Carb-rich food (>=15g per 100g)
                        carbs_per_g = carbs_per_100g / 100
                        adjustment_g = carbs_delta / carbs_per_g

                        # Skip if adjustment would change quantity by more than 200%
                        if abs(adjustment_g) > quantity * 2:
                            continue

                        # Apply adjustment with caps (min 30% of original, max 3x original)
                        min_qty = max(20, quantity * MIN_SCALE)
                        max_qty = quantity * MAX_SCALE
                        new_qty = max(min_qty, min(max_qty, quantity - adjustment_g))
                        if abs(new_qty - quantity) >= 10:
                            ing["adjusted_quantity_g"] = round(new_qty)
                            was_adjusted = True
                            direction = "reduced" if carbs_delta > 0 else "increased"
                            adjustment_log.append(
                                f"{ing_name}: {quantity:.0f}g → {new_qty:.0f}g ({direction} for carbs)"
                            )

                elif priority_macro == "fat" and fat_pct > TOLERANCE:
                    if fat_per_100g >= 10:  # Fat-rich food (>=10g per 100g)
                        fat_per_g = fat_per_100g / 100
                        adjustment_g = fat_delta / fat_per_g

                        # Skip if adjustment would change quantity by more than 200%
                        if abs(adjustment_g) > quantity * 2:
                            continue

                        # Apply adjustment with caps (min 30% of original, max 3x original)
                        min_qty = max(5, quantity * MIN_SCALE)  # Fats can be smaller (5g min)
                        max_qty = quantity * MAX_SCALE
                        new_qty = max(min_qty, min(max_qty, quantity - adjustment_g))
                        if abs(new_qty - quantity) >= 3:  # Smaller threshold for fats
                            ing["adjusted_quantity_g"] = round(new_qty)
                            was_adjusted = True
                            direction = "reduced" if fat_delta > 0 else "increased"
                            adjustment_log.append(
                                f"{ing_name}: {quantity:.0f}g → {new_qty:.0f}g ({direction} for fat)"
                            )

        # 5. Recalculate totals if adjusted using Passio macros
        if was_adjusted:
            new_protein = 0
            new_carbs = 0
            new_fat = 0
            new_calories = 0

            for meal in adjusted_plan.get("meals", []):
                meal_protein = 0
                meal_carbs = 0
                meal_fat = 0
                meal_calories = 0

                for ing in meal.get("validated_ingredients", []):
                    # Use adjusted quantity if available, otherwise original
                    qty = ing.get("adjusted_quantity_g") or ing.get("quantity_g") or 0

                    # Read Passio macros directly from ingredient schema
                    protein_per_100g = ing.get("protein_per_100g") or 0
                    carbs_per_100g = ing.get("carbs_per_100g") or 0
                    fat_per_100g = ing.get("fat_per_100g") or 0
                    calories_per_100g = ing.get("calories_per_100g") or 0

                    # Calculate macros for this ingredient
                    scale = qty / 100
                    meal_protein += protein_per_100g * scale
                    meal_carbs += carbs_per_100g * scale
                    meal_fat += fat_per_100g * scale
                    meal_calories += calories_per_100g * scale

                # Update meal macros
                meal["protein_g"] = round(meal_protein, 1)
                meal["carbs_g"] = round(meal_carbs, 1)
                meal["fat_g"] = round(meal_fat, 1)
                meal["calories"] = round(meal_calories)

                new_protein += meal_protein
                new_carbs += meal_carbs
                new_fat += meal_fat
                new_calories += meal_calories

            # Update daily totals
            adjusted_plan["daily_totals"] = {
                "protein_g": round(new_protein, 1),
                "carbs_g": round(new_carbs, 1),
                "fat_g": round(new_fat, 1),
                "calories": round(new_calories),
            }

            # Add note about Python adjustment
            existing_notes = adjusted_plan.get("notes", "") or ""
            adjustment_note = f" [Python adjusted: {'; '.join(adjustment_log)}]"
            adjusted_plan["notes"] = existing_notes + adjustment_note

            print(
                f"   🔧 Python quantity adjustment: {'; '.join(adjustment_log)}",
                file=sys.stderr,
            )

        return adjusted_plan, was_adjusted

    def _validate_daily_macros(
        self,
        daily_plan: Dict[str, Any],
        target: Dict[str, Any],
        meal_targets: List[Dict[str, Any]],
        use_relaxed_thresholds: bool = False,
    ) -> Dict[str, Any]:
        """Validate daily totals against targets using unified thresholds.

        Uses centralized validation_config.py thresholds with OR logic:
        pass if EITHER percentage OR absolute tolerance is met.

        Args:
            daily_plan: The generated daily meal plan with daily_totals
            target: The nutritional target for the day
            meal_targets: Per-meal targets from Hexis
            use_relaxed_thresholds: If True, use +50% relaxed thresholds (4th attempt fallback)

        Returns:
            Dict with: passed (bool), reason (str), issues (list), adjustments (list), missing_meals (list)
        """
        from validation_config import (
            check_compliance,
            check_dietary_restrictions,
            get_adaptive_thresholds,
        )

        adjustments: List[str] = []
        missing_meals: List[str] = []

        # =================================================================
        # Dietary Restrictions Check (FIRST - before macro validation)
        # =================================================================
        all_ingredients: List[str] = []
        for meal in daily_plan.get("meals", []):
            # Check ingredient strings
            all_ingredients.extend(meal.get("ingredients", []))
            # Also check validated_ingredients if present
            for vi in meal.get("validated_ingredients", []):
                if isinstance(vi, dict):
                    all_ingredients.append(vi.get("name", "") or "")
                    all_ingredients.append(vi.get("passio_food_name", "") or "")

        dietary_check = check_dietary_restrictions(all_ingredients)
        if not dietary_check["passed"]:
            violation_msgs = [
                f"FORBIDDEN: {v['ingredient']} (matched '{v['matched_keyword']}' - {v['restriction']})"
                for v in dietary_check["violations"]
            ]
            adjustment_msgs = [
                f"Remove {v['ingredient']} and substitute with allowed alternative"
                for v in dietary_check["violations"]
            ]
            print(f"   ❌ Dietary restriction violations: {violation_msgs}", file=sys.stderr)
            return {
                "passed": False,
                "reason": "Dietary restriction violation",
                "issues": violation_msgs,
                "adjustments": adjustment_msgs,
                "missing_meals": [],
            }

        # Log any warnings (non-blocking)
        if dietary_check.get("warnings"):
            for w in dietary_check["warnings"]:
                print(f"   ⚠️ Dietary warning: {w['ingredient']} ({w['restriction']})", file=sys.stderr)

        # Extract daily totals from plan
        daily_totals = daily_plan.get("daily_totals", {})
        if not daily_totals:
            return {
                "passed": False,
                "reason": "No daily_totals in meal plan",
                "issues": ["Missing daily_totals in plan output"],
                "adjustments": ["Ensure Reviewer includes daily_totals"],
                "missing_meals": [],
            }

        # Build actual and target dicts for compliance check
        actual = {
            "calories": daily_totals.get("calories", 0) or 0,
            "protein_g": daily_totals.get("protein_g", 0) or 0,
            "carbs_g": daily_totals.get("carbs_g", 0) or 0,
            "fat_g": daily_totals.get("fat_g", 0) or 0,
        }

        target_macros = target.get("macros", {})
        target_vals = {
            "calories": target.get("calories", 0) or 0,
            "protein_g": target_macros.get("protein_g", 0) or 0,
            "carbs_g": target_macros.get("carbs_g", 0) or 0,
            "fat_g": target_macros.get("fat_g", 0) or 0,
        }

        # Use centralized compliance check (OR logic: pass if either % or abs is met)
        # Get adaptive thresholds: relaxed for 4th attempt, or looser for high-energy days
        target_calories = int(target_vals.get("calories", 0) or 0)
        thresholds = get_adaptive_thresholds(target_calories, use_relaxed=use_relaxed_thresholds)
        compliance = check_compliance(actual, target_vals, thresholds)
        issues = list(compliance["issues"])

        # Generate adjustment suggestions for failed macros
        for macro, info in compliance["details"].items():
            if not info["within_tolerance"]:
                diff = info["diff"]
                if macro == "calories":
                    if diff < 0:
                        needed = abs(diff)
                        adjustments.append(
                            f"Add ~{needed:.0f} kcal: increase pasta/rice by {needed/3.5:.0f}g or add {needed/9:.0f}g olive oil"
                        )
                    else:
                        rice_to_remove = diff / 1.3
                        oil_to_remove = diff / 9
                        adjustments.append(
                            f"Remove ~{diff:.0f} kcal: reduce rice by {rice_to_remove:.0f}g or olive oil by {oil_to_remove:.0f}g"
                        )
                elif macro == "protein_g":
                    if diff < 0:
                        adjustments.append(
                            f"Add {abs(diff):.0f}g protein: add {abs(diff)/0.31:.0f}g chicken breast"
                        )
                    else:
                        chicken_to_remove = diff / 0.31
                        adjustments.append(
                            f"Remove {diff:.0f}g protein: reduce chicken by {chicken_to_remove:.0f}g"
                        )
                elif macro == "carbs_g":
                    if diff < 0:
                        adjustments.append(
                            f"Add {abs(diff):.0f}g carbs: add {abs(diff)/0.75:.0f}g dry pasta"
                        )
                    else:
                        rice_to_remove = diff / 0.23
                        adjustments.append(
                            f"Remove {diff:.0f}g carbs: reduce rice by {rice_to_remove:.0f}g"
                        )
                elif macro == "fat_g":
                    if diff < 0:
                        adjustments.append(
                            f"Add {abs(diff):.0f}g fat: add {abs(diff):.0f}g olive oil"
                        )
                    else:
                        adjustments.append(
                            f"Remove {diff:.0f}g fat: reduce oil/butter portions"
                        )

        # Check meal types coverage
        expected_meal_types = {mt.get("meal_type") for mt in meal_targets}
        generated_meals = daily_plan.get("meals", [])
        generated_meal_types = {m.get("meal_type") for m in generated_meals}
        missing = expected_meal_types - generated_meal_types
        if missing:
            missing_meals = list(missing)
            issues.append(f"Missing meal types: {', '.join(missing_meals)}")
            adjustments.append(f"Add meals for: {', '.join(missing_meals)}")

        passed = compliance["compliant"] and len(missing_meals) == 0
        reason = "All macros within tolerance" if passed else "; ".join(issues[:3])

        return {
            "passed": passed,
            "reason": reason,
            "issues": issues,
            "adjustments": adjustments,
            "missing_meals": missing_meals,
        }

    def _compile_weekly_plan(
        self,
        nutrition_plan: Dict[str, Any],
        daily_plans: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Merge daily plans into a weekly plan using the compilation agent when possible."""

        if self.meal_compilation_agent:
            print("\n🔗 Compiling weekly meal plan from daily outputs...\n", file=sys.stderr)

            try:
                compilation_task = create_meal_compilation_task(
                    self.meal_compilation_agent,
                    nutrition_plan,
                    daily_plans,
                )
                compilation_crew = Crew(
                    agents=[self.meal_compilation_agent],
                    tasks=[compilation_task],
                    process=Process.sequential,
                    verbose=True,
                    max_iter=30,  # Increase max iterations to prevent false loop detection
                    memory=False,  # Disable cache to prevent "reusing same input" errors
                )

                compilation_result = compilation_crew.kickoff()
                weekly_model = self._extract_model_from_output(compilation_result, WeeklyMealPlan)

                if weekly_model is not None:
                    compiled_plan = weekly_model.model_dump()
                    print(
                        f"\n✅ Weekly plan compiled with {len(compiled_plan.get('daily_plans', []))} days\n",
                        file=sys.stderr,
                    )
                    return compiled_plan

                print("\n⚠️  Weekly compilation returned invalid JSON. Falling back to deterministic merge.\n", file=sys.stderr)
            except Exception as exc:
                print(
                    f"\n⚠️  Weekly compilation agent failed ({exc}). Falling back to deterministic merge.\n",
                    file=sys.stderr,
                )

        return self._fallback_compile_weekly_plan(nutrition_plan, daily_plans)

    def _fallback_compile_weekly_plan(
        self,
        nutrition_plan: Dict[str, Any],
        daily_plans: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Deterministically merge daily plans into a WeeklyMealPlan-compliant structure."""

        print("\n🧮 Fallback: Merging daily plans without LLM assistance...\n", file=sys.stderr)

        if not daily_plans:
            print("\n❌ Fallback merge aborted: no daily plans supplied\n", file=sys.stderr)
            return None

        try:
            normalized_daily_plans: List[Dict[str, Any]] = []
            ingredient_counter: Counter[str] = Counter()

            for raw_plan in daily_plans:
                plan = copy.deepcopy(raw_plan)
                meals = plan.get("meals", []) or []

                totals = {
                    "calories": 0,
                    "protein_g": 0.0,
                    "carbs_g": 0.0,
                    "fat_g": 0.0,
                }

                for meal in meals:
                    totals["calories"] += int(meal.get("calories", 0) or 0)
                    totals["protein_g"] += float(meal.get("protein_g", 0.0) or 0.0)
                    totals["carbs_g"] += float(meal.get("carbs_g", 0.0) or 0.0)
                    totals["fat_g"] += float(meal.get("fat_g", 0.0) or 0.0)

                    for ingredient in meal.get("ingredients", []) or []:
                        ingredient_str = ingredient.strip()
                        if ingredient_str:
                            ingredient_counter[ingredient_str] += 1

                plan["daily_totals"] = {
                    "calories": int(round(totals["calories"])),
                    "protein_g": round(totals["protein_g"], 1),
                    "carbs_g": round(totals["carbs_g"], 1),
                    "fat_g": round(totals["fat_g"], 1),
                }

                normalized_daily_plans.append(plan)

            shopping_list = self._build_shopping_list(ingredient_counter)
            meal_prep_tips = self._build_meal_prep_tips(nutrition_plan, normalized_daily_plans, ingredient_counter)

            fallback_plan = {
                "week_start_date": nutrition_plan.get("week_start_date"),
                "week_end_date": nutrition_plan.get("week_end_date"),
                "daily_plans": normalized_daily_plans,
                "shopping_list": shopping_list,
                "meal_prep_tips": meal_prep_tips,
            }

            validated = WeeklyMealPlan.model_validate(fallback_plan)
            print(
                f"\n✅ Fallback merge produced weekly plan with {len(normalized_daily_plans)} days\n",
                file=sys.stderr,
            )
            return validated.model_dump()

        except ValidationError as exc:
            print(
                f"\n❌ Fallback merge failed validation: {exc}\n",
                file=sys.stderr,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"\n❌ Unexpected error during fallback merge: {exc}\n",
                file=sys.stderr,
            )

        return None

    @staticmethod
    def _limit_nutrition_plan_days(
        nutrition_plan: Dict[str, Any],
        max_days: int,
    ) -> Dict[str, Any]:
        """Return a copy of the nutrition plan limited to the first max_days targets."""

        if max_days <= 0:
            return nutrition_plan

        plan = copy.deepcopy(nutrition_plan)
        daily_targets = plan.get("daily_targets") or []

        if not daily_targets:
            return plan

        limited_targets = daily_targets[:max_days]
        plan["daily_targets"] = limited_targets

        last_target = limited_targets[-1] if limited_targets else {}
        last_date = last_target.get("date") if isinstance(last_target, dict) else None
        if last_date:
            plan["week_end_date"] = last_date

        return plan

    @staticmethod
    def _simplify_ingredient_name(raw: str) -> str:
        """Return a simplified ingredient label without leading quantities/units."""
        import re

        if not raw:
            return ""

        text = raw.strip()
        pattern = r"^[0-9]+(?:[.,][0-9]+)?\s*(?:g|kg|mg|ml|l|cl|oz|lb|lbs|cups?|cup|tbsp|tablespoons?|tsp|teaspoons?|scoops?|pieces?|slices?|slice|whole|large|small|medium)\b"
        simplified = re.sub(pattern, "", text, flags=re.IGNORECASE).lstrip("-×x· *")
        return simplified.strip()

    @staticmethod
    def _parse_ingredient(raw: str) -> Tuple[float, str, str]:
        """Parse an ingredient string into quantity, unit, and name."""
        import re
        
        raw = raw.strip()
        if not raw:
            return 0.0, "", ""

        valid_units = {
            # Metric
            "g", "kg", "mg", "ml", "l", "cl", 
            # Imperial / US
            "oz", "lb", "lbs", "cup", "cups", "tbsp", "tablespoon", "tablespoons", 
            "tsp", "teaspoon", "teaspoons", 
            # French
            "c.à.s", "cas", "c.à.c", "cac", "cuillère", "cuillères", 
            "tranche", "tranches", "pincée", "pincées", "poignée", "poignées",
            "gousse", "gousses", "branche", "branches", "feuille", "feuilles",
            "tasse", "tasses", "verre", "verres", "bol", "bols",
            # Common
            "scoop", "scoops", "piece", "pieces", "slice", "slices", "whole", 
            "large", "small", "medium", "can", "cans", "jar", "jars",
            "pack", "packs", "bag", "bags", "box", "boxes", "boîte", "boîtes",
            "sachet", "sachets", "paquet", "paquets"
        }

        # Match: quantity (float/int/fraction) + optional unit + name
        # Examples: "100g chicken", "2.5 tbsp oil", "1/2 cup rice", "onion"
        
        # Handle fractions first (e.g., "1/2")
        fraction_match = re.match(r"^(\d+)/(\d+)\s*(.*)", raw)
        if fraction_match:
            try:
                num, den, rest = fraction_match.groups()
                qty = float(num) / float(den)
                
                # Check for unit in the rest
                parts = rest.strip().split(maxsplit=1)
                if parts:
                    potential_unit = parts[0].lower()
                    # Remove trailing 's' for check if needed, but valid_units has plurals
                    if potential_unit in valid_units:
                        name = parts[1] if len(parts) > 1 else ""
                        return qty, potential_unit, name
                
                return qty, "", rest.strip()
            except (ValueError, ZeroDivisionError):
                pass

        # Standard pattern
        pattern = r"^(\d+(?:[.,]\d+)?)\s*([a-zA-Z.à]+)?\s+(.*)$"
        match = re.match(pattern, raw)
        
        if match:
            qty_str, unit, name = match.groups()
            
            try:
                qty = float(qty_str.replace(",", "."))
                unit = unit.lower() if unit else ""
                
                if unit and unit not in valid_units:
                    # Unit is likely part of the name (e.g. "1 salmon fillet")
                    name = f"{unit} {name}"
                    unit = ""
                
                return qty, unit, name.strip()
            except ValueError:
                pass
        
        # No quantity found, assume 1 unit if it looks countable, else just return name
        return 1.0, "", raw

    @staticmethod
    def _categorize_item(name: str) -> str:
        """Categorize an ingredient based on keywords (French & English)."""
        name = name.lower()
        
        categories = {
            "Fruits & Légumes": [
                "apple", "banana", "pear", "peach", "berry", "fruit", "spinach", "kale", "lettuce", "tomato", "cucumber", "pepper", "onion", "garlic", "ginger", "potato", "carrot", "broccoli", "cauliflower", "zucchini", "mushroom", "herb", "parsley", "cilantro", "basil", "lime", "lemon", "avocado",
                "pomme", "banane", "poire", "pêche", "fruit", "épinard", "chou", "laitue", "salade", "tomate", "concombre", "poivron", "oignon", "ail", "gingembre", "patate", "carotte", "brocoli", "chou-fleur", "courgette", "champignon", "herbe", "persil", "coriandre", "basilic", "citron", "avocat", "aubergine", "haricot vert", "poireau", "courge", "navet", "radis", "céleri", "fenouil"
            ],
            "Viande & Poisson": [
                "chicken", "beef", "pork", "lamb", "turkey", "fish", "salmon", "tuna", "cod", "shrimp", "prawn", "steak", "breast", "thigh", "mince", "sausage", "bacon", "egg",
                "poulet", "boeuf", "porc", "agneau", "dinde", "poisson", "saumon", "thon", "cabillaud", "crevette", "steak", "blanc de poulet", "blanc de dinde", "cuisse", "haché", "saucisse", "lardon", "jambon", "viande", "veau", "canard", "colin", "merlu", "sardine", "maquereau"
            ],
            "Crèmerie & Oeufs": [
                "milk", "yogurt", "cheese", "butter", "cream", "ghee", "kefir", "whey", "casein", "egg",
                "lait", "yaourt", "fromage", "beurre", "crème", "skyr", "faisselle", "blanc battu", "emmental", "mozzarella", "parmesan", "comté", "gruyère", "chèvre", "brebis", "oeuf"
            ],
            "Épicerie Salée": [
                "rice", "pasta", "quinoa", "oat", "couscous", "bread", "flour", "oil", "olive", "coconut", "vinegar", "sauce", "soy", "tamari", "spice", "salt", "pepper", "curry", "paprika", "cumin", "turmeric", "powder", "nut", "seed", "almond", "walnut", "chia", "flax", "peanut", "cashew", "bean", "lentil", "chickpea", "stock", "broth", "can",
                "riz", "pâte", "quinoa", "avoine", "couscous", "pain", "farine", "huile", "olive", "coco", "vinaigre", "sauce", "soja", "épice", "sel", "poivre", "curry", "paprika", "cumin", "curcuma", "poudre", "noix", "graine", "amande", "chia", "lin", "cacahuète", "cajou", "haricot", "lentille", "pois chiche", "bouillon", "conserve", "bocal", "moutarde", "mayonnaise", "cornichon", "thon en boîte", "sardine en boîte"
            ],
            "Épicerie Sucrée": [
                "sugar", "honey", "syrup", "chocolate", "cocoa", "jam", "biscuit", "cookie",
                "sucre", "miel", "sirop", "chocolat", "cacao", "confiture", "biscuit", "gâteau", "vanille", "cannelle", "levure"
            ],
            "Surgelés": ["frozen", "surgelé", "glace", "sorbet"],
            "Boissons": ["water", "juice", "tea", "coffee", "eau", "jus", "thé", "café", "boisson", "sirop"],
        }
        
        for category, keywords in categories.items():
            if any(keyword in name for keyword in keywords):
                return category
                
        return "Autre"

    @staticmethod
    def _convert_to_metric(qty: float, unit: str) -> Tuple[float, str]:
        """Convert Imperial units to Metric (approximate for shopping)."""
        unit = unit.lower()

        conversions = {
            # Weight
            "oz": (28.35, "g"),
            "ounce": (28.35, "g"),
            "ounces": (28.35, "g"),
            "lb": (453.59, "g"),
            "lbs": (453.59, "g"),
            "pound": (453.59, "g"),
            "pounds": (453.59, "g"),

            # Volume
            "cup": (240.0, "ml"),
            "cups": (240.0, "ml"),
            "tasse": (240.0, "ml"), # Assuming US cup size for consistency
            "tasses": (240.0, "ml"),
            "tbsp": (1.0, "c.à.s"),
            "tablespoon": (1.0, "c.à.s"),
            "tablespoons": (1.0, "c.à.s"),
            "tsp": (1.0, "c.à.c"),
            "teaspoon": (1.0, "c.à.c"),
            "teaspoons": (1.0, "c.à.c"),
            "fl oz": (29.57, "ml"),
            "fluid ounce": (29.57, "ml"),

            # Length (rare but possible)
            "inch": (2.54, "cm"),
            "inches": (2.54, "cm"),
        }

        if unit in conversions:
            factor, new_unit = conversions[unit]
            return qty * factor, new_unit

        return qty, unit

    def _normalize_unit_by_category(self, name: str, qty: float, unit: str) -> Tuple[float, str]:
        """
        Normalize units based on ingredient category for better consolidation.

        Strategy:
        - Fruits & Légumes: Prefer pieces (convert g → pieces using average weights)
        - Viande & Poisson: Always in grams
        - Crèmerie: Hybrid (pieces for yogurt/eggs, grams for cheese/butter)
        - Épicerie: Keep as-is (g, ml, c.à.s, c.à.c)
        """
        category = self._categorize_item(name)
        name_lower = name.lower()

        # Average weights for common produce (in grams per piece)
        produce_weights = {
            "courgette": 150,
            "zucchini": 150,
            "tomate": 120,
            "tomato": 120,
            "pomme": 150,
            "apple": 150,
            "banane": 120,
            "banana": 120,
            "poire": 150,
            "pear": 150,
            "orange": 150,
            "citron": 80,
            "lemon": 80,
            "oignon": 120,
            "onion": 120,
            "poivron": 150,
            "pepper": 150,
            "concombre": 300,
            "cucumber": 300,
            "carotte": 80,
            "carrot": 80,
            "aubergine": 250,
            "eggplant": 250,
            "avocat": 150,
            "avocado": 150,
            "patate douce": 200,
            "sweet potato": 200,
            "pomme de terre": 150,
            "potato": 150,
        }

        # Fruits & Légumes: Convert g → pieces when possible
        if category == "Fruits & Légumes":
            # If already in pieces/countable units, keep as-is
            if unit in ["", "pièce", "pièces", "piece", "pieces"]:
                return qty, "pièce"

            # If in grams, try to convert to pieces
            if unit in ["g", "gram", "grams", "gramme", "grammes"]:
                # Find matching produce weight
                for produce_name, avg_weight in produce_weights.items():
                    if produce_name in name_lower:
                        # Convert grams to pieces
                        pieces = qty / avg_weight
                        # Round to nearest 0.5 (half pieces make sense for shopping)
                        pieces = round(pieces * 2) / 2
                        return pieces, "pièce"

                # If no match found and quantity is small (< 500g), assume 1 piece
                if qty <= 500:
                    return 1.0, "pièce"

            # If in kg, convert to grams first, then try pieces
            if unit in ["kg", "kilogram", "kilograms", "kilogramme", "kilogrammes"]:
                gram_qty = qty * 1000
                # Try conversion again
                for produce_name, avg_weight in produce_weights.items():
                    if produce_name in name_lower:
                        pieces = gram_qty / avg_weight
                        pieces = round(pieces * 2) / 2
                        return pieces, "pièce"
                # Keep in grams if large quantity
                return gram_qty, "g"

        # Viande & Poisson: Always in grams
        elif category == "Viande & Poisson":
            if unit in ["kg", "kilogram", "kilograms", "kilogramme", "kilogrammes"]:
                return qty * 1000, "g"
            if unit in ["", "pièce", "pièces", "piece", "pieces"]:
                # Assume average piece weight (e.g., chicken breast = 200g)
                return qty * 200, "g"
            if unit in ["g", "gram", "grams", "gramme", "grammes"]:
                return qty, "g"

        # Crèmerie & Oeufs: Hybrid
        elif category == "Crèmerie & Oeufs":
            if "yaourt" in name_lower or "yogurt" in name_lower or "oeuf" in name_lower or "egg" in name_lower:
                # Yogurts and eggs in pieces
                if unit in ["g", "kg"]:
                    # 1 yogurt ≈ 125g, 1 egg ≈ 60g
                    avg_weight = 125 if "yaourt" in name_lower or "yogurt" in name_lower else 60
                    pieces = qty / avg_weight if unit == "g" else (qty * 1000) / avg_weight
                    return round(pieces), "pièce"
                return qty, "pièce"
            else:
                # Cheese, butter, cream in grams
                if unit in ["kg", "kilogram", "kilogrammes"]:
                    return qty * 1000, "g"
                if unit in ["", "pièce", "pièces"]:
                    # Assume average cheese portion = 30g
                    return qty * 30, "g"
                return qty, unit

        # Default: keep as-is
        return qty, unit

    def _build_shopping_list(self, ingredient_counter: Counter[str]) -> List[str]:
        """Aggregate ingredients into a categorized shopping list."""
        if not ingredient_counter:
            return []

        consolidated: Dict[str, Dict[str, Any]] = {}
        
        for raw_ingredient, count in ingredient_counter.items():
            qty, unit, name = self._parse_ingredient(raw_ingredient)

            # Convert to Metric
            qty, unit = self._convert_to_metric(qty, unit)

            # Normalize name (simple singularization)
            if name.endswith("s") and not name.endswith("ss"):
                name = name[:-1]

            # Normalize units by category for better consolidation
            # (e.g., convert "150g courgette" + "2 courgettes" → all in pieces)
            qty, unit = self._normalize_unit_by_category(name, qty, unit)

            key = f"{name}_{unit}"

            if key not in consolidated:
                consolidated[key] = {"name": name, "unit": unit, "qty": 0.0}

            consolidated[key]["qty"] += qty * count

        # Group by category
        by_category: Dict[str, List[str]] = {}
        
        for item in consolidated.values():
            name = item["name"]
            unit = item["unit"]
            qty = item["qty"]
            
            category = self._categorize_item(name)
            
            # Format quantity
            if qty == int(qty):
                qty_str = str(int(qty))
            else:
                qty_str = f"{qty:.1f}"

            # Format entry with proper pluralization
            if unit:
                # Handle piece/pièce pluralization
                if unit == "pièce":
                    if qty <= 1:
                        entry = f"{name} ({qty_str})"  # Just the number, no unit for single piece
                    else:
                        entry = f"{name} ({qty_str} pièces)"  # Plural for multiple pieces
                else:
                    entry = f"{name} ({qty_str} {unit})"
            else:
                entry = f"{name} ({qty_str})"
                
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(entry)

        # Build final list
        final_list: List[str] = []
        
        # Order categories
        cat_order = ["Fruits & Légumes", "Viande & Poisson", "Crèmerie & Oeufs", "Surgelés", "Épicerie Salée", "Épicerie Sucrée", "Boissons", "Autre"]
        
        for cat in cat_order:
            if cat in by_category:
                items = sorted(by_category[cat])
                for item in items:
                    final_list.append(f"{cat}: {item}")
                    
        return final_list

    def _build_meal_prep_tips(
        self,
        nutrition_plan: Dict[str, Any],
        daily_plans: List[Dict[str, Any]],
        ingredient_counter: Counter[str],
    ) -> List[str]:
        """Generate practical prep tips based on the week's structure."""

        tips: List[str] = []

        grain_keywords = ["rice", "quinoa", "couscous", "pasta", "noodle", "oat", "polenta", "grits"]
        roast_keywords = ["sweet potato", "potato", "carrot", "brussels", "squash", "broccolini", "broccoli"]

        grains = {
            self._simplify_ingredient_name(item)
            for item in ingredient_counter
            if any(keyword in item.lower() for keyword in grain_keywords)
        }
        roasted_items = {
            self._simplify_ingredient_name(item)
            for item in ingredient_counter
            if any(keyword in item.lower() for keyword in roast_keywords)
        }

        training_context = {
            target.get("day_name", ""): target.get("training_context", "")
            for target in nutrition_plan.get("daily_targets", [])
        }

        intense_keywords = ["interval", "vo2", "threshold", "long", "tempo"]
        intense_days = [
            day
            for day, context in training_context.items()
            if any(keyword in context.lower() for keyword in intense_keywords)
        ]

        snack_names = [
            meal.get("meal_name", "Snack")
            for plan in daily_plans
            for meal in plan.get("meals", []) or []
            if "snack" in meal.get("meal_type", "").lower()
        ]

        if grains:
            tips.append(
                "Batch-cook grains such as "
                + ", ".join(sorted(grains))
                + " on Sunday evening to cover quick lunches through mid-week."
            )

        if roasted_items:
            tips.append(
                "Roast vegetables like "
                + ", ".join(sorted(roasted_items))
                + " together on large trays to reuse across dinners."
            )

        if intense_days:
            window = ", ".join(intense_days[:4])
            tips.append(
                f"Prepare protein marinades ahead of {window} to speed up recovery dinners after hard sessions."
            )

        if snack_names:
            sampled = ", ".join(sorted(set(snack_names))[:4])
            tips.append(
                f"Portion snacks such as {sampled} into grab-and-go containers for consistent fueling."
            )

        tips.append(
            "Chop aromatics (garlic, ginger, onions) and store in airtight jars to streamline stir-fries and sauces."
        )

        tips.append(
            "Label leftovers with day and meal type so you can double-check Mealy sync status each evening."
        )

        tips.append(
            "Set calendar reminders to initiate Mealy sync immediately after validation to avoid missed updates."
        )

        # Ensure at least six tips by adding generic guidance if needed
        default_tips = [
            "Prepare breakfast bases (overnight oats, pancake batters) the night before early workouts.",
            "Keep a hydration station with electrolytes ready for long ride and interval days.",
        ]

        for tip in default_tips:
            if len(tips) >= 6:
                break
            tips.append(tip)

        return tips[:8]

    @staticmethod
    def _stringify_output(crew_output: CrewOutput) -> str:
        """Convert a crew output into a string for debugging/fallbacks."""
        if crew_output.raw:
            return crew_output.raw

        if crew_output.json_dict:
            return json.dumps(crew_output.json_dict)

        if crew_output.pydantic:
            return crew_output.pydantic.model_dump_json()

        for task_output in reversed(crew_output.tasks_output or []):
            if task_output.raw:
                return task_output.raw
            if task_output.json_dict:
                return json.dumps(task_output.json_dict)
            if task_output.pydantic:
                return task_output.pydantic.model_dump_json()

        return ""


    @staticmethod
    def _format_telegram_message(daily_plan: Dict[str, Any]) -> str:
        """Format a daily meal plan into a Telegram-friendly HTML message."""
        day_name = daily_plan.get("day_name", "Unknown Day")
        date_str = daily_plan.get("date", "")
        
        # Format date (e.g., 2025-11-24 -> 24/11)
        formatted_date = ""
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                formatted_date = dt.strftime("%d/%m")
            except ValueError:
                formatted_date = date_str

        # Header
        message = f"📅 <b>{day_name} {formatted_date}</b>\n\n"
        
        # Meals
        meals = daily_plan.get("meals", [])
        meal_emojis = {
            "Breakfast": "🥣",
            "Lunch": "🥗",
            "Dinner": "🍽️",
            "Snack": "🍎",
            "Pre-workout": "⚡",
            "Post-workout": "💪"
        }
        
        for meal in meals:
            m_type = meal.get("meal_type", "Meal")
            m_name = meal.get("meal_name", "Unknown")
            emoji = meal_emojis.get(m_type, "🥘")
            
            # Simple bold header for meal type
            message += f"{emoji} <b>{m_type}</b>: {m_name}\n"
            
            # Optional: Add macros for main meals if desired, but keeping it clean for now
            # message += f"   <i>({meal.get('calories', 0)}kcal)</i>\n"
        
        message += "\n"
        
        # Daily Totals
        totals = daily_plan.get("daily_totals", {})
        cals = totals.get("calories", 0)
        p = totals.get("protein_g", 0)
        c = totals.get("carbs_g", 0)
        f = totals.get("fat_g", 0)
        
        message += f"📊 <b>Total</b>: {cals}kcal\n"
        message += f"Protocol: {p}P / {c}C / {f}F"
        
        return message

    @staticmethod
    def _calculate_schedule_timestamp(date_str: str) -> int:
        """Calculate Unix timestamp for 07:00 AM Europe/Paris on the given date."""
        if not date_str:
            return 0
            
        try:
            paris_tz = pytz.timezone("Europe/Paris")
            # Parse date
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # Set to 07:00:00
            dt_7am = dt.replace(hour=7, minute=0, second=0, microsecond=0)
            # Localize to Paris time
            dt_localized = paris_tz.localize(dt_7am)
            # Return timestamp
            return int(dt_localized.timestamp())
        except Exception:
            return 0

    def generate_meal_plan(self, week_start_date: str, days_to_generate: int = 7) -> Dict[str, Any]:
        """
        Generate a complete weekly meal plan.

        Args:
            week_start_date: Start date of the week (ISO format YYYY-MM-DD)
            days_to_generate: Number of consecutive days to generate meal plans for.
                Defaults to 7 (full week) and is capped at the number of available
                daily targets in the nutrition plan.

        Returns:
            Complete meal planning result with integration status
        """
        if days_to_generate <= 0:
            raise ValueError("days_to_generate must be at least 1")

        print(
            f"\n🚀 Starting Meal Planning for week of {week_start_date} (requested {days_to_generate} day(s))\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 0: Sync Integration Data - Ensure latest training data
        # ===================================================================
        print("\n🔄 Step 0: Syncing integration data from intervals.icu...\n", file=sys.stderr)

        # Calculate date range (sync 7 days prior to cover training context)
        try:
            start_dt = datetime.fromisoformat(week_start_date)
            from_date = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")
            end_dt = start_dt + timedelta(days=6)
            to_date = end_dt.strftime("%Y-%m-%d")

            print(f"   Sync window: {from_date} → {to_date}\n", file=sys.stderr)
        except (ValueError, TypeError) as e:
            print(f"⚠️  Invalid date format '{week_start_date}': {e}\n", file=sys.stderr)
            print("   Continuing without sync...\n", file=sys.stderr)
            from_date = None
            to_date = None

        # Find hexis_trigger_integration_sync tool
        sync_tool = None
        if from_date and to_date:
            sync_tool = next(
                (t for t in self.mcp_tools if "hexis_trigger_integration_sync" in t.name.lower()),
                None
            )

        if sync_tool:
            try:
                print("   Calling hexis_trigger_integration_sync...\n", file=sys.stderr)
                sync_result = sync_tool._run(
                    modules=["PLANNED_WORKOUTS", "COMPLETED_WORKOUTS", "WELLNESS"],
                    from_date=from_date,
                    to_date=to_date
                )
                print(f"   ✅ Integration sync complete: {sync_result}\n", file=sys.stderr)
            except Exception as e:
                print(f"   ⚠️  Integration sync failed: {e}\n", file=sys.stderr)
                print("   Continuing with existing Hexis data...\n", file=sys.stderr)
        else:
            if from_date and to_date:
                print("   ⚠️  hexis_trigger_integration_sync tool not available\n", file=sys.stderr)
            print("   Continuing with existing Hexis data...\n", file=sys.stderr)

        # ===================================================================
        # STEP 1: Hexis Analysis - Analyze training data to determine needs
        # Using Supervisor/Executor/Reviewer pattern for reliable data retrieval
        # ===================================================================
        print("\n📊 Step 1: Analyzing Hexis training data (Supervisor/Executor/Reviewer pattern)...\n", file=sys.stderr)

        # ---------------------------------------------------------------
        # STEP 1a: SUPERVISOR - Plan the data retrieval strategy
        # ---------------------------------------------------------------
        print("\n   📋 Step 1a: SUPERVISOR planning data retrieval...\n", file=sys.stderr)

        supervisor_task = create_hexis_data_supervisor_task(
            self.hexis_data_supervisor_agent, week_start_date, num_days=days_to_generate
        )
        supervisor_crew = Crew(
            agents=[self.hexis_data_supervisor_agent],
            tasks=[supervisor_task],
            process=Process.sequential,
            verbose=True,
            max_iter=10,
            memory=False,
        )

        supervisor_result = supervisor_crew.kickoff()

        # Extract the retrieval plan
        retrieval_plan = self._extract_model_from_output(supervisor_result, HexisDataRetrievalPlan)
        if retrieval_plan is None:
            # Fallback: Create a minimal plan if supervisor fails
            print("\n   ⚠️  Supervisor failed to create plan, using default...\n", file=sys.stderr)
            start_dt = datetime.fromisoformat(week_start_date)
            end_dt = start_dt + timedelta(days=days_to_generate - 1)

            # Generate batches (max 3 days per batch) for fallback too
            BATCH_SIZE = 3
            batches = []
            current_start = start_dt
            priority = 1
            while current_start <= end_dt:
                batch_end = min(current_start + timedelta(days=BATCH_SIZE - 1), end_dt)
                batches.append({
                    "tool_name": "hexis__hexis_get_weekly_plan",
                    "parameters": {
                        "start_date": current_start.strftime("%Y-%m-%d"),
                        "end_date": batch_end.strftime("%Y-%m-%d")
                    },
                    "purpose": f"Retrieve data for {current_start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}",
                    "priority": priority
                })
                priority += 1
                current_start = batch_end + timedelta(days=1)

            retrieval_plan = HexisDataRetrievalPlan(
                week_start_date=week_start_date,
                week_end_date=end_dt.strftime("%Y-%m-%d"),
                tool_calls=batches,
                analysis_focus=["training_load", "daily_energy_needs", "macro_targets"],
                special_considerations="Focus on accurate macro extraction"
            )

        retrieval_plan_dict = retrieval_plan.model_dump() if hasattr(retrieval_plan, 'model_dump') else retrieval_plan
        print(f"\n   ✅ Retrieval plan created:\n{json.dumps(retrieval_plan_dict, indent=2)[:1000]}...\n", file=sys.stderr)

        # ---------------------------------------------------------------
        # STEP 1b: EXECUTOR - Execute the planned tool calls
        # ---------------------------------------------------------------
        print("\n   🔧 Step 1b: EXECUTOR retrieving Hexis data...\n", file=sys.stderr)

        # Reset captured Hexis results before running Executor
        # This allows us to capture tool results directly in Python
        reset_hexis_tool_results()

        # ---------------------------------------------------------------
        # STEP 1b: DIRECT PYTHON EXECUTION - Execute each batch tool call
        # ---------------------------------------------------------------
        # NOTE: We bypass the executor agent entirely and call the tool directly
        # in Python. This is more reliable than asking an LLM to execute multiple
        # tool calls, as some models (especially Gemini) hallucinate tool results
        # in their text output instead of making real tool calls.
        print("\n   🔧 Step 1b: Executing Hexis tool calls directly (Python)...\n", file=sys.stderr)

        hexis_tool = next((t for t in self.mcp_tools if "hexis_get_weekly_plan" in t.name.lower()), None)
        if not hexis_tool:
            raise ValueError("hexis_get_weekly_plan tool not found in MCP tools")

        tool_calls = retrieval_plan_dict.get("tool_calls", [])
        for i, batch in enumerate(tool_calls, 1):
            # Parameters are nested under "parameters" key in the tool_call structure
            params = batch.get("parameters", {})
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            print(f"   📦 Batch {i}/{len(tool_calls)}: {start_date} → {end_date}", file=sys.stderr)

            try:
                result = hexis_tool._run(start_date=start_date, end_date=end_date)
                # Parse JSON string if needed
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        pass
                capture_hexis_result(
                    tool_name="hexis__hexis_get_weekly_plan",
                    parameters={"start_date": start_date, "end_date": end_date},
                    result=result,
                    success=True,
                )
                print(f"   ✅ Batch {i} succeeded", file=sys.stderr)
            except Exception as e:
                capture_hexis_result(
                    tool_name="hexis__hexis_get_weekly_plan",
                    parameters={"start_date": start_date, "end_date": end_date},
                    result=None,
                    success=False,
                    error_message=str(e),
                )
                print(f"   ❌ Batch {i} failed: {e}", file=sys.stderr)

        # =====================================================================
        # CRITICAL: Use Python-captured tool results instead of LLM output
        # This avoids truncation issues when LLM tries to copy large raw data
        # =====================================================================
        captured_results = get_hexis_tool_results()
        if captured_results:
            print(f"\n   📦 Using {len(captured_results)} captured Hexis tool result(s)\n", file=sys.stderr)
            # Build raw_data_dict from captured results (same format as RawHexisData)
            raw_data_dict = {
                "week_start_date": week_start_date,
                "week_end_date": (datetime.fromisoformat(week_start_date) + timedelta(days=6)).strftime("%Y-%m-%d"),
                "tool_results": captured_results,
                "total_calls": len(captured_results),
                "successful_calls": sum(1 for r in captured_results if r.get("success", False)),
                "weekly_plan_data": None,  # Will be aggregated from tool_results
                "retrieval_notes": f"Captured {len(captured_results)} batch(es) via Python tool wrapper",
            }
        else:
            # Fallback to LLM output extraction (original behavior)
            print("\n   ⚠️  No captured results, falling back to LLM output extraction...\n", file=sys.stderr)
            raw_hexis_data = self._extract_model_from_output(executor_result, RawHexisData)
            if raw_hexis_data is None:
                # Try to extract raw dict if model extraction fails
                print("\n   ⚠️  Executor model extraction failed, trying raw dict...\n", file=sys.stderr)
                candidates = self._collect_payload_candidates(executor_result)
                if candidates:
                    raw_hexis_data = candidates[0]
                else:
                    error_msg = "Executor failed to retrieve Hexis data"
                    print(f"\n❌ {error_msg}\n", file=sys.stderr)
                    raise ValueError(f"CRITICAL: {error_msg}. Cannot proceed without valid Hexis data.")

            raw_data_dict = raw_hexis_data.model_dump() if hasattr(raw_hexis_data, 'model_dump') else raw_hexis_data

        print(f"\n   ✅ Raw data retrieved (keys: {list(raw_data_dict.keys())})\n", file=sys.stderr)

        # ---------------------------------------------------------------
        # STEP 1c: REVIEWER - Analyze raw data and create final analysis
        # ---------------------------------------------------------------
        print("\n   📊 Step 1c: REVIEWER analyzing data...\n", file=sys.stderr)

        reviewer_task = create_hexis_analysis_reviewer_task(
            self.hexis_analysis_reviewer_agent, week_start_date, raw_data_dict
        )
        reviewer_crew = Crew(
            agents=[self.hexis_analysis_reviewer_agent],
            tasks=[reviewer_task],
            process=Process.sequential,
            verbose=True,
            max_iter=15,
            memory=False,
        )

        reviewer_result = reviewer_crew.kickoff()

        # DEBUG: Show raw output length
        raw_output = reviewer_result.raw if hasattr(reviewer_result, 'raw') else ""
        print(f"\n🔍 DEBUG: Raw output length = {len(raw_output)} chars\n", file=sys.stderr)
        if raw_output:
            print(f"🔍 DEBUG: First 500 chars:\n{raw_output[:500]}\n", file=sys.stderr)
            print(f"🔍 DEBUG: Last 500 chars:\n{raw_output[-500:]}\n", file=sys.stderr)

        hexis_model = self._extract_model_from_output(reviewer_result, HexisWeeklyAnalysis)

        if hexis_model is None:
            error_msg = "Hexis analysis reviewer failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            candidates = self._collect_payload_candidates(reviewer_result)
            print(f"❌ DEBUG: Parsing failed. Candidates found: {len(candidates)}\n", file=sys.stderr)

            # Show candidate structure (truncated)
            if candidates:
                candidate_json = json.dumps(candidates[0], indent=2)[:2000]
                print(f"❌ DEBUG: First candidate structure:\n{candidate_json}...\n", file=sys.stderr)

            # CRITICAL: Crash if Hexis analysis fails - no fallback allowed
            raise ValueError(f"CRITICAL: {error_msg}. Cannot proceed without valid Hexis data.")

        hexis_analysis = hexis_model.model_dump()
        print(
            f"\n✅ Hexis analysis complete:\n{json.dumps(hexis_analysis, indent=2)}\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 2: Weekly Structure - Create structured nutrition plan
        # ===================================================================
        print("\n📅 Step 2: Creating weekly nutrition structure...\n", file=sys.stderr)

        structure_task = create_weekly_structure_task(
            self.weekly_structure_agent, hexis_analysis
        )
        structure_crew = Crew(
            agents=[self.weekly_structure_agent],
            tasks=[structure_task],
            process=Process.sequential,
            verbose=True,
            max_iter=30,  # Increase max iterations to prevent false loop detection
            memory=False,  # Disable cache to prevent "reusing same input" errors
        )

        structure_result = structure_crew.kickoff()

        # DEBUG: Show raw output for weekly structure
        raw_output = structure_result.raw if hasattr(structure_result, 'raw') else ""
        print(f"\n🔍 DEBUG: Weekly structure raw output length = {len(raw_output)} chars\n", file=sys.stderr)
        if raw_output:
            print(f"🔍 DEBUG: First 1000 chars:\n{raw_output[:1000]}\n", file=sys.stderr)
            print(f"🔍 DEBUG: Last 500 chars:\n{raw_output[-500:]}\n", file=sys.stderr)

        structure_model = self._extract_model_from_output(
            structure_result, WeeklyNutritionPlan
        )

        if structure_model is None:
            error_msg = "Weekly structure failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)

            # DEBUG: Show candidate structure
            candidates = self._collect_payload_candidates(structure_result)
            print(f"❌ DEBUG: Parsing failed. Candidates found: {len(candidates)}\n", file=sys.stderr)
            if candidates:
                candidate_json = json.dumps(candidates[0], indent=2)[:2000]
                print(f"❌ DEBUG: First candidate structure:\n{candidate_json}...\n", file=sys.stderr)

            return {"error": error_msg, "step": "weekly_structure"}

        nutrition_plan_full = structure_model.model_dump()

        # Fix any LLM-generated day_name errors using deterministic date conversion
        nutrition_plan_full = _fix_daily_target_day_names(nutrition_plan_full)

        original_target_count = len(nutrition_plan_full.get("daily_targets", []))

        days_to_process = max(1, days_to_generate)
        if original_target_count:
            if days_to_generate < original_target_count:
                print(
                    f"\nℹ️  User requested {days_to_generate} day(s); trimming nutrition plan from {original_target_count} day(s) to match request\n",
                    file=sys.stderr,
                )
            elif days_to_generate > original_target_count:
                print(
                    f"\nℹ️  Requested {days_to_generate} day(s) but Hexis provided only {original_target_count} target(s); limiting to available days\n",
                    file=sys.stderr,
                )

            days_to_process = max(1, min(days_to_generate, original_target_count))
            nutrition_plan = self._limit_nutrition_plan_days(
                nutrition_plan_full,
                days_to_process,
            )
        else:
            nutrition_plan = nutrition_plan_full

        print(
            f"\n✅ Nutrition plan created:\n{json.dumps(nutrition_plan, indent=2)}\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 2b: Extract per-meal targets DETERMINISTICALLY from raw Hexis data
        # ===================================================================
        # NOTE: We extract meal targets directly from raw_data_dict instead of
        # relying on the LLM to generate them. This is more reliable because:
        # 1. LLM can truncate output (e.g., only 4/7 days)
        # 2. The data already exists in the raw Hexis response
        # 3. Deterministic extraction ensures ALL days are present
        daily_meal_targets = _extract_daily_meal_targets(raw_data_dict)
        if not daily_meal_targets:
            error_msg = "Hexis data incomplete: daily_meal_targets could not be extracted from raw data"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            raise ValueError(error_msg)

        print(
            f"\n✅ Deterministically extracted meal targets for {len(daily_meal_targets)} days\n",
            file=sys.stderr,
        )

        # Enrich each daily_target with meal_targets from Hexis
        for daily_target in nutrition_plan.get("daily_targets", []):
            day_date = daily_target.get("date", "")
            day_meal_data = daily_meal_targets.get(day_date, {})
            meals = day_meal_data.get("meals", [])

            if not meals:
                error_msg = f"Hexis data incomplete: no meal targets for {day_date}"
                print(f"\n❌ {error_msg}\n", file=sys.stderr)
                raise ValueError(error_msg)

            # Add meal_targets to the daily target
            daily_target["meal_targets"] = meals
            print(
                f"   ✅ Added {len(meals)} meal targets for {day_date}",
                file=sys.stderr,
            )

        print(
            f"\n✅ Per-meal targets from Hexis added to nutrition plan\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 3: Daily Meal Generation and Weekly Compilation (with retry)
        # ===================================================================
        print("\n👨‍🍳 Step 3: Generating meals day by day...\n", file=sys.stderr)

        available_daily_targets = len(nutrition_plan.get("daily_targets", []))
        if available_daily_targets == 0:
            error_msg = "Nutrition plan did not provide any daily targets"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "daily_meal_generation"}

        days_to_process = min(days_to_process, available_daily_targets)
        if days_to_generate > available_daily_targets:
            print(
                f"\nℹ️  Requested {days_to_generate} day(s) but only {available_daily_targets} target(s) provided; generating all available days\n",
                file=sys.stderr,
            )

        print(
            f"\nℹ️  Generating meal plans for {days_to_process} consecutive day(s)\n",
            file=sys.stderr,
        )

        # Retry loop for meal generation + validation
        max_validation_retries = int(os.getenv("MEAL_VALIDATION_MAX_RETRIES", "2"))
        validation_feedback: Optional[Dict[str, Any]] = None
        meal_plan: Optional[Dict[str, Any]] = None
        validation: Dict[str, Any] = {"approved": False, "validation_summary": "Not yet validated"}
        # Track successful days between retries to avoid regenerating them
        previous_successful_plans: Optional[List[Dict[str, Any]]] = None

        for validation_attempt in range(1, max_validation_retries + 1):
            attempt_label = f"[Attempt {validation_attempt}/{max_validation_retries}]"

            if validation_attempt > 1:
                print(
                    f"\n🔄 {attempt_label} Regenerating meals with validation feedback...\n",
                    file=sys.stderr,
                )

            daily_plans = self._generate_daily_meal_plans(
                nutrition_plan,
                max_days=days_to_process,
                validation_feedback=validation_feedback,
                previous_successful_plans=previous_successful_plans,
            )
            # Track successful plans for next retry (if needed)
            if daily_plans:
                previous_successful_plans = daily_plans

            if daily_plans is None:
                error_msg = f"{attempt_label} Daily meal generation failed after all attempts"
                print(f"\n❌ {error_msg}\n", file=sys.stderr)
                if validation_attempt == max_validation_retries:
                    return {"error": error_msg, "step": "daily_meal_generation"}
                continue

            meal_plan = self._compile_weekly_plan(nutrition_plan, daily_plans)

            if meal_plan is None:
                error_msg = f"{attempt_label} Meal compilation failed to produce a valid weekly plan"
                print(f"\n❌ {error_msg}\n", file=sys.stderr)
                if validation_attempt == max_validation_retries:
                    return {"error": error_msg, "step": "meal_compilation"}
                continue

            # ===================================================================
            # STEP 3B: Nutritional validation of compiled plan
            # ===================================================================
            print(f"\n🔍 {attempt_label} Validating compiled meal plan...\n", file=sys.stderr)

            planned_day_count = 0
            if isinstance(meal_plan, dict):
                planned_day_count = len(meal_plan.get("daily_plans", []))
            if planned_day_count <= 0:
                planned_day_count = days_to_generate

            validation_task = create_nutritional_validation_task(
                self.nutritional_validation_agent,
                meal_plan,
                nutrition_plan,
                planned_day_count=planned_day_count,
            )
            validation_crew = Crew(
                agents=[self.nutritional_validation_agent],
                tasks=[validation_task],
                process=Process.sequential,
                verbose=True,
                max_iter=30,
                memory=False,
            )

            validation_result = validation_crew.kickoff()
            validation_model = self._extract_model_from_output(
                validation_result, NutritionalValidation
            )

            if validation_model is None:
                print(f"\n⚠️  {attempt_label} Validation returned invalid JSON\n", file=sys.stderr)
                validation = {
                    "approved": False,
                    "validation_summary": "Validation failed to return valid JSON",
                }
            else:
                validation = validation_model.model_dump()

            is_approved = validation.get("approved", False)
            print(
                f"\n{'✅' if is_approved else '❌'} {attempt_label} Validation status: Approved={is_approved}\n",
                file=sys.stderr,
            )

            if is_approved:
                print(f"\n✅ Meal plan approved on attempt {validation_attempt}\n", file=sys.stderr)
                break

            # Not approved - prepare feedback for next attempt
            if validation_attempt < max_validation_retries:
                issues = validation.get("issues_found", [])
                recommendations = validation.get("recommendations", [])
                macro_accuracy = validation.get("macro_accuracy", {})

                validation_feedback = {
                    "previous_attempt": validation_attempt,
                    "issues_found": issues,
                    "recommendations": recommendations,
                    "macro_accuracy": macro_accuracy,
                    "validation_summary": validation.get("validation_summary", ""),
                }

                print(
                    f"\n⚠️  {attempt_label} Validation failed. Issues found:\n",
                    file=sys.stderr,
                )
                for issue in issues[:5]:
                    print(f"   • {issue}", file=sys.stderr)
                print(
                    f"\n   Recommendations for next attempt:\n",
                    file=sys.stderr,
                )
                for rec in recommendations[:5]:
                    print(f"   → {rec}", file=sys.stderr)
                print("", file=sys.stderr)
            else:
                print(
                    f"\n⚠️  Max validation retries ({max_validation_retries}) reached. Proceeding with current plan.\n",
                    file=sys.stderr,
                )

        if meal_plan is None:
            error_msg = "Meal generation failed after all validation retries"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "meal_generation_validation_loop"}

        print(
            f"\n✅ Final validation status: Approved={validation.get('approved', False)}\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 4: Hexis Integration - Sync meals to Hexis (only if approved)
        # ===================================================================
        print("\n🔗 Step 4: Integrating with Hexis...\n", file=sys.stderr)
        
        integration_task = create_mealy_integration_task(
            self.mealy_integration_agent,
            meal_plan,
            validation,
            planned_day_count=planned_day_count,
        )
        integration_crew = Crew(
            agents=[self.mealy_integration_agent],
            tasks=[integration_task],
            process=Process.sequential,
            verbose=True,
            max_iter=30,  # Increase max iterations to prevent false loop detection
            memory=False,  # Disable cache to prevent "reusing same input" errors
        )

        integration_result = integration_crew.kickoff()
        integration_model = self._extract_model_from_output(
            integration_result, MealyIntegrationResult
        )

        if integration_model is None:
            error_msg = "Mealy integration failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "mealy_integration"}

        integration = integration_model.model_dump()
        print(
            f"\n✅ Integration complete: {integration.get('total_meals_created', 0)} meals created\n",
            file=sys.stderr,
        )

        # ===================================================================
        # FINAL RESULT
        # ===================================================================
        # Extract shopping_list from meal_plan for n8n workflow
        shopping_list = meal_plan.get("shopping_list", []) if isinstance(meal_plan, dict) else []

        final_result = {
            "week_start_date": week_start_date,
            "hexis_analysis": hexis_analysis,
            "nutrition_plan": nutrition_plan,
            "meal_plan": meal_plan,
            "validation": validation,
            "integration": integration,
            "shopping_list": shopping_list,  # Exposed at top level for n8n
            "summary": integration.get("summary", "Meal planning completed"),
        }

        # Inject Telegram formatting into daily plans
        if meal_plan and "daily_plans" in meal_plan:
            for day in meal_plan["daily_plans"]:
                day["telegram_message"] = self._format_telegram_message(day)
                day["schedule_timestamp"] = self._calculate_schedule_timestamp(day.get("date"))

        return final_result
