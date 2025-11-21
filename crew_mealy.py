"""Main Crew definition for weekly meal plan generation."""
from __future__ import annotations

# CRITICAL: Initialize auth BEFORE importing CrewAI to ensure structured outputs work
import llm_auth_init  # noqa: F401

import copy
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Type

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
)
from schemas import (
    HexisWeeklyAnalysis,
    WeeklyNutritionPlan,
    DailyMealPlan,
    WeeklyMealPlan,
    NutritionalValidation,
    MealyIntegrationResult,
)
from tasks import (
    create_hexis_analysis_task,
    create_weekly_structure_task,
    create_meal_generation_task,
    create_meal_compilation_task,
    create_nutritional_validation_task,
    create_mealy_integration_task,
)
from tasks.hexis_analysis_task_fallback import create_hexis_analysis_task_fallback
# NOTE: load_catalog_tool_names kept for compatibility but currently unused
from mcp_utils import build_mcp_references, load_catalog_tool_names
from mcp_auth_wrapper import MetaMCPAdapter
from llm_provider_rotation import create_llm_with_rotation


class MealPlanningCrew:
    """Crew for generating and validating weekly meal plans."""

    def __init__(self):
        """Initialize the crew with LLM and agents."""
        # Environment and auth already configured at module level via llm_auth_init

        # Get configuration
        base_url = os.getenv("OPENAI_API_BASE", "https://ccproxy.emottet.com/v1")
        default_complex_model = os.getenv("OPENAI_MODEL_NAME") or "claude-sonnet-4-5"
        complex_model_name = os.getenv(
            "OPENAI_COMPLEX_MODEL_NAME",
            default_complex_model,
        )
        simple_model_name = os.getenv(
            "OPENAI_SIMPLE_MODEL_NAME",
            "claude-haiku-4.5",
        )

        # Get configured API key (already set by llm_auth_init)
        api_key = os.getenv("OPENAI_API_KEY", "dummy-key")

        # Create per-agent LLMs with optional overrides
        # This allows each agent to use a different model/endpoint for cost optimization
        self.hexis_analysis_llm = self._create_agent_llm(
            agent_name="HEXIS_ANALYSIS",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.weekly_structure_llm = self._create_agent_llm(
            agent_name="WEEKLY_STRUCTURE",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.meal_generation_llm = self._create_agent_llm(
            agent_name="MEAL_GENERATION",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.meal_compilation_llm = self._create_agent_llm(
            agent_name="MEAL_COMPILATION",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.nutritional_validation_llm = self._create_agent_llm(
            agent_name="NUTRITIONAL_VALIDATION",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.mealy_integration_llm = self._create_agent_llm(
            agent_name="MEALY_INTEGRATION",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key
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
        # Hexis Analysis Agent: needs Hexis, Intervals.icu, and Toolbox
        hexis_analysis_tools = hexis_tools + intervals_tools + toolbox_tools
        self.hexis_analysis_agent = create_hexis_analysis_agent(
            self.hexis_analysis_llm, tools=hexis_analysis_tools if hexis_analysis_tools else None
        )

        # Weekly Structure Agent: no MCP tools needed (pure reasoning)
        self.weekly_structure_agent = create_weekly_structure_agent(self.weekly_structure_llm)

        # Meal Generation Agent: needs all food-related tools
        meal_generation_tools = hexis_tools + food_data_tools + toolbox_tools
        self.meal_generation_agent = create_meal_generation_agent(
            self.meal_generation_llm, tools=meal_generation_tools if meal_generation_tools else None
        )

        self.meal_compilation_agent = create_meal_compilation_agent(self.meal_compilation_llm)

        # Nutritional Validation Agent: pure reasoning agent (NO TOOLS)
        # It analyzes the meal plan and nutrition targets provided in the task description
        # without needing to call external APIs
        self.nutritional_validation_agent = create_nutritional_validation_agent(
            self.nutritional_validation_llm, tools=None
        )

        # Hexis Integration Agent: needs Hexis tools for meal verification
        self.mealy_integration_agent = create_mealy_integration_agent(
            self.mealy_integration_llm, tools=hexis_tools if hexis_tools else None
        )

    def _create_agent_llm(
        self,
        agent_name: str,
        default_model: str,
        default_base: str,
        default_key: str,
    ) -> LLM:
        """Create an LLM for a specific agent with optional per-agent overrides.

        Supports environment variables:
        - {AGENT_NAME}_AGENT_MODEL: Model name for this agent
        - {AGENT_NAME}_AGENT_API_BASE: API base URL for this agent
        - {AGENT_NAME}_BLACKLISTED_MODELS: Comma-separated list of models to blacklist

        Args:
            agent_name: Name prefix for env vars (e.g., "HEXIS_ANALYSIS", "MEAL_GENERATION")
            default_model: Fallback model name
            default_base: Fallback API base URL
            default_key: API key to use

        Returns:
            Configured LLM instance
        """
        # Check for agent-specific overrides
        model_key = f"{agent_name}_AGENT_MODEL"
        base_key = f"{agent_name}_AGENT_API_BASE"
        blacklist_key = f"{agent_name}_BLACKLISTED_MODELS"

        agent_model = os.getenv(model_key, default_model)
        agent_base = os.getenv(base_key, default_base)

        # Get blacklisted models from global and agent-specific sources
        global_blacklist = os.getenv("GLOBAL_BLACKLISTED_MODELS", "").split(",") if os.getenv("GLOBAL_BLACKLISTED_MODELS") else []
        agent_blacklist = os.getenv(blacklist_key, "").split(",") if os.getenv(blacklist_key) else []

        # Combine and clean blacklists (agent-specific takes precedence)
        all_blacklisted = [m.strip() for m in global_blacklist + agent_blacklist if m.strip()]

        # Model blacklist for specific agents
        if agent_name == "HEXIS_ANALYSIS" and not agent_blacklist:
            # Default blacklist for HEXIS_ANALYSIS to avoid GPT-5 issues
            default_blacklist = ["gpt-5", "gpt-5-codex"]
            print(f"🚫 {agent_name} Agent: Using default blacklist for {default_blacklist}", file=sys.stderr)
            all_blacklisted.extend(default_blacklist)

        # Check if model is blacklisted
        if agent_model in all_blacklisted:
            source = "global" if agent_model in global_blacklist else ("default" if agent_name == "HEXIS_ANALYSIS" and not agent_blacklist else "agent-specific")
            print(f"⚠️  {agent_name} Agent: Model '{agent_model}' is blacklisted ({source}), using fallback", file=sys.stderr)
            # Use safer fallback models
            agent_model = "claude-sonnet-4.5"
            if "codex" in agent_base.lower():
                agent_model = "gpt-5-mini"  # Safer alternative for codex endpoint

        # Show blacklist info for transparency
        if all_blacklisted:
            print(f"   🚫 Blacklist active: {', '.join(all_blacklisted[:3])}{'...' if len(all_blacklisted) > 3 else ''}", file=sys.stderr)

        # Log configuration for transparency
        if os.getenv(model_key) or os.getenv(base_key):
            print(
                f"🔧 {agent_name} Agent: Using config "
                f"(model={agent_model}, base={agent_base})",
                file=sys.stderr,
            )

        return self._create_llm(
            agent_model,
            agent_base,
            default_key,
            agent_name=agent_name,
        )

    @staticmethod
    def _create_llm(
        model_name: str,
        api_base: str,
        api_key: str,
        agent_name: str,
    ) -> LLM:
        """Instantiate an LLM with provider rotation support when enabled."""

        return create_llm_with_rotation(
            agent_name=agent_name,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
        )

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Remove markdown fences and whitespace from JSON text."""
        import re
        # Remove markdown code fences (anywhere in text, including with leading/trailing whitespace)
        text = re.sub(r'```json\s*\n?', '', text)  # Remove opening fence
        text = re.sub(r'\n?\s*```', '', text)  # Remove closing fence
        return text.strip()

    @staticmethod
    def _payloads_from_task_output(task_output: TaskOutput) -> List[Dict[str, Any]]:
        """Extract potential JSON payloads from a task output."""
        payloads: List[Dict[str, Any]] = []

        if task_output.json_dict:
            payloads.append(task_output.json_dict)

        if task_output.pydantic:
            payloads.append(task_output.pydantic.model_dump())

        raw_value = task_output.raw
        if raw_value:
            try:
                # Clean markdown fences before parsing
                cleaned = MealPlanningCrew._clean_json_text(raw_value)
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
                    except json.JSONDecodeError:
                        try:
                            potential, _ = decoder.raw_decode(normalized)
                            if isinstance(potential, dict):
                                parsed_raw = potential
                        except json.JSONDecodeError:
                            continue

                    if parsed_raw is not None:
                        payloads.append(parsed_raw)
            except json.JSONDecodeError:
                pass

        return payloads

    def _collect_payload_candidates(
        self, crew_output: CrewOutput
    ) -> List[Dict[str, Any]]:
        """Collect payload candidates across crew and individual task outputs."""
        candidates: List[Dict[str, Any]] = []

        if crew_output.json_dict:
            candidates.append(crew_output.json_dict)

        if crew_output.pydantic:
            candidates.append(crew_output.pydantic.model_dump())

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

    def _generate_daily_meal_plans(
        self,
        nutrition_plan: Dict[str, Any],
        max_days: Optional[int] = None,
        validation_feedback: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate daily meal plans sequentially to keep tasks focused."""
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

        generated_daily_plans: List[Dict[str, Any]] = []

        for target in daily_targets:
            day_name = target.get("day_name", "Unknown day")
            day_date = target.get("date", "?")
            print(
                f"\n📆 Generating meals for {day_name} ({day_date})...\n",
                file=sys.stderr,
            )

            max_attempts = 3
            daily_plan: Optional[Dict[str, Any]] = None

            for attempt in range(1, max_attempts + 1):
                print(
                    f"\n🔄 {day_name} - Attempt {attempt}/{max_attempts}...\n",
                    file=sys.stderr,
                )

                meal_task = create_meal_generation_task(
                    self.meal_generation_agent,
                    daily_target=target,
                    weekly_context=nutrition_plan,
                    previous_days=generated_daily_plans,
                    validation_feedback=validation_feedback,
                )
                meal_crew = Crew(
                    agents=[self.meal_generation_agent],
                    tasks=[meal_task],
                    process=Process.sequential,
                    verbose=True,
                    max_iter=30,  # Increase max iterations to prevent false loop detection
                    memory=False,  # Disable cache to prevent "reusing same input" errors
                )

                meal_result = meal_crew.kickoff()
                meal_model = self._extract_model_from_output(meal_result, DailyMealPlan)

                if meal_model is None:
                    print(
                        f"\n⚠️  {day_name} - Attempt {attempt}: Invalid JSON, retrying\n",
                        file=sys.stderr,
                    )
                    continue

                daily_plan = meal_model.model_dump()
                print(
                    f"\n✅ {day_name} - Attempt {attempt}: Generated {len(daily_plan.get('meals', []))} meals\n",
                    file=sys.stderr,
                )
                break

            if daily_plan is None:
                print(
                    f"\n❌ {day_name}: Failed to produce a valid daily plan after {max_attempts} attempts\n",
                    file=sys.stderr,
                )
                return None

            generated_daily_plans.append(daily_plan)

        print(
            f"\n✅ Generated {len(generated_daily_plans)} daily plans successfully\n",
            file=sys.stderr,
        )
        return generated_daily_plans

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

    def _build_shopping_list(self, ingredient_counter: Counter[str]) -> List[str]:
        """Aggregate ingredients into a shopping list with simple deduplication."""

        if not ingredient_counter:
            return []

        shopping_entries: List[str] = []
        for ingredient, count in ingredient_counter.most_common():
            label = ingredient
            if count > 1:
                label = f"{count}× {ingredient}"
            shopping_entries.append(label)

        return sorted(shopping_entries, key=str.lower)

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
        # STEP 1: Hexis Analysis - Analyze training data to determine needs
        # ===================================================================
        print("\n📊 Step 1: Analyzing Hexis training data...\n", file=sys.stderr)

        # DIRECT FALLBACK: Use simplified task without tools to avoid infinite loops
        print(f"\n🔄 Using direct fallback task without tool calls to avoid loops...\n", file=sys.stderr)
        hexis_task = create_hexis_analysis_task_fallback(
            self.hexis_analysis_agent, week_start_date
        )
        hexis_crew = Crew(
            agents=[self.hexis_analysis_agent],
            tasks=[hexis_task],
            process=Process.sequential,
            verbose=True,
            max_iter=30,  # Increase max iterations to prevent false loop detection
            memory=False,  # Disable cache to prevent "reusing same input" errors
        )

        hexis_result = hexis_crew.kickoff()

        # DEBUG: Show raw output length
        raw_output = hexis_result.raw if hasattr(hexis_result, 'raw') else ""
        print(f"\n🔍 DEBUG: Raw output length = {len(raw_output)} chars\n", file=sys.stderr)
        if raw_output:
            print(f"🔍 DEBUG: First 500 chars:\n{raw_output[:500]}\n", file=sys.stderr)
            print(f"🔍 DEBUG: Last 500 chars:\n{raw_output[-500:]}\n", file=sys.stderr)

        hexis_model = self._extract_model_from_output(hexis_result, HexisWeeklyAnalysis)

        if hexis_model is None:
            error_msg = "Hexis analysis failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            candidates = self._collect_payload_candidates(hexis_result)
            print(f"❌ DEBUG: Parsing failed. Candidates found: {len(candidates)}\n", file=sys.stderr)

            # Show candidate structure (truncated)
            if candidates:
                candidate_json = json.dumps(candidates[0], indent=2)[:2000]
                print(f"❌ DEBUG: First candidate structure:\n{candidate_json}...\n", file=sys.stderr)

            # FALLBACK: Use simplified task without tools
            print(f"\n🔄 Using fallback task without tool calls...\n", file=sys.stderr)

            hexis_fallback_task = create_hexis_analysis_task_fallback(
                self.hexis_analysis_agent, week_start_date
            )
            hexis_fallback_crew = Crew(
                agents=[self.hexis_analysis_agent],
                tasks=[hexis_fallback_task],
                process=Process.sequential,
                verbose=True,
                max_iter=5,
                memory=False,
            )

            hexis_fallback_result = hexis_fallback_crew.kickoff()
            hexis_model = self._extract_model_from_output(hexis_fallback_result, HexisWeeklyAnalysis)

            if hexis_model is None:
                print(f"\n❌ Even fallback task failed\n", file=sys.stderr)
                return {"error": f"{error_msg} (fallback also failed)", "step": "hexis_analysis"}
            else:
                print(f"\n✅ Fallback task succeeded\n", file=sys.stderr)

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
            )

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

        return final_result
