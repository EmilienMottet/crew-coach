"""Main Crew definition for weekly meal plan generation."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from crewai import Crew, LLM, Process
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput

from agents import (
    create_hexis_analysis_agent,
    create_weekly_structure_agent,
    create_meal_generation_agent,
    create_nutritional_validation_agent,
    create_mealy_integration_agent,
)
from schemas import (
    HexisWeeklyAnalysis,
    WeeklyNutritionPlan,
    WeeklyMealPlan,
    NutritionalValidation,
    MealyIntegrationResult,
)
from tasks import (
    create_hexis_analysis_task,
    create_weekly_structure_task,
    create_meal_generation_task,
    create_nutritional_validation_task,
    create_mealy_integration_task,
)
from mcp_utils import build_mcp_references, load_catalog_tool_names


class MealPlanningCrew:
    """Crew for generating and validating weekly meal plans."""

    def __init__(self):
        """Initialize the crew with LLM and agents."""
        load_dotenv()

        # Configure environment variables for LiteLLM/OpenAI
        base_url = os.getenv("OPENAI_API_BASE", "https://ghcopilot.emottet.com/v1")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5mini")

        # Configure authentication - support both API key and Basic Auth
        auth_token = os.getenv("OPENAI_API_AUTH_TOKEN")  # Base64-encoded Basic Auth

        # Set environment variables that LiteLLM expects
        os.environ["OPENAI_API_BASE"] = base_url

        # Configure litellm/OpenAI authentication (same pattern as crew.py)
        if auth_token:
            basic_token = auth_token
            if not auth_token.startswith("Basic "):
                basic_token = f"Basic {auth_token}"
            os.environ["OPENAI_API_KEY"] = basic_token

            # Patch OpenAI client classes to honour Basic auth tokens
            from openai import AsyncOpenAI, OpenAI  # type: ignore
            from litellm.llms.openai.openai import OpenAIConfig  # type: ignore

            def _basic_auth_headers(self: Any) -> Dict[str, str]:
                key = getattr(self, "api_key", "") or ""
                if isinstance(key, str) and key.startswith("Basic "):
                    return {"Authorization": key}
                return {"Authorization": f"Bearer {key}"}

            if not getattr(OpenAI, "_basic_auth_patched", False):
                OpenAI.auth_headers = property(_basic_auth_headers)  # type: ignore[assignment]
                AsyncOpenAI.auth_headers = property(_basic_auth_headers)  # type: ignore[assignment]
                setattr(OpenAI, "_basic_auth_patched", True)
                setattr(AsyncOpenAI, "_basic_auth_patched", True)

            if not getattr(OpenAIConfig, "_basic_auth_patched", False):
                original_validate_env = OpenAIConfig.validate_environment

                def validate_environment_with_basic(
                    self: Any,
                    headers: Dict[str, str],
                    model: str,
                    messages: List[Any],
                    optional_params: Dict[str, Any],
                    api_key: Optional[str] = None,
                    api_base: Optional[str] = None,
                ) -> Dict[str, str]:
                    result = original_validate_env(
                        self,
                        headers,
                        model,
                        messages,
                        optional_params,
                        api_key=api_key,
                        api_base=api_base,
                    )
                    if isinstance(api_key, str) and api_key.startswith("Basic "):
                        result["Authorization"] = api_key
                    return result

                OpenAIConfig.validate_environment = validate_environment_with_basic  # type: ignore[assignment]
                setattr(OpenAIConfig, "_basic_auth_patched", True)

            api_key = basic_token
        else:
            # Use standard API key
            api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
            os.environ["OPENAI_API_KEY"] = api_key

        # Create LLM instance
        self.llm = LLM(
            model=f"openai/{model_name}",
            api_base=base_url,
            api_key=api_key,
        )

        # Configure MCP references for Hexis tools
        self.hexis_mcps = self._build_hexis_mcp_references()
        if not self.hexis_mcps:
            print(
                "\n⚠️  Warning: No MCP references configured for Hexis tools. "
                "Hexis analysis will be limited without live training data.\n",
                file=sys.stderr,
            )
        else:
            print(
                f"\n🏃 Hexis MCP references configured: {', '.join(self.hexis_mcps)}\n",
                file=sys.stderr,
            )

        # Configure MCP references for Mealy tools
        self.mealy_mcps = self._build_mealy_mcp_references()
        if not self.mealy_mcps:
            print(
                "\n⚠️  Warning: No MCP references configured for Mealy tools. "
                "Meal integration will fail without Mealy MCP.\n",
                file=sys.stderr,
            )
        else:
            print(
                f"\n🍽️  Mealy MCP references configured: {', '.join(self.mealy_mcps)}\n",
                file=sys.stderr,
            )

        # Create agents
        self.hexis_analysis_agent = create_hexis_analysis_agent(
            self.llm, mcps=self.hexis_mcps
        )
        self.weekly_structure_agent = create_weekly_structure_agent(self.llm)
        self.meal_generation_agent = create_meal_generation_agent(
            self.llm, mcps=self.mealy_mcps  # Mealy MCP for fetching preferences
        )
        self.nutritional_validation_agent = create_nutritional_validation_agent(
            self.llm
        )
        self.mealy_integration_agent = create_mealy_integration_agent(
            self.llm, mcps=self.mealy_mcps
        )

    def _build_hexis_mcp_references(self) -> List[str]:
        """Build MCP references for Hexis integration."""
        raw_urls = os.getenv("HEXIS_MCP_SERVER_URL") or os.getenv(
            "MCP_SERVER_URL", ""
        )
        tool_names_env = os.getenv("HEXIS_MCP_TOOL_NAMES", "")
        tool_names = (
            [name.strip() for name in tool_names_env.split(",") if name.strip()]
            if tool_names_env
            else []
        )

        if not tool_names:
            tool_names = load_catalog_tool_names(["hexis__"])

        return build_mcp_references(raw_urls, tool_names)

    def _build_mealy_mcp_references(self) -> List[str]:
        """Build MCP references for Mealy integration."""
        raw_urls = os.getenv("MEALY_MCP_SERVER_URL") or os.getenv(
            "MCP_SERVER_URL", ""
        )
        tool_names_env = os.getenv("MEALY_MCP_TOOL_NAMES", "")
        tool_names = (
            [name.strip() for name in tool_names_env.split(",") if name.strip()]
            if tool_names_env
            else []
        )

        if not tool_names:
            tool_names = load_catalog_tool_names(["mealy__"])

        return build_mcp_references(raw_urls, tool_names)

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
                parsed_raw = json.loads(raw_value)
                if isinstance(parsed_raw, dict):
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
        for candidate in self._collect_payload_candidates(crew_output):
            try:
                return model_type.model_validate(candidate)
            except ValidationError:
                continue
        return None

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

    def generate_meal_plan(self, week_start_date: str) -> Dict[str, Any]:
        """
        Generate a complete weekly meal plan.

        Args:
            week_start_date: Start date of the week (ISO format YYYY-MM-DD)

        Returns:
            Complete meal planning result with integration status
        """
        print(f"\n🚀 Starting Meal Planning for week of {week_start_date}\n", file=sys.stderr)

        # ===================================================================
        # STEP 1: Hexis Analysis - Analyze training data to determine needs
        # ===================================================================
        print("\n📊 Step 1: Analyzing Hexis training data...\n", file=sys.stderr)

        hexis_task = create_hexis_analysis_task(
            self.hexis_analysis_agent, week_start_date
        )
        hexis_crew = Crew(
            agents=[self.hexis_analysis_agent],
            tasks=[hexis_task],
            process=Process.sequential,
            verbose=True,
        )

        hexis_result = hexis_crew.kickoff()
        hexis_model = self._extract_model_from_output(hexis_result, HexisWeeklyAnalysis)

        if hexis_model is None:
            error_msg = "Hexis analysis failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "hexis_analysis"}

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
        )

        structure_result = structure_crew.kickoff()
        structure_model = self._extract_model_from_output(
            structure_result, WeeklyNutritionPlan
        )

        if structure_model is None:
            error_msg = "Weekly structure failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "weekly_structure"}

        nutrition_plan = structure_model.model_dump()
        print(
            f"\n✅ Nutrition plan created:\n{json.dumps(nutrition_plan, indent=2)}\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 3: Meal Generation - Generate actual meals
        # ===================================================================
        print("\n👨‍🍳 Step 3: Generating weekly meals...\n", file=sys.stderr)

        meal_task = create_meal_generation_task(
            self.meal_generation_agent, nutrition_plan
        )
        meal_crew = Crew(
            agents=[self.meal_generation_agent],
            tasks=[meal_task],
            process=Process.sequential,
            verbose=True,
        )

        meal_result = meal_crew.kickoff()
        meal_model = self._extract_model_from_output(meal_result, WeeklyMealPlan)

        if meal_model is None:
            error_msg = "Meal generation failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "meal_generation"}

        meal_plan = meal_model.model_dump()
        print(
            f"\n✅ Meal plan generated: {len(meal_plan.get('daily_plans', []))} days\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 4: Nutritional Validation - Validate the meal plan
        # ===================================================================
        print("\n🔍 Step 4: Validating nutritional quality...\n", file=sys.stderr)

        validation_task = create_nutritional_validation_task(
            self.nutritional_validation_agent, meal_plan, nutrition_plan
        )
        validation_crew = Crew(
            agents=[self.nutritional_validation_agent],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=True,
        )

        validation_result = validation_crew.kickoff()
        validation_model = self._extract_model_from_output(
            validation_result, NutritionalValidation
        )

        if validation_model is None:
            error_msg = "Validation failed to return valid JSON"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "validation"}

        validation = validation_model.model_dump()
        print(
            f"\n✅ Validation complete: Approved={validation.get('approved', False)}\n",
            file=sys.stderr,
        )

        # ===================================================================
        # STEP 5: Mealy Integration - Sync to Mealy
        # ===================================================================
        print("\n🔗 Step 5: Integrating with Mealy...\n", file=sys.stderr)

        integration_task = create_mealy_integration_task(
            self.mealy_integration_agent, meal_plan, validation
        )
        integration_crew = Crew(
            agents=[self.mealy_integration_agent],
            tasks=[integration_task],
            process=Process.sequential,
            verbose=True,
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
        final_result = {
            "week_start_date": week_start_date,
            "hexis_analysis": hexis_analysis,
            "nutrition_plan": nutrition_plan,
            "meal_plan": meal_plan,
            "validation": validation,
            "integration": integration,
            "summary": integration.get("summary", "Meal planning completed"),
        }

        return final_result
