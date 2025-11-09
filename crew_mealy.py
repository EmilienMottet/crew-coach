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
from mcp_auth_wrapper import MetaMCPAdapter
from llm_provider_rotation import create_llm_with_rotation

# Load environment variables from .env file (override shell variables)
load_dotenv(override=True)


class MealPlanningCrew:
    """Crew for generating and validating weekly meal plans."""

    def __init__(self):
        """Initialize the crew with LLM and agents."""
        # Environment already loaded at module level

        # Configure environment variables for LiteLLM/OpenAI
        base_url = os.getenv("OPENAI_API_BASE", "https://ccproxy.emottet.com/copilot/v1")
        default_complex_model = os.getenv("OPENAI_MODEL_NAME") or "claude-sonnet-4-5"
        complex_model_name = os.getenv(
            "OPENAI_COMPLEX_MODEL_NAME",
            default_complex_model,
        )
        simple_model_name = os.getenv(
            "OPENAI_SIMPLE_MODEL_NAME",
            "claude-haiku-4.5",
        )

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

        # Nutritional Validation Agent: needs food data for validation
        validation_tools = food_data_tools + hexis_tools
        self.nutritional_validation_agent = create_nutritional_validation_agent(
            self.nutritional_validation_llm, tools=validation_tools if validation_tools else None
        )

        # Mealy Integration Agent: needs Hexis tools for meal creation
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

        agent_model = os.getenv(model_key, default_model)
        agent_base = os.getenv(base_key, default_base)

        # Log configuration for transparency
        if os.getenv(model_key) or os.getenv(base_key):
            print(
                f"🔧 {agent_name} Agent: Using custom config "
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
        # STEP 3: Meal Generation with Validation Loop
        # Keep regenerating meals until the validator approves
        # ===================================================================
        print("\n👨‍🍳 Step 3: Generating weekly meals with validation loop...\n", file=sys.stderr)
        
        max_attempts = 3
        meal_plan = None
        validation = None
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n🔄 Attempt {attempt}/{max_attempts}: Generating meals...\n", file=sys.stderr)
            
            # Generate meals
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
                print(f"\n⚠️  Attempt {attempt}: Meal generation returned invalid JSON\n", file=sys.stderr)
                continue

            meal_plan = meal_model.model_dump()
            print(
                f"\n✅ Attempt {attempt}: Meal plan generated ({len(meal_plan.get('daily_plans', []))} days)\n",
                file=sys.stderr,
            )

            # Validate the generated meals
            print(f"\n🔍 Validating meal plan (attempt {attempt})...\n", file=sys.stderr)
            
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
                print(f"\n⚠️  Attempt {attempt}: Validation returned invalid JSON\n", file=sys.stderr)
                continue

            validation = validation_model.model_dump()
            is_approved = validation.get('approved', False)
            
            if is_approved:
                print(
                    f"\n✅ SUCCESS (Attempt {attempt}): Meal plan APPROVED by validator!\n",
                    file=sys.stderr,
                )
                break
            else:
                issues = validation.get('issues', [])
                print(
                    f"\n❌ Attempt {attempt}: Meal plan REJECTED ({len(issues)} issues found)\n",
                    file=sys.stderr,
                )
                print(f"   Issues summary:\n", file=sys.stderr)
                for i, issue in enumerate(issues[:3], 1):  # Show first 3 issues
                    print(f"   {i}. {issue[:100]}...\n", file=sys.stderr)
                
                if attempt < max_attempts:
                    print(f"\n🔄 Regenerating meals with validator feedback...\n", file=sys.stderr)
                    # Update nutrition_plan with feedback for next iteration
                    nutrition_plan['validation_feedback'] = {
                        'attempt': attempt,
                        'issues': issues,
                        'recommendations': validation.get('recommendations', [])
                    }
                else:
                    print(
                        f"\n⚠️  Maximum attempts ({max_attempts}) reached. Using last generated plan.\n",
                        file=sys.stderr,
                    )
        
        # Check if we have valid results
        if meal_plan is None or validation is None:
            error_msg = "Meal generation and validation failed after all attempts"
            print(f"\n❌ {error_msg}\n", file=sys.stderr)
            return {"error": error_msg, "step": "meal_generation_validation_loop"}
        
        print(
            f"\n✅ Final validation status: Approved={validation.get('approved', False)}\n",
            file=sys.stderr,
        )

        
        # ===================================================================
        # STEP 4: Mealy Integration - Sync to Mealy (only if approved)
        # ===================================================================
        print("\n🔗 Step 4: Integrating with Mealy...\n", file=sys.stderr)
        
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
