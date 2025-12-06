"""Main Crew definition for Strava activity description generation."""
from __future__ import annotations

# Ensure authentication patches and environment loading happen first
import llm_auth_init  # noqa: F401

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from crewai import Crew, LLM, Process
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput

from agents import (
    create_description_agent,
    create_music_agent,
    create_privacy_agent,
    create_translation_agent,
    create_lyrics_agent,
    # Supervisor/Executor/Reviewer pattern for DESCRIPTION
    create_description_supervisor_agent,
    create_data_retrieval_executor_agent,
    create_description_reviewer_agent,
)
from schemas import (
    ActivityMusicSelection,
    GeneratedActivityContent,
    PrivacyAssessment,
    TranslationPayload,
    LyricsVerificationResult,
    # Supervisor/Executor/Reviewer pattern schemas for DESCRIPTION
    ActivityDataRetrievalPlan,
    RawActivityData,
)
from tasks import (
    create_description_task,
    create_music_task,
    create_privacy_task,
    create_translation_task,
    create_lyrics_task,
    # Supervisor/Executor/Reviewer pattern tasks for DESCRIPTION
    create_description_supervisor_task,
    create_data_retrieval_executor_task,
    create_description_reviewer_task,
)
from mcp_auth_wrapper import MetaMCPAdapter
from llm_provider_rotation import create_llm_with_rotation
from mcp_tool_wrapper import wrap_mcp_tools
from observability import setup_structured_logger


class StravaDescriptionCrew:
    """Crew for generating and validating Strava activity descriptions."""

    def __init__(self):
        """Initialize the crew with LLM and agents."""
        load_dotenv()

        # Initialize structured logger
        self.logger = setup_structured_logger("crew.main")

        # Configure environment variables for LiteLLM/OpenAI
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

        # Configure authentication - use Bearer token API key
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it with your API key (e.g., 'cWCsrv7H-SKZl0Z9-2JOk7pmzsdO7yQ2abmmR1D0vBs')"
            )
        
        # Set environment variables that LiteLLM expects
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_API_KEY"] = api_key

        # Create per-agent LLMs with category-based defaults
        # All endpoints now support tools - no restrictions needed

        # COMPLEXE: Heavy MCP tool usage (Strava, Intervals.icu, Weather, Toolbox)
        # has_tools=True excludes thinking models that hallucinate tool calls
        self.description_llm = self._create_agent_llm(
            agent_name="DESCRIPTION",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=True,  # Agent uses MCP tools
        )

        # INTERMÉDIAIRE: Data analysis (Spotify from n8n)
        # No tools - analyzes data provided by n8n (thinking models OK)
        self.music_llm = self._create_agent_llm(
            agent_name="MUSIC",
            default_model=intermediate_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - No MCP tools
        )

        # SIMPLE: Pure reasoning agents (no tools, thinking models OK)
        self.privacy_llm = self._create_agent_llm(
            agent_name="PRIVACY",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - Pure reasoning
        )

        self.translation_llm = self._create_agent_llm(
            agent_name="TRANSLATION",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - Pure reasoning
        )

        # ═══════════════════════════════════════════════════════════════════════════════
        # MCP Configuration using MetaMCPAdapter (Working Approach)
        # ═══════════════════════════════════════════════════════════════════════════════
        #
        # Note: We tried using CrewAI's native DSL `mcps` field, but it has issues with
        # MetaMCP endpoints. The native DSL expects specific transport types and doesn't
        # work well with MetaMCP's custom protocol. We're reverting to the working
        # MetaMCPAdapter approach until CrewAI better supports MetaMCP.
        #
        # ═══════════════════════════════════════════════════════════════════════════════

        # Initialize MCP adapters
        self.mcp_adapters = []
        self.mcp_tools = []

        mcp_api_key = os.getenv("MCP_API_KEY", "")
        require_mcp = os.getenv("REQUIRE_MCP", "true").lower() == "true"

        # Define MCP server URLs for Strava Description Crew
        mcp_servers = {
            "Strava": os.getenv("STRAVA_MCP_SERVER_URL", ""),
            "Intervals.icu": os.getenv("INTERVALS_MCP_SERVER_URL", ""),
            # Note: Music data now comes from n8n (spotify_recently_played field)
            "Meteo": os.getenv("METEO_MCP_SERVER_URL", ""),
            "Toolbox": os.getenv("TOOLBOX_MCP_SERVER_URL", ""),
            "Music": os.getenv("MUSIC_MCP_SERVER_URL", "https://mcp.emottet.com/metamcp/Music"),
        }

        # Filter out empty URLs
        active_servers = {name: url for name, url in mcp_servers.items() if url}

        if active_servers and mcp_api_key:
            print(f"\n🔗 Connecting to {len(active_servers)} MCP servers via MetaMCPAdapter...\n", file=sys.stderr)

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
                "The Description Agent requires MCP tools to fetch workout data.\n"
                "Please set MCP server URLs and MCP_API_KEY environment variables.\n"
                "Required: STRAVA_MCP_SERVER_URL, INTERVALS_MCP_SERVER_URL\n"
                "Optional: METEO_MCP_SERVER_URL, TOOLBOX_MCP_SERVER_URL\n"
                "Note: Music data is now provided by n8n (no MCP server needed)\n"
                "To disable this check, set REQUIRE_MCP=false (not recommended).\n"
            )
            print(error_msg, file=sys.stderr)
            raise ValueError("MCP configuration is required but not provided")
        else:
            print("\n⚠️  Warning: No MCP configuration. Agents will operate without live data.\n", file=sys.stderr)

        # Filter tools by type for different agents
        # Note: Spotify tools are no longer needed - music data comes from n8n
        strava_tools = [t for t in self.mcp_tools if "strava" in t.name.lower()]
        intervals_tools = [t for t in self.mcp_tools if "intervals" in t.name.lower()]
        weather_tools = [t for t in self.mcp_tools if "weather" in t.name.lower() or "openweathermap" in t.name.lower()]
        toolbox_tools = [t for t in self.mcp_tools if any(keyword in t.name.lower() for keyword in ["fetch", "time", "task"])]

        if strava_tools:
            print(f"🏃 Found {len(strava_tools)} Strava tools\n", file=sys.stderr)
        if intervals_tools:
            print(f"📊 Found {len(intervals_tools)} Intervals.icu tools\n", file=sys.stderr)
        if weather_tools:
            print(f"🌤️  Found {len(weather_tools)} Weather tools\n", file=sys.stderr)
        if toolbox_tools:
            print(f"🛠️  Found {len(toolbox_tools)} Toolbox tools\n", file=sys.stderr)
        
        music_tools = [t for t in self.mcp_tools if "lyrics" in t.name.lower()]
        if music_tools:
            print(f"🎵 Found {len(music_tools)} Music tools\n", file=sys.stderr)
        
        print(f"🎵 Music: Using n8n Spotify data (no MCP tools)\n", file=sys.stderr)

        # Create agents with MCP tools (using tools parameter, not mcps)
        # Description Agent: needs Strava, Intervals.icu, Weather, and Toolbox
        description_tools = strava_tools + intervals_tools + weather_tools + toolbox_tools
        self.description_agent = create_description_agent(
            self.description_llm,
            tools=description_tools if description_tools else None
        )

        # Music Agent: analyzes Spotify data from n8n (no MCP tools needed)
        print(f"🎵 Creating Music Agent (using n8n Spotify data, no MCP tools)\n", file=sys.stderr)
        self.music_agent = create_music_agent(self.music_llm)
        print(f"   Data source: n8n (spotify_recently_played field)\n", file=sys.stderr)

        # Privacy Agent: no MCP tools needed (pure reasoning)
        self.privacy_agent = create_privacy_agent(self.privacy_llm)

        # Translation Agent: no MCP tools needed
        # Translation Agent: no MCP tools needed
        self.translation_agent = create_translation_agent(self.translation_llm)

        # Lyrics Agent: needs Music MCP tools
        # has_tools=True excludes thinking models that hallucinate tool calls
        self.lyrics_llm = self._create_agent_llm(
            agent_name="LYRICS",
            default_model=intermediate_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=True,  # Agent uses MCP tools
        )
        self.lyrics_agent = create_lyrics_agent(
            self.lyrics_llm,
            tools=music_tools if music_tools else None
        )

        # ═══════════════════════════════════════════════════════════════════════════════
        # Supervisor/Executor/Reviewer Pattern for DESCRIPTION
        # ═══════════════════════════════════════════════════════════════════════════════
        # Separates reasoning from tool execution to avoid thinking model tool hallucination
        #
        # Flow: Supervisor → Executor → Reviewer
        #   1. Supervisor: Plans data retrieval (NO tools → thinking models OK)
        #   2. Executor: Executes tool calls (HAS tools → thinking models EXCLUDED)
        #   3. Reviewer: Creates description from raw data (NO tools → thinking models OK)
        # ═══════════════════════════════════════════════════════════════════════════════

        # DESCRIPTION_SUPERVISOR: Pure reasoning, plans tool calls (complex)
        self.description_supervisor_llm = self._create_agent_llm(
            agent_name="DESCRIPTION_SUPERVISOR",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - Pure reasoning
        )

        # DATA_RETRIEVAL_EXECUTOR: Executes planned tool calls (simple)
        # has_tools=True excludes thinking models that hallucinate tool calls
        self.data_retrieval_executor_llm = self._create_agent_llm(
            agent_name="DATA_RETRIEVAL_EXECUTOR",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key,
            has_tools=True,  # Agent uses MCP tools
        )

        # DESCRIPTION_REVIEWER: Creates description from raw data (intermediate)
        self.description_reviewer_llm = self._create_agent_llm(
            agent_name="DESCRIPTION_REVIEWER",
            default_model=intermediate_model_name,
            default_base=base_url,
            default_key=api_key,
            # has_tools=False (default) - Pure reasoning
        )

        # Create Supervisor/Executor/Reviewer agents for DESCRIPTION
        self.description_supervisor_agent = create_description_supervisor_agent(
            self.description_supervisor_llm
        )

        # Executor needs Strava + Intervals.icu + Weather + Toolbox tools
        self.data_retrieval_executor_agent = create_data_retrieval_executor_agent(
            self.data_retrieval_executor_llm,
            tools=description_tools if description_tools else None
        )

        self.description_reviewer_agent = create_description_reviewer_agent(
            self.description_reviewer_llm
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
            agent_name: Name prefix for env vars (e.g., "DESCRIPTION", "MUSIC")
            default_model: Fallback model name (or category like "complex")
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
        endpoint_type = "ccproxy"
        if "/copilot/v1" in (agent_base or "").lower():
            endpoint_type = "copilot"
        elif "/codex/v1" in (agent_base or "").lower():
            endpoint_type = "codex"
        elif "/claude/v1" in (agent_base or "").lower():
            endpoint_type = "claude"

        print(
            f"🤖 {agent_name} Agent:\n"
            f"   Model: {agent_model}\n"
            f"   Endpoint: {endpoint_type} ({agent_base})\n",
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

    def _collect_payload_candidates(self, crew_output: CrewOutput) -> List[Dict[str, Any]]:
        """Collect payload candidates across crew and individual task outputs."""
        candidates: List[Dict[str, Any]] = []

        if crew_output.json_dict:
            candidates.append(crew_output.json_dict)

        if crew_output.pydantic:
            candidates.append(crew_output.pydantic.model_dump())

        for task_output in reversed(crew_output.tasks_output or []):
            candidates.extend(self._payloads_from_task_output(task_output))

        return candidates

    @staticmethod
    def _is_pydantic_schema(data: Dict[str, Any]) -> bool:
        """Detect if JSON looks like a Pydantic schema definition instead of data."""
        schema_indicators = ["properties", "required", "type", "additionalProperties", "$schema"]
        return any(key in data for key in schema_indicators)

    @staticmethod
    def _extract_data_from_schema(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to extract actual data from a Pydantic schema response.

        Example problematic input:
        {
            "description": "🚴 Sortie puissance midi...",  # ACTUAL DATA (full description)
            "properties": {  # SCHEMA METADATA
                "title": {"description": "English activity title", ...},
                "description": {"description": "English activity description", ...}
            },
            "required": ["title", "description"],
            "title": "TranslationPayload"
        }

        The LLM mistakenly put the translated DESCRIPTION in the top-level,
        but forgot to include the title. We need to detect this and reject it.
        """
        # Schema metadata fields that should NOT be in data responses
        schema_metadata_fields = {"properties", "required", "additionalProperties", "$schema", "definitions"}

        # If we find schema metadata, this is NOT a valid data response
        if any(field in data for field in schema_metadata_fields):
            # Check if there's a "properties" field - this is a strong indicator of a schema
            if "properties" in data:
                # Invalid response - LLM returned the schema instead of data
                return None

            # Other schema fields also indicate this is not data
            return None

        # If no schema metadata, this is actual data
        return data

    def _extract_model_from_output(
        self,
        crew_output: CrewOutput,
        model_type: Type[BaseModel],
    ) -> Optional[BaseModel]:
        """Safely parse the final task output into a typed model."""
        # First try structured candidates (json_dict, pydantic)
        for candidate in self._collect_payload_candidates(crew_output):
            # Check if this looks like a Pydantic schema instead of data
            if self._is_pydantic_schema(candidate):
                print("\n⚠️  Detected Pydantic schema instead of data, attempting to extract...\n", file=sys.stderr)
                extracted_data = self._extract_data_from_schema(candidate)
                if extracted_data:
                    candidate = extracted_data
                else:
                    continue

            try:
                return model_type.model_validate(candidate)
            except ValidationError:
                continue

        # If no structured data found, try to extract JSON from raw text
        raw_text = self._stringify_output(crew_output)
        if raw_text:
            extracted_json = self._extract_json_from_text(raw_text)
            if extracted_json:
                # Check for Pydantic schema in extracted JSON too
                if self._is_pydantic_schema(extracted_json):
                    extracted_data = self._extract_data_from_schema(extracted_json)
                    if extracted_data:
                        extracted_json = extracted_data

                try:
                    return model_type.model_validate(extracted_json)
                except ValidationError:
                    pass

        return None
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text that may contain thoughts or explanations.
        
        Looks for JSON objects in the text, trying multiple strategies:
        1. Look for JSON in code fences (```json ... ```)
        2. Look for standalone JSON objects { ... }
        3. Parse the entire text as JSON
        4. Check if JSON values contain nested JSON strings and parse them
        """
        import re
        
        # Strategy 1: JSON in markdown code fences
        code_fence_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(code_fence_pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    # Check for nested JSON strings in values
                    return StravaDescriptionCrew._unnest_json_strings(parsed)
            except json.JSONDecodeError:
                continue
        
        # Strategy 2: Find JSON objects in text (look for { ... })
        # Find all potential JSON objects
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        
        # Try each match, preferring longer ones (more complete)
        for match in sorted(matches, key=len, reverse=True):
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    # Check for nested JSON strings in values
                    return StravaDescriptionCrew._unnest_json_strings(parsed)
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Try parsing entire text
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Check for nested JSON strings in values
                return StravaDescriptionCrew._unnest_json_strings(parsed)
        except json.JSONDecodeError:
            pass
        
        return None
    
    @staticmethod
    def _unnest_json_strings(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively parse JSON strings found in dictionary values.

        Sometimes LLMs return JSON where some values are themselves JSON strings.
        This function detects and parses those nested JSON strings.

        Example:
            Input:  {"description": "{\"title\": \"Test\", \"value\": 123}"}
            Output: {"description": {"title": "Test", "value": 123}}
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        # Recursively unnest
                        result[key] = StravaDescriptionCrew._unnest_json_strings(parsed)
                    else:
                        result[key] = parsed
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, sanitize LLM reasoning from string
                    result[key] = StravaDescriptionCrew._sanitize_llm_reasoning(value)
            elif isinstance(value, dict):
                # Recursively process nested dicts
                result[key] = StravaDescriptionCrew._unnest_json_strings(value)
            elif isinstance(value, list):
                # Process list items
                result[key] = [
                    StravaDescriptionCrew._unnest_json_strings(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    @staticmethod
    def _sanitize_llm_reasoning(text: str) -> str:
        """Remove LLM reasoning/thought prefixes from string values.

        LLMs sometimes include their reasoning process in the output values.
        This strips common patterns like "Thought: I now have..." from the beginning.

        Example:
            Input:  "\\n\\nThought: I now have all the info.\\n\\n{actual content}"
            Output: "{actual content}"
        """
        import re

        # Strip leading whitespace first
        cleaned = text.strip()

        # Remove common LLM reasoning patterns at the start
        # Pattern: "Thought: ... actual content" or "I now have... {content}"
        reasoning_patterns = [
            r'^Thought:\s*.*?(?=\n\n|\{|$)',  # "Thought: ..." until double newline or JSON
            r'^I now have all.*?(?=\n\n|\{|$)',  # "I now have all..." pattern
            r'^Let me.*?(?=\n\n|\{|$)',  # "Let me provide..." pattern
            r'^Final Answer:\s*',  # "Final Answer:" prefix
        ]

        for pattern in reasoning_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Strip again after removal
        cleaned = cleaned.strip()

        # If we stripped everything, return original (minus whitespace)
        if not cleaned:
            return text.strip()

        return cleaned

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

    def _default_generated_content(self, raw_summary: str) -> Dict[str, Any]:
        """Return a safe fallback when generation fails with basic activity info."""
        # Access current activity_data for better fallback
        activity_data = getattr(self, 'current_activity_data', {})
        object_data = activity_data.get("object_data", {})

        activity_type = object_data.get("type", "Unknown")
        distance_km = object_data.get("distance", 0) / 1000
        moving_time_min = object_data.get("moving_time", 0) / 60

        # Generate appropriate title and type based on activity data
        if activity_type.lower() == "ride":
            title = f"🚴 Ride {distance_km:.1f}K"
            workout_type = "Bike Ride"
        elif activity_type.lower() == "run":
            title = f"🏃 Run {distance_km:.1f}K"
            workout_type = "Run"
        else:
            title = f"🏃 {activity_type.title()} {distance_km:.1f}K"
            workout_type = activity_type.title()

        # Generate basic description with available metrics
        description_parts = []
        if distance_km > 0:
            description_parts.append(f"Distance: {distance_km:.1f} km")
        if moving_time_min > 0:
            description_parts.append(f"Duration: {moving_time_min:.0f} minutes")

        description = " | ".join(description_parts) + "\n\nGenerated by AI"

        # Extract any useful info from raw_summary but limit it
        if raw_summary and len(raw_summary.strip()) > 10:
            # Take first meaningful sentence if available
            first_sentence = raw_summary.split('.')[0].split('\n')[0].strip()
            if len(first_sentence) > 20 and len(first_sentence) < 200:
                # Combine with basic metrics
                description = first_sentence + "\n\nGenerated by AI"

        # Create basic key metrics
        key_metrics = {}
        if distance_km > 0:
            key_metrics["distance"] = f"{distance_km:.1f} km"
        if moving_time_min > 0:
            key_metrics["duration"] = f"{moving_time_min:.0f} min"

        return {
            "title": title,
            "description": description,
            "workout_type": workout_type,
            "key_metrics": key_metrics,
        }

    @staticmethod
    def _normalize_activity_data(activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize activity_data to handle double-nested object_data from n8n.

        Sometimes n8n wraps the payload twice, resulting in:
        {
            "object_data": {
                "object_type": "activity",
                "object_data": { ...actual data... },
                "spotify_recently_played": { ... }
            }
        }

        This method flattens it to:
        {
            "object_data": { ...actual data... },
            "spotify_recently_played": { ... }
        }
        """
        object_data = activity_data.get("object_data", {})

        if isinstance(object_data, dict) and "object_data" in object_data:
            # Double-nested structure detected
            inner_object_data = object_data.get("object_data", {})
            spotify_data = object_data.get("spotify_recently_played", {})

            # Flatten the structure
            normalized = {
                "object_type": activity_data.get("object_type", object_data.get("object_type", "activity")),
                "object_id": activity_data.get("object_id") or object_data.get("object_id"),
                "object_data": inner_object_data,
            }

            # Preserve spotify_recently_played at top level
            if spotify_data:
                normalized["spotify_recently_played"] = spotify_data
            elif "spotify_recently_played" in activity_data:
                normalized["spotify_recently_played"] = activity_data["spotify_recently_played"]

            return normalized

        return activity_data

    @staticmethod
    def _safe_activity_id(activity_data: Dict[str, Any]) -> Optional[int]:
        """Extract activity id from the webhook payload."""
        # First normalize the data structure
        normalized = StravaDescriptionCrew._normalize_activity_data(activity_data)
        object_data = normalized.get("object_data")
        if isinstance(object_data, dict):
            activity_id = object_data.get("id")
            if isinstance(activity_id, int):
                return activity_id
        return None

    @staticmethod
    def _privacy_failure_fallback(
        error_message: str,
        generated_content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback privacy assessment when the agent fails."""
        return {
            "privacy_approved": False,
            "during_work_hours": True,
            "should_be_private": True,
            "issues_found": [f"Privacy agent execution failed: {error_message}"],
            "recommended_changes": {
                "title": generated_content.get("title", "Activity"),
                "description": generated_content.get("description", ""),
            },
            "reasoning": f"Privacy agent failed: {error_message}",
        }

    @staticmethod
    def _resolve_final_content(
        generated_content: Dict[str, Any],
        privacy_assessment: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Merge generator output with privacy adjustments to derive final text."""
        recommended = privacy_assessment.get("recommended_changes", {})
        if not isinstance(recommended, dict):
            recommended = {}

        fallback_title = generated_content.get("title", "Activity")
        fallback_description = generated_content.get("description", "")

        final_title = recommended.get("title") or fallback_title
        final_description = recommended.get("description")

        if final_description is None:
            final_description = fallback_description

        return final_title, final_description
    
    def process_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Strava activity webhook to generate localized content."""

        # Normalize the data structure to handle double-nested object_data from n8n
        activity_data = self._normalize_activity_data(activity_data)

        # Store activity data for fallback generation
        self.current_activity_data = activity_data

        # ═══════════════════════════════════════════════════════════════════════════════
        # Step 1: Generate Activity Description (Supervisor/Executor/Reviewer Pattern)
        # ═══════════════════════════════════════════════════════════════════════════════
        # This pattern separates reasoning from tool execution to prevent thinking model
        # tool hallucination. The flow is:
        #   1a. Supervisor: Plans which tools to call (pure reasoning, no tools)
        #   1b. Executor: Executes the planned tool calls (has tools, simple model)
        #   1c. Reviewer: Creates description from raw data (pure reasoning, no tools)
        # ═══════════════════════════════════════════════════════════════════════════════

        print("\n🚀 Step 1a: Planning data retrieval strategy...\n", file=sys.stderr)

        supervisor_task = create_description_supervisor_task(
            self.description_supervisor_agent,
            activity_data,
        )
        supervisor_crew = Crew(
            agents=[self.description_supervisor_agent],
            tasks=[supervisor_task],
            process=Process.sequential,
            verbose=True,
        )

        supervisor_result = supervisor_crew.kickoff()
        retrieval_plan = self._extract_model_from_output(
            supervisor_result, ActivityDataRetrievalPlan
        )

        if retrieval_plan is None:
            raw_supervisor = self._stringify_output(supervisor_result)
            print(
                f"\n⚠️  Supervisor returned invalid plan:\n"
                f"   Raw output (first 500 chars): {raw_supervisor[:500] if raw_supervisor else '(empty)'}\n",
                file=sys.stderr,
            )
            # Fallback: create a minimal retrieval plan
            object_data = activity_data.get("object_data", {})
            retrieval_plan = ActivityDataRetrievalPlan(
                activity_id=str(object_data.get("id", "unknown")),
                activity_date=object_data.get("start_date_local", "unknown"),
                activity_type=object_data.get("type", "Run"),
                tool_calls=[],  # Empty - Executor will use defaults
                data_focus=["basic_metrics"],
                description_style="engaging",
            )
            print("   Using fallback retrieval plan.\n", file=sys.stderr)
        else:
            print(
                f"\n✅ Retrieval plan created:\n"
                f"   Activity: {retrieval_plan.activity_id} ({retrieval_plan.activity_type})\n"
                f"   Tool calls planned: {len(retrieval_plan.tool_calls)}\n"
                f"   Data focus: {retrieval_plan.data_focus}\n",
                file=sys.stderr,
            )

        print("\n🔧 Step 1b: Executing data retrieval...\n", file=sys.stderr)

        executor_task = create_data_retrieval_executor_task(
            self.data_retrieval_executor_agent,
            activity_data,
            retrieval_plan.model_dump(),
        )
        executor_crew = Crew(
            agents=[self.data_retrieval_executor_agent],
            tasks=[executor_task],
            process=Process.sequential,
            verbose=True,
        )

        executor_result = executor_crew.kickoff()
        raw_activity_data = self._extract_model_from_output(
            executor_result, RawActivityData
        )

        if raw_activity_data is None:
            raw_executor = self._stringify_output(executor_result)
            print(
                f"\n⚠️  Executor returned invalid data:\n"
                f"   Raw output (first 500 chars): {raw_executor[:500] if raw_executor else '(empty)'}\n",
                file=sys.stderr,
            )
            # Fallback: create minimal raw data from activity_data
            object_data = activity_data.get("object_data", {})
            raw_activity_data = RawActivityData(
                activity_id=str(object_data.get("id", "unknown")),
                activity_date=object_data.get("start_date_local", "unknown"),
                activity_type=object_data.get("type", "Run"),
                tool_results=[],
                total_calls=0,
                successful_calls=0,
                activity_details=object_data,  # Use original Strava data
            )
            print("   Using fallback raw data from Strava.\n", file=sys.stderr)
        else:
            print(
                f"\n✅ Data retrieval complete:\n"
                f"   Activity: {raw_activity_data.activity_id}\n"
                f"   Tool calls: {raw_activity_data.successful_calls}/{raw_activity_data.total_calls} successful\n",
                file=sys.stderr,
            )

        print("\n✍️  Step 1c: Creating activity description...\n", file=sys.stderr)

        reviewer_task = create_description_reviewer_task(
            self.description_reviewer_agent,
            activity_data,
            raw_activity_data.model_dump(),
        )
        reviewer_crew = Crew(
            agents=[self.description_reviewer_agent],
            tasks=[reviewer_task],
            process=Process.sequential,
            verbose=True,
        )

        reviewer_result = reviewer_crew.kickoff()
        generated_model = self._extract_model_from_output(
            reviewer_result, GeneratedActivityContent
        )

        if generated_model is None:
            raw_summary = self._stringify_output(reviewer_result)
            print(
                f"\n📦 Raw reviewer result:\n"
                f"   Raw output (first 500 chars): {raw_summary[:500] if raw_summary else '(empty)'}\n"
                f"   JSON dict: {reviewer_result.json_dict}\n"
                f"   Pydantic: {reviewer_result.pydantic}\n",
                file=sys.stderr,
            )
            if not raw_summary:
                print(
                    "   ❌ Reviewer returned EMPTY output!\n",
                    file=sys.stderr,
                )
            generated_content = self._default_generated_content(raw_summary)
            print(
                "\n⚠️  Warning: Reviewer returned invalid JSON, using fallback.\n",
                file=sys.stderr,
            )
        else:
            generated_content = generated_model.model_dump()

        print(
            f"\n✅ Generated content:\n{json.dumps(generated_content, indent=2)}\n",
            file=sys.stderr,
        )

        print("\n🎧 Step 2: Capturing soundtrack details...\n", file=sys.stderr)

        # Debug: Show activity timing information
        object_data = activity_data.get("object_data", {})
        start_date_local = object_data.get("start_date_local", "N/A")
        moving_time = object_data.get("moving_time", "N/A")
        # Intervals.icu fields: 'feel' and 'icu_rpe'
        icu_feel = object_data.get("feel")
        icu_rpe = object_data.get("icu_rpe")
        
        print(f"📅 Activity timing:", file=sys.stderr)
        print(f"   Start: {start_date_local}", file=sys.stderr)
        print(f"   Duration: {moving_time}s", file=sys.stderr)
        if icu_feel is not None:
            print(f"   Feel (ICU): {icu_feel}", file=sys.stderr)
        if icu_rpe is not None:
            print(f"   RPE (ICU): {icu_rpe}/10", file=sys.stderr)

        # Check if n8n provided Spotify data
        spotify_data = activity_data.get("spotify_recently_played", {})
        spotify_items_count = len(spotify_data.get("items", [])) if isinstance(spotify_data, dict) else 0
        print(f"   Spotify data from n8n: {spotify_items_count} track(s)", file=sys.stderr)
        print("", file=sys.stderr)

        music_tracks: List[str] = []

        try:
            music_task = create_music_task(
                self.music_agent,
                activity_data,
                generated_content,
            )

            print("📋 Music task created with following context:", file=sys.stderr)
            print(f"   Activity ID: {object_data.get('id', 'N/A')}", file=sys.stderr)
            print(f"   Distance: {object_data.get('distance', 0) / 1000:.2f} km", file=sys.stderr)
            print("", file=sys.stderr)

            # Reset tool call counter before executing music crew
            music_crew = Crew(
                agents=[self.music_agent],
                tasks=[music_task],
                process=Process.sequential,
                verbose=True,
            )

            print("🔄 Executing music crew...\n", file=sys.stderr)
            music_result = music_crew.kickoff()

            # Debug: Show raw music result
            print("\n📦 Raw music crew result:", file=sys.stderr)
            print(f"   Type: {type(music_result)}", file=sys.stderr)
            if hasattr(music_result, 'raw'):
                raw_output = music_result.raw
                print(f"   Raw output (first 500 chars): {str(raw_output)[:500]}", file=sys.stderr)
            if hasattr(music_result, 'json_dict'):
                print(f"   JSON dict: {music_result.json_dict}", file=sys.stderr)
            if hasattr(music_result, 'pydantic'):
                print(f"   Pydantic: {music_result.pydantic}", file=sys.stderr)
            print("", file=sys.stderr)

            # ENHANCED LOGGING: Log music crew execution completion
            self.logger.info(
                "Music crew execution completed",
                extra={"extra_fields": {
                    "result_type": type(music_result).__name__,
                    "has_raw": hasattr(music_result, 'raw'),
                    "has_json_dict": hasattr(music_result, 'json_dict'),
                    "has_pydantic": hasattr(music_result, 'pydantic'),
                }}
            )

            # ENHANCED LOGGING: Log raw output for debugging
            if hasattr(music_result, 'raw'):
                raw_output = music_result.raw
                self.logger.debug(
                    "Music raw output",
                    extra={"extra_fields": {
                        "raw_output": str(raw_output)[:1000],  # First 1000 chars
                        "raw_output_length": len(str(raw_output)),
                    }}
                )

            # ENHANCED LOGGING: Log JSON extraction strategies
            json_extraction_attempts = []

            # Strategy 1: Direct json_dict
            if hasattr(music_result, 'json_dict') and music_result.json_dict:
                json_extraction_attempts.append({
                    "strategy": "direct_json_dict",
                    "success": True,
                    "data_preview": str(music_result.json_dict)[:500],
                })

            # Strategy 2: Pydantic model
            if hasattr(music_result, 'pydantic') and music_result.pydantic:
                json_extraction_attempts.append({
                    "strategy": "pydantic_model",
                    "success": True,
                    "model_type": type(music_result.pydantic).__name__,
                })

            # Strategy 3: _extract_json_from_text (from raw)
            if hasattr(music_result, 'raw'):
                try:
                    extracted_json = self._extract_json_from_text(str(music_result.raw))
                    json_extraction_attempts.append({
                        "strategy": "extract_json_from_text",
                        "success": extracted_json is not None,
                        "data_preview": str(extracted_json)[:500] if extracted_json else None,
                    })
                except Exception as e:
                    json_extraction_attempts.append({
                        "strategy": "extract_json_from_text",
                        "success": False,
                        "error": str(e),
                    })

            self.logger.info(
                "JSON extraction strategies attempted",
                extra={"extra_fields": {
                    "attempts": json_extraction_attempts,
                    "total_strategies": len(json_extraction_attempts),
                    "successful_strategies": sum(1 for a in json_extraction_attempts if a.get("success", False)),
                }}
            )

            # ENHANCED LOGGING: Log Pydantic validation with detailed error handling
            music_model = None
            try:
                music_model = self._extract_model_from_output(
                    music_result, ActivityMusicSelection
                )
            except ValidationError as e:
                self.logger.error(
                    "Pydantic validation failed for music result",
                    extra={"extra_fields": {
                        "validation_errors": [
                            {
                                "loc": str(err["loc"]),
                                "msg": err["msg"],
                                "type": err["type"],
                            }
                            for err in e.errors()
                        ],
                        "error_count": len(e.errors()),
                    }},
                    exc_info=True,
                )
                print(f"❌ Pydantic validation failed: {e}\n", file=sys.stderr)

            print(f"🔍 Music model extraction result: {music_model is not None}\n", file=sys.stderr)

            # Log music model result
            if music_model is not None:
                music_payload_preview = music_model.model_dump()
                tracks_preview = music_payload_preview.get("music_tracks", [])

                # Log the music tracks found
                if tracks_preview:
                    print(
                        f"\n🎵 Music agent returned {len(tracks_preview)} track(s)\n",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "\n⚠️  Music agent returned no tracks.\n"
                        "   This suggests no music was played during the activity.\n",
                        file=sys.stderr,
                    )

            if music_model is None:
                print(
                    "\n⚠️  Warning: Music agent returned invalid JSON, keeping original description.\n",
                    file=sys.stderr,
                )
            else:
                music_payload = music_model.model_dump()
                print("📄 Parsed music payload:", file=sys.stderr)
                print(f"   Keys: {list(music_payload.keys())}", file=sys.stderr)
                
                # Extract candidate tracks
                candidate_tracks = music_payload.get("candidate_tracks", [])
                if isinstance(candidate_tracks, list):
                    candidate_tracks = [
                        track for track in candidate_tracks if isinstance(track, str) and track
                    ]

                print(f"🎵 Candidate tracks: {candidate_tracks}", file=sys.stderr)
                print(f"   Count: {len(candidate_tracks)}\n", file=sys.stderr)

                if candidate_tracks:
                    # ------------------------------------------------------------------
                    # Step 3: Lyrics Verification and Quote Selection
                    # ------------------------------------------------------------------
                    print("\n🎤 Step 3: Verifying lyrics and selecting quote...\n", file=sys.stderr)
                    
                    current_description = generated_content.get("description", "")
                    
                    lyrics_task = create_lyrics_task(
                        self.lyrics_agent,
                        candidate_tracks,
                        current_description,
                        activity_data
                    )
                    
                    lyrics_crew = Crew(
                        agents=[self.lyrics_agent],
                        tasks=[lyrics_task],
                        process=Process.sequential,
                        verbose=True,
                    )
                    
                    lyrics_result = lyrics_crew.kickoff()
                    
                    lyrics_model = self._extract_model_from_output(
                        lyrics_result, LyricsVerificationResult
                    )

                    if lyrics_model:
                        print("\n✅ Lyrics verification complete:", file=sys.stderr)
                        print(f"   Accepted: {lyrics_model.accepted_tracks}", file=sys.stderr)
                        print(f"   Rejected: {lyrics_model.rejected_tracks}", file=sys.stderr)
                        print(f"   Quote: \"{lyrics_model.selected_quote}\" ({lyrics_model.quote_source})", file=sys.stderr)

                        # ENHANCED LOGGING: Log lyrics verification results
                        self.logger.info(
                            "Lyrics verification completed",
                            extra={"extra_fields": {
                                "accepted_tracks_count": len(lyrics_model.accepted_tracks),
                                "rejected_tracks_count": len(lyrics_model.rejected_tracks),
                                "accepted_tracks": lyrics_model.accepted_tracks,
                                "rejected_tracks": lyrics_model.rejected_tracks,
                                "quote_selected": lyrics_model.selected_quote,
                                "quote_source": lyrics_model.quote_source,
                            }}
                        )

                        # ENHANCED LOGGING: Log rejection reasoning for each rejected track
                        if lyrics_model.rejected_tracks:
                            for track in lyrics_model.rejected_tracks:
                                self.logger.info(
                                    f"Track rejected: {track}",
                                    extra={"extra_fields": {
                                        "track_name": track,
                                        "rejection_reason": "political_content",  # Infer from agent backstory
                                    }}
                                )

                        # Update description with the final version from Lyrics Agent
                        generated_content["description"] = lyrics_model.final_description
                        
                        # Update key metrics with music info if available
                        if lyrics_model.accepted_tracks:
                            metrics = generated_content.get("key_metrics", {})
                            metrics["music"] = f"{len(lyrics_model.accepted_tracks)} tracks"
                            generated_content["key_metrics"] = metrics
                    else:
                        print("\n⚠️  Warning: Lyrics agent returned invalid JSON, using original description.\n", file=sys.stderr)
                        # Fallback: append music tracks manually if lyrics agent failed but we have candidates
                        if candidate_tracks:
                            music_section = "\n\n🎧 Music: " + "; ".join(candidate_tracks)
                            generated_content["description"] += music_section
                else:
                    print(
                        "\n⚠️  No candidate tracks found. Skipping lyrics verification and quote selection.\n",
                        file=sys.stderr,
                    )

        except Exception as exc:  # noqa: BLE001
            print(
                f"\n⚠️  Warning: Music agent failed ({exc}), keeping original description.\n",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc(file=sys.stderr)

        print("\n🔒 Step 3: Checking privacy and compliance...\n", file=sys.stderr)

        privacy_task = create_privacy_task(
            self.privacy_agent,
            activity_data,
            generated_content,
        )

        privacy_crew = Crew(
            agents=[self.privacy_agent],
            tasks=[privacy_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            privacy_result = privacy_crew.kickoff()
            privacy_model = self._extract_model_from_output(
                privacy_result, PrivacyAssessment
            )
            if privacy_model is None:
                print(
                    "\n⚠️  Warning: Privacy agent response invalid, defaulting to safe settings.\n",
                    file=sys.stderr,
                )
                raw_privacy = self._stringify_output(privacy_result)
                print(f"Raw privacy output: {raw_privacy[:200]}...", file=sys.stderr)
                privacy_check = self._privacy_failure_fallback(
                    "Invalid JSON format", generated_content
                )
            else:
                privacy_check = privacy_model.model_dump()

        except Exception as exc:  # noqa: BLE001
            print(f"\n❌ Privacy crew execution failed: {exc}\n", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            privacy_check = self._privacy_failure_fallback(str(exc), generated_content)

        if not isinstance(privacy_check, dict):
            print(
                "\n⚠️  Warning: Privacy assessment was not a dict, using fallback.\n",
                file=sys.stderr,
            )
            privacy_check = self._privacy_failure_fallback(
                "Unexpected privacy response type", generated_content
            )

        print(
            f"\n✅ Privacy check result:\n{json.dumps(privacy_check, indent=2)}\n",
            file=sys.stderr,
        )

        final_title, final_description = self._resolve_final_content(
            generated_content, privacy_check
        )

        print(
            f"\n🧾 Post-privacy content:\nTitle: {final_title}\nDescription: "
            f"{final_description[:120]}{'…' if len(final_description) > 120 else ''}\n",
            file=sys.stderr,
        )

        print("\n🌐 Step 4: Translating content (if enabled)...\n", file=sys.stderr)

        translation_enabled = os.getenv("TRANSLATION_ENABLED", "false").lower() == "true"

        if translation_enabled:
            translation_payload = {
                "title": final_title,
                "description": final_description,
                "workout_type": generated_content.get("workout_type", "Unknown"),
                "key_metrics": generated_content.get("key_metrics", {}),
            }

            translation_task = create_translation_task(
                self.translation_agent,
                json.dumps(translation_payload, indent=2),
            )

            translation_crew = Crew(
                agents=[self.translation_agent],
                tasks=[translation_task],
                process=Process.sequential,
                verbose=True,
            )

            try:
                translation_result = translation_crew.kickoff()
                translation_model = self._extract_model_from_output(
                    translation_result, TranslationPayload
                )

                if translation_model is None:
                    print(
                        "\n⚠️  Warning: Translation output invalid, keeping English content.\n",
                        file=sys.stderr,
                    )
                else:
                    translated_content = translation_model.model_dump()
                    final_title = translated_content.get("title", final_title)
                    final_description = translated_content.get(
                        "description", final_description
                    )
                    print(
                        f"\n✅ Translation successful:\nTitle: {final_title}\nDescription: "
                        f"{final_description[:120]}{'…' if len(final_description) > 120 else ''}\n",
                        file=sys.stderr,
                    )

            except Exception as exc:  # noqa: BLE001
                print(
                    f"\n⚠️  Warning: Translation failed ({exc}), keeping English content.\n",
                    file=sys.stderr,
                )
                import traceback

                traceback.print_exc(file=sys.stderr)
        else:
            print("\n⏭️  Translation disabled, skipping.\n", file=sys.stderr)

        activity_id = self._safe_activity_id(activity_data)

        issues = privacy_check.get("issues_found", [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []

        final_result = {
            "activity_id": activity_id,
            "title": final_title,
            "description": final_description,
            "should_be_private": bool(privacy_check.get("should_be_private", True)),
            "privacy_check": {
                "approved": bool(privacy_check.get("privacy_approved", False)),
                "during_work_hours": bool(
                    privacy_check.get("during_work_hours", False)
                ),
                "issues": issues,
                "reasoning": privacy_check.get(
                    "reasoning", "Privacy agent returned no explanation"
                ),
            },
            "workout_analysis": {
                "type": generated_content.get("workout_type", "Unknown"),
                "metrics": generated_content.get("key_metrics", {}),
            },
        }

        if music_tracks:
            final_result["music_tracks"] = music_tracks

        return final_result



class RedirectStdoutToStderr:
    """Context manager to redirect stdout to stderr."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout


def main():
    """Main entry point for n8n integration."""
    # Read input from stdin (n8n will provide this)
    input_data = sys.stdin.read()

    crew = None
    try:
        activity_data = json.loads(input_data)

        # Handle array input (from n8n)
        if isinstance(activity_data, list) and len(activity_data) > 0:
            activity_data = activity_data[0]

        # Process the activity
        # Redirect stdout to stderr during processing to keep stdout clean for final JSON
        with RedirectStdoutToStderr():
            crew = StravaDescriptionCrew()
            result = crew.process_activity(activity_data)
        
        # Output result as JSON to stdout
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        # Log error to stderr with full traceback
        print(f"\n❌ Error: {str(e)}\n", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        # Try to extract activity_id even on error
        try:
            activity_id = crew._safe_activity_id(activity_data) if crew else None
        except:
            activity_id = None

        # Output error result to stdout
        error_result = {
            "error": str(e),
            "activity_id": activity_id,
            "title": "Error processing activity",
            "description": "An error occurred while generating the description",
            "should_be_private": True,
            "privacy_check": {
                "approved": False,
                "during_work_hours": False,
                "issues": [str(e)],
                "reasoning": "Error during processing"
            }
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)
    finally:
        # Cleanup MCP adapters
        if crew and crew.mcp_adapters:
            for adapter in crew.mcp_adapters:
                try:
                    adapter.stop()
                except Exception:
                    pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
