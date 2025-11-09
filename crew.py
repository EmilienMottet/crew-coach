"""Main Crew definition for Strava activity description generation."""
from __future__ import annotations

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
)
from schemas import (
    ActivityMusicSelection,
    GeneratedActivityContent,
    PrivacyAssessment,
    TranslationPayload,
)
from tasks import (
    create_description_task,
    create_music_task,
    create_privacy_task,
    create_translation_task,
)
from mcp_utils import build_mcp_references, load_catalog_tool_names
from mcp_auth_wrapper import MetaMCPAdapter
from llm_provider_rotation import create_llm_with_rotation

# Load environment variables from .env file (override shell variables)
load_dotenv(override=True)


class StravaDescriptionCrew:
    """Crew for generating and validating Strava activity descriptions."""

    def __init__(self):
        """Initialize the crew with LLM and agents."""
        load_dotenv()

        # Configure environment variables for LiteLLM/OpenAI
        base_url = os.getenv("OPENAI_API_BASE", "https://ccproxy.emottet.com/copilot/v1")
        default_complex_model = os.getenv("OPENAI_MODEL_NAME") or "claude-sonnet-4-5"
        complex_model_name = os.getenv(
            "OPENAI_COMPLEX_MODEL_NAME",
            default_complex_model,
        )
        simple_model_name = os.getenv(
            "OPENAI_SIMPLE_MODEL_NAME",
            "claude-haiku-4-5",
        )

        # Configure authentication - support both API key and Basic Auth
        auth_token = os.getenv("OPENAI_API_AUTH_TOKEN")  # Base64-encoded Basic Auth

        # Set environment variables that LiteLLM expects
        os.environ["OPENAI_API_BASE"] = base_url

        # Configure litellm/OpenAI authentication
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
        self.description_llm = self._create_agent_llm(
            agent_name="DESCRIPTION",
            default_model=complex_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.music_llm = self._create_agent_llm(
            agent_name="MUSIC",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.privacy_llm = self._create_agent_llm(
            agent_name="PRIVACY",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key
        )

        self.translation_llm = self._create_agent_llm(
            agent_name="TRANSLATION",
            default_model=simple_model_name,
            default_base=base_url,
            default_key=api_key
        )

        # Initialize MCP adapters with MetaMCP authentication fix
        self.mcp_adapters = []
        self.mcp_tools = []

        mcp_api_key = os.getenv("MCP_API_KEY", "")
        require_mcp = os.getenv("REQUIRE_MCP", "true").lower() == "true"

        # Define MCP server URLs for Strava Description Crew
        mcp_servers = {
            "Strava": os.getenv("STRAVA_MCP_SERVER_URL", ""),
            "Intervals.icu": os.getenv("INTERVALS_MCP_SERVER_URL", ""),
            "Music": os.getenv("MUSIC_MCP_SERVER_URL", ""),
            "Meteo": os.getenv("METEO_MCP_SERVER_URL", ""),
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
                "The Description Agent requires MCP tools to fetch workout data.\n"
                "Please set MCP server URLs and MCP_API_KEY environment variables.\n"
                "Required: STRAVA_MCP_SERVER_URL, INTERVALS_MCP_SERVER_URL, MUSIC_MCP_SERVER_URL\n"
                "To disable this check, set REQUIRE_MCP=false (not recommended).\n"
            )
            print(error_msg, file=sys.stderr)
            raise ValueError("MCP configuration is required but not provided")
        else:
            print("\n⚠️  Warning: No MCP configuration. Agents will operate without live data.\n", file=sys.stderr)

        # Filter tools by type for different agents
        strava_tools = [t for t in self.mcp_tools if "strava" in t.name.lower()]
        intervals_tools = [t for t in self.mcp_tools if "intervals" in t.name.lower()]
        spotify_tools = [t for t in self.mcp_tools if "spotify" in t.name.lower()]
        weather_tools = [t for t in self.mcp_tools if "weather" in t.name.lower() or "openweathermap" in t.name.lower()]
        toolbox_tools = [t for t in self.mcp_tools if any(keyword in t.name.lower() for keyword in ["fetch", "time", "task"])]

        if strava_tools:
            print(f"🏃 Found {len(strava_tools)} Strava tools\n", file=sys.stderr)
        if intervals_tools:
            print(f"📊 Found {len(intervals_tools)} Intervals.icu tools\n", file=sys.stderr)
        if spotify_tools:
            print(f"🎵 Found {len(spotify_tools)} Spotify tools\n", file=sys.stderr)
        if weather_tools:
            print(f"🌤️  Found {len(weather_tools)} Weather tools\n", file=sys.stderr)
        if toolbox_tools:
            print(f"🛠️  Found {len(toolbox_tools)} Toolbox tools\n", file=sys.stderr)

        # Create agents with MCP tools
        # Description Agent: needs Strava, Intervals.icu, Weather, and Toolbox
        description_tools = strava_tools + intervals_tools + weather_tools + toolbox_tools
        self.description_agent = create_description_agent(
            self.description_llm,
            tools=description_tools if description_tools else None
        )

        # Music Agent: needs Spotify tools
        self.music_agent = create_music_agent(
            self.music_llm,
            tools=spotify_tools if spotify_tools else None
        )

        # Privacy Agent: no MCP tools needed (pure reasoning)
        self.privacy_agent = create_privacy_agent(self.privacy_llm)

        # Translation Agent: no MCP tools needed
        self.translation_agent = create_translation_agent(self.translation_llm)

    def _load_intervals_tool_names(self) -> List[str]:
        """Load MCP tool names for Intervals.icu integration."""
        env_value = os.getenv("INTERVALS_MCP_TOOL_NAMES", "")
        if env_value:
            tool_names = [name.strip() for name in env_value.split(",") if name.strip()]
            if tool_names:
                return tool_names
        # Fallback to local catalogue so we can still address specific tools if discovery fails.
        return load_catalog_tool_names(["IntervalsIcu__"])

    def _build_intervals_mcp_references(self, tool_names: List[str]) -> List[str]:
        """Build MCP references (DSL syntax) for the description agent."""
        raw_urls = os.getenv("MCP_SERVER_URL", "")
        api_key = os.getenv("MCP_API_KEY", "")
        return build_mcp_references(raw_urls, tool_names, api_key)

    def _load_spotify_tool_names(self) -> List[str]:
        """Load MCP tool names for Spotify integration."""
        env_value = os.getenv("SPOTIFY_MCP_TOOL_NAMES", "")
        if env_value:
            tool_names = [name.strip() for name in env_value.split(",") if name.strip()]
            if tool_names:
                return tool_names
        return load_catalog_tool_names(["spotify__"])

    def _build_spotify_mcp_references(self, tool_names: List[str]) -> List[str]:
        """Build MCP references for Spotify playback history tools."""
        raw_urls = os.getenv("SPOTIFY_MCP_SERVER_URL") or os.getenv(
            "MCP_SERVER_URL", ""
        )
        api_key = os.getenv("MCP_API_KEY", "")
        return build_mcp_references(raw_urls, tool_names, api_key)

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
            agent_name: Name prefix for env vars (e.g., "DESCRIPTION", "MUSIC")
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

    @staticmethod
    def _default_generated_content(raw_summary: str) -> Dict[str, Any]:
        """Return a safe fallback when generation fails."""
        truncated = raw_summary[:500] if raw_summary else ""
        return {
            "title": "Activity completed",
            "description": truncated,
            "workout_type": "Unknown",
            "key_metrics": {},
        }

    @staticmethod
    def _safe_activity_id(activity_data: Dict[str, Any]) -> Optional[int]:
        """Extract activity id from the webhook payload."""
        object_data = activity_data.get("object_data")
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

        print("\n🚀 Step 1: Generating activity description...\n", file=sys.stderr)

        description_task = create_description_task(self.description_agent, activity_data)
        description_crew = Crew(
            agents=[self.description_agent],
            tasks=[description_task],
            process=Process.sequential,
            verbose=True,
        )

        description_result = description_crew.kickoff()
        generated_model = self._extract_model_from_output(
            description_result, GeneratedActivityContent
        )

        if generated_model is None:
            raw_summary = self._stringify_output(description_result)
            generated_content = self._default_generated_content(raw_summary)
            print(
                "\n⚠️  Warning: Description agent returned invalid JSON, using fallback."\
                "\n",
                file=sys.stderr,
            )
        else:
            generated_content = generated_model.model_dump()

        print(
            f"\n✅ Generated content:\n{json.dumps(generated_content, indent=2)}\n",
            file=sys.stderr,
        )

        print("\n🎧 Step 2: Capturing soundtrack details...\n", file=sys.stderr)

        music_tracks: List[str] = []

        try:
            music_task = create_music_task(
                self.music_agent,
                activity_data,
                generated_content,
            )

            music_crew = Crew(
                agents=[self.music_agent],
                tasks=[music_task],
                process=Process.sequential,
                verbose=True,
            )

            music_result = music_crew.kickoff()
            music_model = self._extract_model_from_output(
                music_result, ActivityMusicSelection
            )

            if music_model is None:
                print(
                    "\n⚠️  Warning: Music agent returned invalid JSON, keeping original description.\n",
                    file=sys.stderr,
                )
            else:
                music_payload = music_model.model_dump()
                updated_description = music_payload.get(
                    "updated_description",
                    generated_content.get("description", ""),
                )

                if not isinstance(updated_description, str):
                    updated_description = generated_content.get("description", "")

                if len(updated_description) > 500:
                    print(
                        "\n⚠️  Warning: Music description exceeds 500 characters, trimming to limit.\n",
                        file=sys.stderr,
                    )
                    updated_description = updated_description[:500]

                generated_content["description"] = updated_description

                payload_tracks = music_payload.get("music_tracks", [])
                if isinstance(payload_tracks, list):
                    music_tracks = [
                        track for track in payload_tracks if isinstance(track, str) and track
                    ]

                if music_tracks:
                    metrics = generated_content.get("key_metrics")
                    if not isinstance(metrics, dict):
                        metrics = {}
                    metrics["playlist_tracks"] = "; ".join(music_tracks)
                    generated_content["key_metrics"] = metrics

                print(
                    f"\n✅ Music enrichment complete: {len(music_tracks)} track(s) captured.\n",
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
        crew = StravaDescriptionCrew()
        result = crew.process_activity(activity_data)
        
        # Output result as JSON to stdout
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        # Log error to stderr with full traceback
        print(f"\n❌ Error: {str(e)}\n", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # Output error result to stdout
        error_result = {
            "error": str(e),
            "activity_id": None,
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
