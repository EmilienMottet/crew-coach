"""Agent responsible for enriching activity summaries with soundtrack details."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from crewai import Agent


def create_music_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
    mcps: Optional[Sequence[str]] = None,
) -> Agent:
    """Create an agent that retrieves music played during the activity via Spotify MCP."""
    tools_list = list(tools) if tools else []
    mcps_list = list(mcps) if mcps else []

    agent_kwargs = {
        "role": "Activity Soundtrack Curator",
        "goal": (
            "MANDATORY: Call Spotify MCP tools to retrieve REAL playback data and deliver a concise list "
            "of ACTUAL tracks played during the workout. Never invent or guess music tracks."
        ),
        "backstory": (
            "You are a data retrieval specialist who MUST use the available Spotify MCP tools to fetch "
            "real playback history. You never invent, guess, or hallucinate music tracks.\n\n"
            "WORKFLOW:\n"
            "1. ALWAYS call the spotify__getRecentlyPlayed tool FIRST to get actual playback data\n"
            "2. Analyze the API response to extract tracks played during the activity time window\n"
            "3. If the API returns tracks: format up to 5 as '<artist> – <title>' and append to description\n"
            "4. If the API returns NO tracks or empty data: return original description UNCHANGED with music_tracks=[]\n"
            "5. FINAL STEP: Output ONLY a JSON object in this exact format:\n"
            "   {\"updated_description\": \"text\", \"music_tracks\": [\"Artist – Track\", ...]}\n\n"
            "CRITICAL RULES:\n"
            "- You MUST call spotify__getRecentlyPlayed before returning any result\n"
            "- NEVER invent music tracks if the API returns no data\n"
            "- Only report tracks that are ACTUALLY returned by the Spotify API\n"
            "- If uncertain or no API data available, return music_tracks=[] and keep original description\n"
            "- Always keep the final wording under the provided character limit\n"
            "- Your FINAL message must be ONLY the JSON object, no thoughts or explanations"
        ),
        "verbose": True,
        "allow_delegation": False,
        "llm": llm,
        "tools": tools_list,
    }

    if mcps_list:
        agent_kwargs["mcps"] = mcps_list

    agent = Agent(**agent_kwargs)

    # Debug: Verify tools were properly set
    import sys
    actual_tools = getattr(agent, 'tools', None)
    print(
        f"🔍 Music agent created:\n"
        f"   tools parameter: {len(tools_list)}\n"
        f"   agent.tools: {len(actual_tools) if actual_tools else 'None'}\n",
        file=sys.stderr
    )

    return agent
