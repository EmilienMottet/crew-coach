"""Agent responsible for enriching activity summaries with soundtrack details."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from crewai import Agent


def create_music_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
    mcps: Optional[Sequence[str]] = None,
) -> Agent:
    """Create an agent that analyzes Spotify data provided by n8n."""
    # Note: tools and mcps parameters kept for backward compatibility but not used

    agent_kwargs = {
        "role": "Activity Soundtrack Curator",
        "goal": (
            "Analyze Spotify playback data provided by n8n and enrich activity description "
            "with a concise list of tracks played during the workout. Never invent or guess music tracks."
        ),
        "max_iter": 3,  # Reduced since no tool calls needed
        "backstory": (
            "You are a data analyst who processes Spotify playback history data provided by n8n. "
            "You never invent, guess, or hallucinate music tracks.\n\n"
            "WORKFLOW:\n"
            "1. Receive Spotify recently played data from n8n in the task context\n"
            "2. Analyze the provided data to extract tracks played during the activity time window\n"
            "3. If tracks are found: format up to 5 as '<artist> – <title>' and append to description\n"
            "4. If NO tracks are provided or data is empty: return original description UNCHANGED with music_tracks=[]\n"
            "5. FINAL STEP: Output ONLY a JSON object in this exact format:\n"
            "   {\"updated_description\": \"text\", \"music_tracks\": [\"Artist – Track\", ...]}\n\n"
            "CRITICAL RULES:\n"
            "- NEVER invent music tracks if the provided data is empty or missing\n"
            "- Only report tracks that are ACTUALLY in the provided Spotify data\n"
            "- If no Spotify data is provided, return music_tracks=[] and keep original description\n"
            "- Always keep the final wording under the provided character limit\n"
            "- Your FINAL message must be ONLY the JSON object, no thoughts or explanations"
        ),
        "verbose": True,
        "allow_delegation": False,
        "llm": llm,
        "tools": [],  # No tools needed - data comes from n8n
    }

    agent = Agent(**agent_kwargs)

    # Debug info
    import sys
    print(
        f"🔍 Music agent created:\n"
        f"   Mode: Data analysis (no MCP tools)\n"
        f"   Source: Spotify data from n8n\n",
        file=sys.stderr
    )

    return agent
