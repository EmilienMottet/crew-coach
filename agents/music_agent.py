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
            "Use Spotify listening history to recover tracks played during the workout and "
            "deliver a concise list suitable for appending to the activity description"
        ),
        "backstory": (
            "You specialise in analysing Spotify playback history around the time of a Strava activity. "
            "Given a time window, you consult the available Spotify MCP tools (e.g., recently played "
            "endpoints) to identify the exact tracks the athlete listened to. You return a short ordered "
            "list (maximum five items) formatted as '<artist> – <title>' so it can be appended to the workout "
            "summary. If no music is detected you clearly state that no tracks were captured. You always keep "
            "the final wording under the provided character limit to preserve room in the Strava description."
        ),
        "verbose": True,
        "allow_delegation": False,
        "llm": llm,
        "tools": tools_list,
    }

    if mcps_list:
        agent_kwargs["mcps"] = mcps_list

    return Agent(**agent_kwargs)
