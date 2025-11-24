"""Agent responsible for verifying lyrics and selecting quotes."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from crewai import Agent


def create_lyrics_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
) -> Agent:
    """Create an agent that verifies lyrics and selects quotes."""

    agent_kwargs = {
        "role": "Lyrics Verification and Quote Selector",
        "goal": (
            "Verify song lyrics for political/controversial content and select an inspiring quote."
        ),
        "backstory": (
            "You are a content moderator and curator for a sports activity feed. "
            "Your job is to ensure that music tracks listed in activity descriptions are appropriate "
            "and to enhance the description with a relevant quote.\n\n"
            "You have access to a lyrics database. For each song candidate:\n"
            "1. Check the lyrics for political content (especially anti-capitalist or anti-Michelin).\n"
            "2. Reject songs that are too political or controversial.\n"
            "3. Accept songs that are engaged but respectful (e.g., against violence).\n\n"
            "After filtering the songs, you must select a quote to add to the description.\n"
            "The quote should be:\n"
            "- Thematic with the sport or session.\n"
            "- Or inspiring/motivational.\n"
            "- Or related to Michelin (the user's employer) in a positive or neutral way.\n"
            "- Or a line from one of the accepted songs if it fits the above criteria.\n\n"
            "You will output the final description including the approved music tracks and the selected quote."
        ),
        "verbose": True,
        "allow_delegation": False,
        "llm": llm,
        "tools": tools or [],
    }

    agent = Agent(**agent_kwargs)

    # Debug info
    import sys
    print(
        f"🔍 Lyrics agent created:\n"
        f"   Role: Lyrics Verification and Quote Selector\n"
        f"   Tools: {[t.name for t in tools] if tools else 'None'}\n",
        file=sys.stderr
    )

    return agent
