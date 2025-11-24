"""Task for verifying lyrics and selecting quotes."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from crewai import Task

def create_lyrics_task(
    agent,
    candidate_tracks: List[str],
    original_description: str,
    activity_data: Dict[str, Any],
) -> Task:
    """Create a task that verifies lyrics and updates the description."""
    
    object_data = activity_data.get("object_data", {})
    sport_type = object_data.get("type", "Activity")
    
    description = f"""
    Verify the lyrics of the following candidate tracks and update the activity description.

    CANDIDATE TRACKS:
    {json.dumps(candidate_tracks, indent=2)}

    ORIGINAL DESCRIPTION:
    {original_description}

    SPORT TYPE: {sport_type}

    REQUIREMENTS:
    1. For each candidate track:
       - Fetch the lyrics using the `lyrics__get_lyrics` tool (or similar).
       - Analyze the lyrics for political content (anti-capitalist, anti-Michelin).
       - REJECT if too political/controversial (e.g. "Sales Majestés - Macron").
       - ACCEPT if engaged but acceptable (e.g. "Sales Majestés - Tous les jours").
       - ACCEPT if neutral/safe.
    
    2. Select a Quote:
       - Choose a quote that is thematic with {sport_type}, inspiring, motivating, or related to Michelin.
       - You can use a line from one of the ACCEPTED songs if it fits.
       - Decide where to place the quote:
         - At the BEGINNING if it sets a good tone.
         - At the END if it works better as a sign-off.

    3. Format the Final Description:
       - Start with the selected quote (if placed at beginning).
       - Include the original description.
       - Append the "Music" section with ONLY the accepted tracks.
       - Append the selected quote (if placed at end).
       - Ensure the final text is in French (except for the quote if it's a song lyric in another language, but prefer French context).
       - Keep the total length reasonable (approx same as original + music + quote).

    OUTPUT FORMAT:
    Return a JSON object with the following structure:
    {{
        "final_description": "The complete updated description text",
        "accepted_tracks": ["Artist - Title", ...],
        "rejected_tracks": ["Artist - Title", ...],
        "selected_quote": "The quote text",
        "quote_source": "Source of the quote (e.g. Song Title, Author)"
    }}
    """

    return Task(
        description=description,
        agent=agent,
        expected_output=(
            "A JSON object with 'final_description', 'accepted_tracks', 'rejected_tracks', "
            "'selected_quote', and 'quote_source'."
        ),
    )
