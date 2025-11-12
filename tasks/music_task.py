"""Task for enriching the activity summary with soundtrack information."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict

from crewai import Task

from schemas import ActivityMusicSelection


def create_music_task(
    agent,
    activity_data: Dict[str, Any],
    generated_content: Dict[str, Any],
) -> Task:
    """Create a task that fetches Spotify tracks and appends them to the description."""
    object_data = activity_data.get("object_data", {})

    activity_id = object_data.get("id")
    start_date_local = object_data.get("start_date_local", "")
    moving_time = int(object_data.get("moving_time", 0) or 0)

    try:
        start_dt = datetime.fromisoformat(start_date_local.replace("Z", "+00:00"))
        end_dt = start_dt + timedelta(seconds=moving_time)
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()
        print(f"⏰ Music task timestamps calculated:", file=sys.stderr)
        print(f"   Start: {start_iso}", file=sys.stderr)
        print(f"   End: {end_iso}", file=sys.stderr)
        print(f"   Duration: {moving_time}s\n", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        start_iso = start_date_local
        end_iso = ""
        print(f"⚠️  Warning: Failed to parse timestamps: {e}", file=sys.stderr)
        print(f"   Using raw start: {start_iso}\n", file=sys.stderr)

    existing_title = generated_content.get("title", "")
    existing_description = generated_content.get("description", "")

    distance_km = object_data.get("distance", 0) / 1000 if object_data else 0

    snapshot = {
        "activity_id": activity_id,
        "distance_km": round(distance_km, 2),
        "start_utc": start_iso,
        "end_utc": end_iso,
        "moving_time_seconds": moving_time,
        "title": existing_title,
        "description_before_music": existing_description,
    }

    print(f"📸 Music task snapshot:", file=sys.stderr)
    print(f"   {json.dumps(snapshot, indent=6)}\n", file=sys.stderr)

    description = f"""
    Retrieve the music listened to during the Strava activity and append it to the summary.

    CONTEXT SNAPSHOT (JSON):
    {json.dumps(snapshot, indent=2)}

    DATA REQUIREMENTS - MANDATORY TOOL USAGE:

    ⚠️  CRITICAL: You MUST use the Spotify MCP tools to retrieve REAL playback data.
    DO NOT invent or guess music tracks. Only report tracks that are returned by the Spotify API.

    REQUIRED STEPS:
    1. FIRST: Call the `spotify__getRecentlyPlayed` tool with appropriate time parameters.
       Example call:
       - Tool: spotify__getRecentlyPlayed
       - Parameters: {{"limit": 50}} or with time filtering if supported

    2. THEN: Analyze the ACTUAL response from Spotify API:
       - Extract tracks that were played during the activity window (start_utc to end_utc)
       - Filter by timestamps to ensure tracks overlap with the activity period
       - Select up to five distinct tracks ordered by playback time
       - Format each entry as "<artist> – <track>"

    3. If the Spotify API returns NO tracks or an empty response:
       - DO NOT add any music section to the description
       - DO NOT invent placeholder tracks
       - Return the original description unchanged with music_tracks: []

    OUTPUT CONTRACT:
    - If tracks are found:
      * Start from the existing English description and append a short "Music" section at the end.
      * Example pattern: "... main description text.\n\n🎧 Music: Artist – Track; ..."
      * Keep the entire updated description ≤ 500 characters. Trim the track list if necessary.
    - If NO tracks are found:
      * Return the original description UNCHANGED (do not add any music section or message).
      * Set music_tracks to an empty list [].
    - Return valid JSON following the ActivityMusicSelection schema.
    - Do not wrap the JSON in markdown fences.
    - Ensure the appended section is suitable for later French translation and keeps emoji intact.

    CRITICAL FINAL ACTION:
    After retrieving and analyzing the Spotify data, you MUST output a JSON object with this exact structure:
    {{
        "updated_description": "description text with optional music section",
        "music_tracks": ["Artist – Track", ...]
    }}
    
    Do NOT include any explanatory text, thoughts, or reasoning in your final output.
    Your final message must be ONLY the JSON object, nothing else.
    """

    return Task(
        description=description,
        agent=agent,
        expected_output=(
            "A JSON object (not wrapped in markdown) with exactly two fields:\n"
            "1. 'updated_description': the activity description with optional music section appended\n"
            "2. 'music_tracks': array of track strings in format 'Artist – Track'\n"
            "Example: {\"updated_description\": \"...\", \"music_tracks\": [\"Artist – Track\"]}\n"
            "IMPORTANT: Your FINAL response must be ONLY this JSON object, no explanatory text."
        ),
        # CRITICAL: output_json disables tool calling! Must parse JSON manually instead.
        # output_json=ActivityMusicSelection,
        tools=agent.tools if hasattr(agent, 'tools') and agent.tools else None,
    )
