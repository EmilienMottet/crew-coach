"""Task for enriching the activity summary with soundtrack information."""
from __future__ import annotations

import json
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
    except Exception:  # noqa: BLE001
        start_iso = start_date_local
        end_iso = ""

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

    description = f"""
    Retrieve the music listened to during the Strava activity and append it to the summary.

    CONTEXT SNAPSHOT (JSON):
    {json.dumps(snapshot, indent=2)}

    DATA REQUIREMENTS:
    1. Use the configured Spotify MCP tools to query the playback history between the start and end timestamps.
       - If only a start time is available, inspect a ±5 minute window around the start.
       - Prefer precise tools such as `Spotify__get_recently_played` or equivalent.
    2. Extract up to five distinct tracks ordered by playback time. Format each entry as "<artist> – <track>".
    3. If no tracks are found, explicitly state that no music was captured instead of guessing.

    OUTPUT CONTRACT:
    - Start from the existing English description and append a short "Music" section at the end.
      Example pattern:
        "... main description text.\n\n🎧 Music: Artist – Track; ..."
    - Keep the entire updated description ≤ 500 characters. Trim the track list if necessary.
    - Return valid JSON following the ActivityMusicSelection schema.
    - Do not wrap the JSON in markdown fences.
    - Ensure the appended section is suitable for later French translation and keeps emoji intact.
    """

    return Task(
        description=description,
        agent=agent,
        expected_output=(
            "Valid JSON adhering to the ActivityMusicSelection schema with the "
            "description updated to include a concise music list"
        ),
        output_json=ActivityMusicSelection,
    )
