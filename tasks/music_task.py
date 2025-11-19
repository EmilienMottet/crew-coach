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
    """Create a task that analyzes Spotify data from n8n and appends tracks to the description."""
    object_data = activity_data.get("object_data", {})
    spotify_data = activity_data.get("spotify_recently_played", {})

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

    # Detect model type to customize prompt
    model_name = getattr(agent.llm, 'model', '').lower() if hasattr(agent, 'llm') and hasattr(agent.llm, 'model') else ''
    api_base = getattr(agent.llm, 'api_base', '').lower() if hasattr(agent, 'llm') and hasattr(agent.llm, 'api_base') else ''

    # Extract Spotify track count for logging
    spotify_items = spotify_data.get("items", []) if isinstance(spotify_data, dict) else []
    print(f"🎵 Spotify data from n8n:", file=sys.stderr)
    print(f"   Tracks available: {len(spotify_items)}", file=sys.stderr)

    snapshot = {
        "activity_id": activity_id,
        "distance_km": round(distance_km, 2),
        "start_utc": start_iso,
        "end_utc": end_iso,
        "moving_time_seconds": moving_time,
        "title": existing_title,
        "description_before_music": existing_description,
        "spotify_tracks_count": len(spotify_items),
    }

    print(f"📸 Music task snapshot:", file=sys.stderr)
    print(f"   {json.dumps(snapshot, indent=6)}\n", file=sys.stderr)

    description = f"""
    Analyze Spotify playback data provided by n8n and enrich the activity description with music tracks.

    CONTEXT SNAPSHOT (JSON):
    {json.dumps(snapshot, indent=2)}

    SPOTIFY DATA FROM N8N:
    {json.dumps(spotify_data, indent=2)}

    DATA PROCESSING REQUIREMENTS:

    ⚠️  CRITICAL: You are receiving Spotify data directly from n8n (see SPOTIFY DATA FROM N8N above).
    DO NOT invent or guess music tracks. Only report tracks that are ACTUALLY in the provided data.

    REQUIRED STEPS:
    1. ANALYZE the provided Spotify data (spotify_recently_played) above:
       - Check if the "items" array contains any tracks
       - Extract tracks that were played during the activity window (start_utc to end_utc)
       - Filter by "played_at" timestamps to ensure tracks overlap with the activity period
       - Select up to five distinct tracks ordered by playback time

    2. FORMAT the tracks:
       - Each track should be formatted as: "<artist_name> – <track_name>"
       - Extract artist_name from: item.track.artists[0].name
       - Extract track_name from: item.track.name

    3. If the Spotify data is EMPTY or contains NO tracks:
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
    After analyzing the provided Spotify data, you MUST output a JSON object with this exact structure:
    {{
        "updated_description": "description text with optional music section",
        "music_tracks": ["Artist – Track", ...]
    }}

    Do NOT include any explanatory text, thoughts, or reasoning in your final output.
    Your final message must be ONLY the JSON object, nothing else.
    """

    # Model-specific prompt adaptations for GLM-4.6
    if 'glm-4.6' in model_name or 'z.ai' in api_base:
        # GLM-4.6: Add extremely strict JSON output requirements
        description += """

    🚨 CRITICAL FOR GLM-4.6 MODEL:

    YOUR FINAL RESPONSE MUST BE ONLY THIS JSON FORMAT:
    {"updated_description": "...text with optional music section...", "music_tracks": ["Artist – Track", ...]}

    ABSOLUTELY NO:
    - Explanatory text
    - "Here is the JSON:"
    - Markdown code blocks
    - Thinking or analysis
    - Any text before or after the JSON

    ONLY OUTPUT THE RAW JSON OBJECT.
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
        # Note: No tools needed - music data comes from n8n (spotify_recently_played)
        tools=None,
    )
