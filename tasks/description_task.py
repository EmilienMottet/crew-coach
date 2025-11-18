"""Task for generating activity descriptions."""
from __future__ import annotations

import json
from typing import Any, Dict

from crewai import Task

from schemas import GeneratedActivityContent


def create_description_task(agent, activity_data: Dict[str, Any]) -> Task:
    """
    Create a task for generating activity title and description.
    
    Args:
        agent: The agent responsible for this task
        activity_data: Raw activity data from Strava webhook
        
    Returns:
        Configured Task instance
    """
    object_data = activity_data.get("object_data", {})
    strava_id = object_data.get("id")
    activity_type = object_data.get("type", "Run")
    distance = object_data.get("distance", 0) / 1000  # Convert to km
    moving_time = object_data.get("moving_time", 0)
    moving_time_min = moving_time / 60  # Convert to minutes
    start_date = object_data.get("start_date_local")
    
    # Calculate average pace (min/km)
    avg_pace_min_km = (moving_time / 60) / distance if distance > 0 else 0
    pace_min = int(avg_pace_min_km)
    pace_sec = int((avg_pace_min_km - pace_min) * 60)

    # Extract date for Intervals.icu lookup
    start_date_str = start_date.split('T')[0] if start_date and 'T' in start_date else start_date

    description = f"""
    Analyze the Strava activity below and craft an engaging English summary.

    ACTIVITY SNAPSHOT:
    - Strava Activity ID: {strava_id}
    - Sport Type: {activity_type}
    - Distance: {distance:.2f} km
    - Moving Duration: {moving_time_min:.0f} minutes
    - Average Pace: {pace_min}:{pace_sec:02d} /km
    - Local Start Time: {start_date}
    - Activity Date: {start_date_str}

    RAW PAYLOAD (for additional context):
    {json.dumps(object_data, indent=2)}

    YOUR MISSION:

    1. FETCH INTERVALS.ICU TRAINING DATA (FOLLOW THIS EXACT WORKFLOW!)
        ⚠️  CRITICAL: Follow this EXACT sequence - do NOT deviate!
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🔹 CALL #1: IntervalsIcu__get_activities
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Tool: IntervalsIcu__get_activities
        Input: {{"start_date": "{start_date_str}", "end_date": "{start_date_str}", "limit": 10}}
        
        This returns:
        ```
        Activity: 🏃 Lunch Run - 9.9K
        ID: i107359661          ← COPY THIS ID!
        Type: Run
        ```
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🔹 CALL #2: IntervalsIcu__get_activity_details (DIFFERENT TOOL!)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        ⚠️  MANDATORY: Your next tool call MUST be IntervalsIcu__get_activity_details
        ⚠️  DO NOT call IntervalsIcu__get_activities again - CrewAI will block duplicate inputs!
        
        Tool: IntervalsIcu__get_activity_details
        Input: {{"activity_id": "THE_ID_FROM_CALL_1"}}
        
        Example: If Call #1 returned "ID: i107359661", then:
        Input: {{"activity_id": "i107359661"}}
        
        This gives you the FULL training data with intervals, HR zones, power zones, etc.
        
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🔹 AFTER 2 CALLS: Generate JSON output IMMEDIATELY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        After these 2 tool calls, STOP calling tools and generate the final JSON.

        ⚠️  CRITICAL: DO NOT EXPLAIN what you found. DO NOT say "Perfect!" or "I found...".
        Your NEXT message after tool calls must be ONLY the JSON object!

        ⚠️  CRITICAL RULES:
        - Call #1 MUST be IntervalsIcu__get_activities
        - Call #2 MUST be IntervalsIcu__get_activity_details (with the ID from Call #1)
        - DO NOT call get_activities twice - the second call will be blocked!
        - DO NOT call get_activities with different parameters - use the ID immediately!
        - If you can't extract an ID, skip to generating JSON with Strava data only
        - After getting data, OUTPUT JSON DIRECTLY - no commentary or thinking!

    2. PRODUCE A TITLE (≤ 50 characters)
        - Highlight the workout intent or key achievement
        - Include 1-2 relevant emojis for energy
        - Keep wording natural and punchy

    3. PRODUCE A DESCRIPTION (≤ 500 characters)
        - Open with the workout goal/type
        - Summarise the structure (warm-up, blocks, cool-down)
        - Mention 2-3 concrete metrics (pace, HR, splits, power)
        - Close with a short feeling/mindset note
        - End with the watermark "Generated by AI" on a new line
        - Preserve readability with short sentences or line breaks

    4. CLASSIFY THE WORKOUT
        - Choose the best matching type (Easy Run, Tempo Run, Intervals, Long Run, Recovery, etc.)

    OUTPUT CONTRACT:
    - After gathering data (or after 3 tool calls), OUTPUT THE FINAL JSON IMMEDIATELY
    - Respond strictly with valid JSON matching the GeneratedActivityContent schema
    - Do not wrap the JSON in markdown fences
    - Respect character limits and ensure emojis render correctly
    - DO NOT continue calling tools endlessly - use max 3-5 tool calls then STOP

    CRITICAL FINAL ACTION:
    After retrieving and analyzing all data (or reaching tool limit), you MUST output a JSON object with this exact structure:
    {{
        "title": "Activity title with emoji",
        "description": "Activity description text\\n\\nGenerated by AI",
        "workout_type": "Workout classification",
        "key_metrics": {{"metric_name": "value", ...}}
    }}
    
    Do NOT include any explanatory text, thoughts, or reasoning in your final output.
    Your final message must be ONLY the JSON object, nothing else.
    """
    
    return Task(
        description=description,
        agent=agent,
        expected_output=(
            'ONLY a JSON object starting with { and ending with }. NO other text.\n'
            'Required fields:\n'
            '- "title": activity title with emoji (≤50 chars)\n'
            '- "description": description ending with "Generated by AI"\n'
            '- "workout_type": e.g. "Tempo", "Intervals", "Easy Ride"\n'
            '- "key_metrics": dict of metrics\n\n'
            'Example:\n'
            '{"title": "🚴 Lunch Ride 53K", "description": "Zone 2 endurance ride...\\n\\nGenerated by AI", '
            '"workout_type": "Endurance Ride", "key_metrics": {"avg_power": "278W", "avg_hr": "148 bpm"}}\n\n'
            'FORBIDDEN: explanations, "Perfect!", "I found...", "Let me...", thinking text.'
        ),
        # CRITICAL: output_json disables tool calling! Must parse JSON manually instead.
        # output_json=GeneratedActivityContent,
    )
