"""Task for generating activity descriptions."""
from crewai import Task
from typing import Dict, Any


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
    
    description = f"""
    Analyze the following Strava activity and create an engaging title and description.
    
    ACTIVITY INFORMATION:
    - Strava Activity ID: {strava_id}
    - Type: {activity_type}
    - Distance: {distance:.2f} km
    - Duration: {moving_time_min:.0f} minutes
    - Average Pace: {pace_min}:{pace_sec:02d} /km
    - Start Time: {start_date}
    
    YOUR TASK:
    
     1. USE INTERVALS.ICU DATA VIA MCP TOOLS:
         - Call IntervalsIcu__get_activity_details with activity_id "{strava_id}" to get detailed metrics
         - Call IntervalsIcu__get_activity_intervals with activity_id "{strava_id}" to understand workout structure
       - Analyze the data to understand the type of workout (tempo, intervals, easy run, etc.)
    
    2. CREATE A TITLE (max 50 characters):
       - Should capture the essence of the workout
       - Include key metric or achievement
       - Use 1-2 relevant emojis (running shoe, fire, lightning, etc.)
       - Examples:
         * "🏃 12K Tempo Run - Sub 4:00 pace"
         * "⚡ 5x1000m Intervals - PR effort"
         * "🔥 20K Long Run - Strong finish"
    
    3. WRITE A DESCRIPTION (max 500 characters):
       - Start with the workout type/purpose
       - Highlight the structure (warm-up, main set, cool-down)
       - Include 2-3 key metrics:
         * Average pace and heart rate
         * Interval paces if applicable
         * Notable achievements or segments
       - End with how it felt or the goal achieved
       - Use line breaks for readability
       - Examples:
         * "Tempo run focusing on threshold pace. Started with 2K warm-up, then 8K at tempo pace (avg 3:55/km, 165 bpm), finished with 2K cool-down. Felt strong throughout - exactly the effort I was targeting! 💪"
         * "Interval session: 5x1000m with 400m recovery. Target was sub-4:00 pace. Nailed it: 3:52, 3:55, 3:54, 3:56, 3:50 (avg HR 172 bpm). Legs felt fresh, great workout! ⚡"
    
    4. IDENTIFY WORKOUT TYPE:
       - Based on the interval data and pacing, determine the workout category
       - Common types: Easy Run, Tempo Run, Intervals, Long Run, Recovery, Fartlek, Race
    
    OUTPUT FORMAT:
    Return a JSON object with this exact structure:
    {{
        "title": "Your generated title here",
        "description": "Your generated description here",
        "workout_type": "Tempo Run, Intervals, Easy Run, etc.",
        "key_metrics": {{
            "average_pace": "4:30 /km",
            "average_hr": "145 bpm",
            "max_hr": "165 bpm",
            "intervals_summary": "5x1000m @ 3:50-3:56" (if applicable)
        }}
    }}
    """
    
    return Task(
        description=description,
        agent=agent,
        expected_output="A JSON object containing title, description, workout_type, and key_metrics"
    )
