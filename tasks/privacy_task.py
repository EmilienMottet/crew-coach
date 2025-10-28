"""Task for privacy and compliance checking."""
from crewai import Task
from typing import Dict, Any
from datetime import datetime
import os
import pytz


def create_privacy_task(agent, activity_data: Dict[str, Any], generated_content: str) -> Task:
    """
    Create a task for checking privacy and working hours compliance.
    
    Args:
        agent: The agent responsible for this task
        activity_data: Raw activity data from Strava webhook
        generated_content: The generated title and description to check
        
    Returns:
        Configured Task instance
    """
    object_data = activity_data.get("object_data", {})
    start_date_local = object_data.get("start_date_local", "")
    
    # Parse the start time to check working hours in Europe/Paris timezone
    try:
        # Format: "2025-10-27T11:54:41Z"
        start_datetime = datetime.fromisoformat(start_date_local.replace('Z', '+00:00'))
        
        # Convert to Europe/Paris timezone
        paris_tz = pytz.timezone('Europe/Paris')
        start_datetime_paris = start_datetime.astimezone(paris_tz)
        
        start_time_str = start_datetime_paris.strftime("%H:%M")
        day_of_week = start_datetime_paris.strftime("%A")
    except:
        start_time_str = "unknown"
        day_of_week = "unknown"
    
    work_start_morning = os.getenv("WORK_START_MORNING", "08:30")
    work_end_morning = os.getenv("WORK_END_MORNING", "12:00")
    work_start_afternoon = os.getenv("WORK_START_AFTERNOON", "14:00")
    work_end_afternoon = os.getenv("WORK_END_AFTERNOON", "17:00")
    
    # Extract basic activity metrics for fallback title generation
    distance = object_data.get("distance", 0) / 1000  # Convert to km
    activity_type = object_data.get("type", "Run")
    
    # Determine time of day for fallback title
    time_of_day = "Run"  # Default
    try:
        start_datetime = datetime.fromisoformat(start_date_local.replace('Z', '+00:00'))
        paris_tz = pytz.timezone('Europe/Paris')
        start_datetime_paris = start_datetime.astimezone(paris_tz)
        hour = start_datetime_paris.hour
        
        # Categorize by time of day (Europe/Paris timezone)
        if 5 <= hour < 12:
            time_of_day = "Morning Run"
        elif 12 <= hour < 14:
            time_of_day = "Lunch Run"
        elif 14 <= hour < 18:
            time_of_day = "Afternoon Run"
        elif 18 <= hour < 22:
            time_of_day = "Evening Run"
        else:  # 22-5h
            time_of_day = "Night Run"
    except:
        # If parsing fails, use default
        pass
    
    # Generate a simple fallback title based on activity type, time, and distance
    if activity_type == "Run":
        if distance >= 20:
            fallback_title = f"🏃 Long Run - {distance:.1f}K"
        else:
            fallback_title = f"🏃 {time_of_day} - {distance:.1f}K"
    else:
        fallback_title = f"{activity_type} - {distance:.1f}K"
    
    description = f"""
    Review the following activity details for privacy compliance and working hours policy.
    
    ACTIVITY TIMING:
    - Start Date/Time (UTC): {start_date_local}
    - Local Time (Europe/Paris): {start_time_str}
    - Day of Week: {day_of_week}
    
    WORKING HOURS POLICY (Europe/Paris timezone):
    - Morning: {work_start_morning} - {work_end_morning}
    - Afternoon: {work_start_afternoon} - {work_end_afternoon}
    - Weekends: Generally OK to be public
    
    GENERATED CONTENT TO REVIEW:
    {generated_content}
    
    YOUR TASK:
    
    1. DETECT FAILED GENERATION:
       Check if the generated content contains error messages or indicates failure:
       - Mentions of "unable to complete", "issues retrieving data", "verify connection"
       - Generic titles like "Activity completed" or "Activity"
       - Descriptions that explain technical errors instead of describing the workout
       - ANY indication that the generation process failed
       
       **IF GENERATION FAILED:**
       - Use fallback title: "{fallback_title}"
       - Set description to empty string: ""
       - Mark as approved (no privacy issues with empty content)
       - Set should_be_private based ONLY on working hours check
       - Explain that fallback was used due to generation failure
    
    2. CHECK FOR SENSITIVE INFORMATION:
       Review the title and description for:
       - Full names (first names only are acceptable)
       - Street names, addresses, specific locations beyond city/area
       - Phone numbers or email addresses
       - Financial information
       - Detailed medical conditions or injuries
       - Any other personally identifiable information
    
    3. VERIFY WORKING HOURS COMPLIANCE:
       - Determine if the activity occurred during working hours
       - If yes, it MUST be marked as private
       - Weekends are generally acceptable for public activities
       - Consider if the timing might reveal work schedule patterns
    
    4. PROVIDE RECOMMENDATIONS:
       - If issues found: Suggest sanitized versions of title/description
       - Determine visibility: public or private
       - Explain your reasoning clearly
       - If no changes needed, confirm the content is safe
    
    5. CONSIDER CONTEXT:
       - Training with a group: "Morning run with the crew" is OK
       - Location mentions: "Park run" is OK, "123 Main Street loop" is NOT OK
       - Personal records: Mentioning PRs and metrics is encouraged
       - Health notes: "Felt strong" is OK, "Running despite knee injury" might be TMI
    
    OUTPUT FORMAT:
    Return a JSON object with this exact structure:
    {{
        "privacy_approved": true/false,
        "during_work_hours": true/false,
        "should_be_private": true/false,
        "issues_found": [
            "List of specific privacy concerns if any"
        ],
        "recommended_changes": {{
            "title": "Sanitized title (only if changes needed, otherwise null)",
            "description": "Sanitized description (only if changes needed, otherwise null)"
        }},
        "reasoning": "Clear explanation of your decision and any recommendations"
    }}
    
    IMPORTANT:
    - Be specific about what needs to change and why
    - If content is safe, say so clearly
    - Always err on the side of privacy protection
    - Consider the user's professional reputation
    """
    
    return Task(
        description=description,
        agent=agent,
        expected_output="A JSON object with privacy assessment, recommendations, and reasoning"
        # Note: Do NOT use context parameter with strings, it expects Task objects
    )
