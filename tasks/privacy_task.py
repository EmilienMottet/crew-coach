"""Task for privacy and compliance checking."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

import pytz
from crewai import Task

from schemas import PrivacyAssessment


def create_privacy_task(
    agent,
    activity_data: Dict[str, Any],
    generated_content: Dict[str, Any] | str,
) -> Task:
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
    
    if isinstance(generated_content, str):
        content_payload = generated_content
    else:
        content_payload = json.dumps(generated_content, indent=2)

    description = f"""
    Review the activity below for privacy compliance and adherence to working hours policy.

    ACTIVITY TIMING SUMMARY:
    - Start Date/Time (UTC): {start_date_local}
    - Local Time (Europe/Paris): {start_time_str}
    - Day of Week: {day_of_week}

    WORKING HOURS POLICY (Europe/Paris):
    - Morning block: {work_start_morning} → {work_end_morning}
    - Afternoon block: {work_start_afternoon} → {work_end_afternoon}
    - Weekends: Usually acceptable for public visibility

    GENERATED CONTENT TO INSPECT (title & description in English):
    {content_payload}

    DECISION PLAYBOOK:

    1. DETECT GENERATION FAILURES
       - Look for error language ("unable to complete", "connection issue", etc.)
       - Detect placeholder titles such as "Activity" or "Activity completed"
       - If failure detected:
         • Replace title with fallback "{fallback_title}"
         • Force description to empty string
         • Approve privacy (content-free is safe)
         • Decide visibility strictly based on working hours
         • Document that a fallback was issued

    2. SCAN FOR SENSITIVE INFORMATION
       - No surnames, precise addresses, contact details, finances, or medical diagnoses
       - Generic locations ("park", "city trail") are fine; exact street names are not
       - Ensure tone remains professional and suitable for public Strava posts

    3. ENFORCE WORKING HOURS POLICY
       - Activities between the configured work slots must be private
       - Outside of work hours can remain public unless other privacy issues exist

    4. RETURN ACTIONABLE GUIDANCE
       - Populate `issues_found` with specific red flags (empty list when safe)
       - Supply sanitized `recommended_changes` only when modifications are required
       - Provide a concise `reasoning` paragraph summarising the decision

    RESPONSE FORMAT:
    - Output must be valid JSON compliant with the PrivacyAssessment schema
    - Do not surround the JSON with markdown fences
    - Use `null` for unchanged title/description in `recommended_changes`
    - Keep explanations short, precise, and professional
    """
    
    return Task(
        description=description,
        agent=agent,
        expected_output=(
            "Valid JSON adhering to the PrivacyAssessment schema (privacy_approved, "
            "during_work_hours, should_be_private, issues_found, recommended_changes, reasoning)"
        ),
        output_json=PrivacyAssessment,
        # Note: Do NOT use context parameter with strings, it expects Task objects
    )
