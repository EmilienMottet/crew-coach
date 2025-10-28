"""Tools for accessing Intervals.icu data via MCP server."""
from crewai_tools import tool
from .mcp_client import get_mcp_client
import json


@tool("Get Intervals.icu Activity Details")
def get_intervals_activity_details(activity_id: str) -> str:
    """
    Get detailed information about a specific activity from Intervals.icu.
    
    This tool retrieves comprehensive data about a training activity including:
    - Duration, distance, and pace
    - Heart rate metrics (average, max)
    - Power data (if available)
    - Training load and intensity
    - Workout type and structure
    
    Args:
        activity_id: The Intervals.icu activity ID (from Strava ID mapping)
        
    Returns:
        Detailed activity information as JSON string
    """
    client = get_mcp_client()
    result = client.call_tool("IntervalsIcu__get_activity_details", {
        "activity_id": str(activity_id)
    })
    
    if "error" in result:
        return f"Error fetching activity details: {result['error']}"
    
    return json.dumps(result, indent=2)


@tool("Get Intervals.icu Activity Intervals")
def get_intervals_activity_intervals(activity_id: str) -> str:
    """
    Get interval/segment data for a specific activity from Intervals.icu.
    
    This tool provides detailed metrics for each interval in a workout:
    - Interval duration and distance
    - Power, heart rate, cadence for each interval
    - Pace and speed data
    - Grouped intervals (sets of repeats)
    
    Useful for understanding structured workouts like:
    - Tempo runs with warm-up/cool-down
    - Interval sessions (e.g., 5x1000m)
    - Fartlek or progressive runs
    
    Args:
        activity_id: The Intervals.icu activity ID
        
    Returns:
        Interval data as JSON string
    """
    client = get_mcp_client()
    result = client.call_tool("IntervalsIcu__get_activity_intervals", {
        "activity_id": str(activity_id)
    })
    
    if "error" in result:
        return f"Error fetching activity intervals: {result['error']}"
    
    return json.dumps(result, indent=2)


@tool("Get Recent Activities from Intervals.icu")
def get_recent_intervals_activities(limit: int = 10) -> str:
    """
    Get a list of recent activities from Intervals.icu.
    
    Useful for finding an activity by date or searching for a specific workout.
    
    Args:
        limit: Maximum number of activities to return (default: 10)
        
    Returns:
        List of recent activities as JSON string
    """
    client = get_mcp_client()
    result = client.call_tool("IntervalsIcu__get_activities", {
        "limit": limit,
        "include_unnamed": False
    })
    
    if "error" in result:
        return f"Error fetching activities: {result['error']}"
    
    return json.dumps(result, indent=2)
