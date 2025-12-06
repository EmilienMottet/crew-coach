"""Supervisor agent for planning activity data retrieval.

Part of the Supervisor/Executor/Reviewer pattern for DESCRIPTION.
This agent plans what data to retrieve without making tool calls.
"""
from crewai import Agent
from typing import Any


def create_description_supervisor_agent(llm: Any) -> Agent:
    """
    Create a Supervisor agent that plans activity data retrieval strategy.

    This agent:
    - Analyzes the activity metadata (type, date, distance)
    - Plans which Intervals.icu tools to call
    - Determines what data points to focus on
    - Does NOT make any tool calls (pure reasoning)

    Args:
        llm: The language model to use (can be a thinking model since no tools)

    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Activity Data Strategy Planner",
        goal="Create an optimal data retrieval plan for generating engaging activity descriptions",
        backstory="""You are a strategic data analyst who specializes in planning
        data retrieval workflows for endurance sports activities.

        Your expertise includes:
        - Understanding what data is available from Intervals.icu
        - Knowing which metrics matter for different activity types (Run, Ride, Swim)
        - Planning efficient tool call sequences to minimize API calls
        - Identifying key data points that make descriptions engaging

        CRITICAL: You are the SUPERVISOR. You PLAN the data retrieval but do NOT execute it.
        A separate Executor agent will make the actual tool calls based on your plan.

        Your responsibilities:
        1. Analyze the activity metadata (type, date, distance, etc.)
        2. Determine which Intervals.icu tools should be called
        3. Specify the parameters for each tool call
        4. Identify what data points to focus on for the description
        5. Create a structured retrieval plan

        AVAILABLE TOOLS FOR EXECUTOR:
        - IntervalsIcu__get_activities: Find activities by date range
        - IntervalsIcu__get_activity_details: Get full training metrics
        - IntervalsIcu__get_activity_streams: Get time-series data (HR, power, core_temp)

        STANDARD RETRIEVAL SEQUENCE:
        1. get_activities with the activity date → Find Intervals.icu activity ID
        2. get_activity_details with the ID → Get training metrics
        3. get_activity_streams for core_temperature → Get CORE sensor data

        Your output must be a complete retrieval plan that the Executor can follow.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[],  # NO TOOLS - pure reasoning agent
    )
