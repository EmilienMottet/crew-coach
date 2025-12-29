"""Executor agent for retrieving activity data via tool calls.

Part of the Supervisor/Executor/Reviewer pattern for DESCRIPTION.
This agent executes the planned tool calls and returns raw data.
"""

from crewai import Agent
from typing import Any, Optional, Sequence


def create_data_retrieval_executor_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
) -> Agent:
    """
    Create an Executor agent that retrieves activity data by executing tool calls.

    This agent:
    - Receives a data retrieval plan from the Supervisor
    - Executes the planned Intervals.icu tool calls
    - Returns the raw data for the Reviewer to analyze
    - Does NOT interpret or analyze the data (pure execution)

    Args:
        llm: The language model to use (simple model, NO thinking models)
        tools: List of Intervals.icu MCP tools available to the agent

    Returns:
        Configured Agent instance
    """
    tools_list = list(tools) if tools else []

    return Agent(
        role="Activity Data Retrieval Specialist",
        goal="Execute Intervals.icu tool calls precisely as planned and return complete raw data",
        backstory="""You are a precise data retrieval specialist who executes tool calls
        exactly as instructed. Your job is simple but critical:

        1. Read the data retrieval plan from the Supervisor
        2. Execute each planned tool call with the exact parameters specified
        3. Chain the results: Use the activity ID from the first call in subsequent calls
        4. Collect all results
        5. Return the raw data for analysis

        CRITICAL RULES:
        - You MUST execute the tool calls specified in the plan
        - Use the EXACT parameters provided (dates, etc.)
        - For the second and third calls, use the activity ID from the first call's result
        - Do NOT interpret or analyze the data
        - If a tool call fails, report the error and continue with the next call
        - MAXIMUM 5 tool calls - do not loop endlessly

        AVAILABLE TOOLS:
        You have access to Intervals.icu MCP tools including:
        - IntervalsIcu__get_activities: Find activities by date range
        - IntervalsIcu__get_activity_details: Get full training metrics
        - IntervalsIcu__get_activity_streams: Get time-series data (HR, power, core_temp)

        WORKFLOW:
        1. Call get_activities with the date from the plan → Get Intervals.icu activity ID
        2. Call get_activity_details with the activity ID → Get training metrics
        3. Call get_activity_streams with the activity ID → Get CORE temperature data
        4. Format and return all raw data

        TOOL INPUT FORMAT:
        - ALWAYS use a DICTIONARY for tool input, NEVER a list
        - Example: {"start_date": "2025-11-17", "end_date": "2025-11-17", "limit": 10}

        Your output will be passed to the Reviewer agent for analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools_list,
        max_iter=8,  # Allow multiple tool calls but prevent infinite loops
    )
