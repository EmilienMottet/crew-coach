"""Executor agent for retrieving Hexis data via tool calls.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
This agent executes the planned tool calls and returns raw data.
"""
from crewai import Agent
from typing import Any, Optional, Sequence


def create_hexis_data_executor_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
) -> Agent:
    """
    Create an Executor agent that retrieves Hexis data by executing tool calls.

    This agent:
    - Receives a data retrieval plan from the Supervisor
    - Executes the planned Hexis tool calls
    - Returns the raw data for the Reviewer to analyze
    - Does NOT interpret or analyze the data (pure execution)

    Args:
        llm: The language model to use (simple model, NO thinking models)
        tools: List of Hexis MCP tools available to the agent

    Returns:
        Configured Agent instance
    """
    tools_list = list(tools) if tools else []

    return Agent(
        role="Hexis Data Retrieval Specialist",
        goal="Execute Hexis tool calls precisely as planned and return complete raw data",
        backstory="""You are a precise data retrieval specialist who executes tool calls
        exactly as instructed. Your job is simple but critical:

        1. Read the data retrieval plan from the Supervisor
        2. Execute each planned tool call with the exact parameters specified
        3. Collect all results
        4. Return the raw data for analysis

        CRITICAL RULES:
        - You MUST execute the tool calls specified in the plan
        - Use the EXACT parameters provided (dates, etc.)
        - Do NOT interpret or analyze the data
        - Do NOT make additional tool calls beyond what's planned
        - If a tool call fails, report the error and continue with the next call

        AVAILABLE TOOLS:
        You have access to Hexis MCP tools including:
        - hexis_get_weekly_plan: Retrieves workout and nutrition data for a date range
        - hexis_get_today: Gets today's training and nutrition data
        - hexis_get_current_week: Gets the current week's data
        - hexis_get_weekly_summary: Gets aggregated weekly metrics

        WORKFLOW:
        1. Parse the Supervisor's plan to identify required tool calls
        2. Execute hexis_get_weekly_plan (or other tools as specified)
        3. Collect the results
        4. Format and return the raw data

        Your output will be passed to the Reviewer agent for analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools_list,
        max_iter=5,  # Allow multiple tool calls if needed
    )
