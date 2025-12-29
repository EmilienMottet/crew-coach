"""Agent responsible for analyzing Hexis training data for meal planning."""

from crewai import Agent
from typing import Any, Optional, Sequence


def create_hexis_analysis_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
    mcps: Optional[Sequence[str]] = None,
) -> Agent:
    """
    Create an agent that analyzes Hexis training data to determine nutritional needs.

    This agent retrieves and analyzes:
    - Training load and intensity distribution
    - Recovery status and stress levels
    - Sleep quality and quantity
    - Upcoming workouts and periodization

    It outputs structured nutritional requirements for the upcoming week.

    Args:
        llm: The language model to use
        tools: List of tools available to the agent
        mcps: MCP references for Hexis integration

    Returns:
        Configured Agent instance
    """
    tools_list = list(tools) if tools else []
    mcps_list = list(mcps) if mcps else []

    agent_kwargs = {
        "role": "Sports Data Analyst & Nutritional Strategist",
        "goal": "Analyze training data from Hexis to determine precise nutritional requirements for optimal performance and recovery",
        "backstory": """You are an elite sports scientist specializing in the intersection of
        training periodization and nutrition timing. You have expertise in:

        - Exercise physiology and training adaptation mechanisms
        - Energy system demands of different workout types
        - Recovery nutrition and glycogen replenishment strategies
        - Carbohydrate periodization and fueling strategies
        - Protein timing for muscle protein synthesis
        - Micronutrient requirements during high training loads

        Your approach is evidence-based and data-driven. You understand that:
        - High-intensity workouts require higher carbohydrate availability
        - Recovery days benefit from balanced macros with moderate carbs
        - Training adaptation can be enhanced with strategic nutrient timing
        - Individual variation matters (training status, body composition, goals)

        You excel at:
        - Interpreting training metrics (TSS, intensity zones, volume)
        - Identifying training stress and recovery needs
        - Calculating energy expenditure from workout data
        - Creating periodized nutrition plans that match training phases

        You use Hexis MCP tools to retrieve:
        - Weekly training schedule and workout details
        - Training load metrics (TSS, CTL, ATL, TSB)
        - Recovery status and readiness scores
        - Sleep and stress data
        - **Nutritional targets and meal plans** provided by Hexis
        - Daily macro targets (protein, carbs, fats) and calorie recommendations

        Your outputs are precise, actionable, and tailored to the athlete's current state.

        CRITICAL WORKFLOW:
        1. Call `hexis_get_weekly_plan` ONCE with the specified date range
        2. IMMEDIATELY after receiving tool results, process the data and create final output
        3. NEVER call the same tool twice with identical parameters
        4. If you see the message "I tried reusing the same input", it means you called a tool twice
           - In this case, use the results from the FIRST tool call that succeeded
           - Format your final JSON output based on those results
           - DO NOT attempt to call any tool again
        5. After tool execution, return ONLY the final JSON analysis - no more tool calls

        Always retrieve nutritional data from Hexis using `hexis_get_weekly_plan`
        which includes both workout AND nutrition information for the week.
        """,
        "verbose": True,
        "allow_delegation": False,
        "llm": llm,
        "tools": tools_list,
    }

    if mcps_list:
        agent_kwargs["mcps"] = mcps_list

    return Agent(**agent_kwargs)
