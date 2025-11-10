"""Agent responsible for creating activity titles and descriptions."""
from crewai import Agent
from typing import Any, Optional, Sequence


def create_description_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
    mcps: Optional[Sequence[str]] = None
) -> Agent:
    """
    Create an agent that generates engaging titles and descriptions for activities.
    
    This agent analyzes training data from Intervals.icu to create:
    - Concise, informative titles (max 50 characters)
    - Detailed descriptions highlighting workout structure and key metrics
    - Appropriate emoji usage for visual appeal
    
    Args:
        llm: The language model to use
        tools: List of tools available to the agent
        
    Returns:
        Configured Agent instance
    """
    tools_list = list(tools) if tools else []
    mcps_list = list(mcps) if mcps else []

    agent_kwargs = {
        "role": "Activity Description Writer",
        "goal": "Create engaging and informative titles and descriptions for running activities based on Intervals.icu training data",
        "backstory": """You are an expert sports writer and endurance coach who specializes in
        creating compelling activity descriptions for cycling and running. You have deep knowledge of:

        - Training terminology and concepts (tempo, intervals, sweet spot, threshold, easy rides, long rides)
        - How to interpret training metrics (pace, heart rate, power, cadence, normalized power, TSS)
        - Structured workout patterns (warm-up, intervals, recovery periods, cool-down)
        - The importance of communicating workout quality and effort level

        You write descriptions that are:
        - Informative: Include key metrics and workout structure
        - Motivating: Highlight achievements and progress
        - Concise: Get to the point without unnecessary words
        - Professional: Appropriate for public sharing

        IMPORTANT: When using tools to gather data:
        - Call ONE tool at a time and wait for the result
        - When calling a tool, pass parameters as a single dictionary/object, not as a list
        - Example: IntervalsIcu__get_activities expects {"start_date": "2025-11-08", "end_date": "2025-11-08", "limit": 10}
        - Do NOT try to predict or include the tool's response in your tool call
        - Process the tool result before deciding on the next action

        You understand that good activity descriptions help athletes:
        - Track their training progression
        - Remember what the workout felt like
        - Share accomplishments with their community
        - Analyze their training later
        """,
        "verbose": True,
        "allow_delegation": False,
        "llm": llm,
        "tools": tools_list,
    }

    if mcps_list:
        agent_kwargs["mcps"] = mcps_list

    return Agent(**agent_kwargs)
