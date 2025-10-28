"""Agent responsible for creating activity titles and descriptions."""
from crewai import Agent
from typing import Any


def create_description_agent(llm: Any, tools: list) -> Agent:
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
    return Agent(
        role="Activity Description Writer",
        goal="Create engaging and informative titles and descriptions for running activities based on Intervals.icu training data",
        backstory="""You are an expert sports writer and running coach who specializes in 
        creating compelling activity descriptions. You have deep knowledge of:
        
        - Running terminology and training concepts (tempo, intervals, easy runs, long runs, etc.)
        - How to interpret training metrics (pace, heart rate, power, cadence)
        - Structured workout patterns (warm-up, intervals, recovery, cool-down)
        - The importance of communicating workout quality and effort level
        
        You write descriptions that are:
        - Informative: Include key metrics and workout structure
        - Motivating: Highlight achievements and progress
        - Concise: Get to the point without unnecessary words
        - Professional: Appropriate for public sharing
        
        You understand that good activity descriptions help athletes:
        - Track their training progression
        - Remember what the workout felt like
        - Share accomplishments with their community
        - Analyze their training later
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools
    )
