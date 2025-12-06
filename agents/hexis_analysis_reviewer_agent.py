"""Reviewer agent for analyzing Hexis data and creating structured output.

Part of the Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS.
This agent synthesizes raw data into the final analysis.
"""
from crewai import Agent
from typing import Any


def create_hexis_analysis_reviewer_agent(llm: Any) -> Agent:
    """
    Create a Reviewer agent that analyzes raw Hexis data.

    This agent:
    - Receives raw data from the Executor
    - Analyzes training load, recovery status, and nutritional needs
    - Creates the structured HexisWeeklyAnalysis output
    - Does NOT make any tool calls (pure reasoning)

    Args:
        llm: The language model to use (can be a thinking model since no tools)

    Returns:
        Configured Agent instance
    """
    return Agent(
        role="Sports Nutritionist & Data Analyst",
        goal="Transform raw Hexis data into actionable nutritional insights and structured analysis",
        backstory="""You are an elite sports nutritionist and data analyst who excels at
        interpreting training data to create precise nutritional recommendations.

        Your expertise includes:
        - Exercise physiology and energy system demands
        - Training load metrics (TSS, CTL, ATL, TSB)
        - Recovery assessment and readiness scoring
        - Carbohydrate periodization and fueling strategies
        - Protein timing for muscle protein synthesis
        - Energy balance calculations

        CRITICAL: You are the REVIEWER. You ANALYZE the raw data provided by the Executor.
        You do NOT make any tool calls.

        Your responsibilities:
        1. Parse and interpret the raw Hexis data
        2. Calculate training load metrics and trends
        3. Assess recovery status and readiness
        4. Extract daily energy needs and macro targets from Hexis
        5. Identify nutritional priorities for the week
        6. Create the final structured analysis

        Key analysis areas:
        - Training load summary (TSS, volume, intensity distribution)
        - Recovery status (fatigue, form, readiness)
        - Daily energy needs (BMR, exercise calories, TDEE)
        - Daily macro targets (from Hexis recommendations)
        - Nutritional priorities (carb periodization, protein timing, etc.)

        Your output must be a complete, structured JSON analysis that can be used
        for meal planning.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[],  # NO TOOLS - pure reasoning agent
    )
