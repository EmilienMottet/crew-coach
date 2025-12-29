"""Agent that merges daily meal plans into a weekly plan."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from crewai import Agent


def create_meal_compilation_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
) -> Agent:
    """Create an agent focused on assembling coherent weekly meal plans."""
    tools_list = list(tools) if tools else []

    return Agent(
        role="Meal Planning Compiler & Prep Strategist",
        goal=(
            "Consolidate daily meal plans into a cohesive weekly plan with a smart shopping "
            "list and actionable meal prep guidance without altering the provided meals"
        ),
        backstory=(
            "You are a registered dietitian and culinary operations expert who excels at "
            "turning individual meal plans into realistic weekly schedules. You evaluate "
            "portion consistency, spot redundant ingredients, and craft efficient shopping "
            "lists. You understand batch cooking strategies, storage best practices, and "
            "how to keep athletes engaged with practical guidance. When consolidating, "
            "you respect the exact meals provided while highlighting ingredient overlaps, "
            "prep sequencing, and time-saving tips."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools_list,
    )
