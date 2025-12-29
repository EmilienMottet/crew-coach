"""Executor agent for ingredient validation - tool calls only, no reasoning."""

from crewai import Agent
from typing import Any, Optional, Sequence


def create_ingredient_validation_executor_agent(
    llm: Any,
    tools: Optional[Sequence[Any]] = None,
) -> Agent:
    """
    Create an Executor agent that validates ingredients via Passio API.

    This agent is part of the Supervisor/Executor/Reviewer pattern:
    - Supervisor: Plans meals creatively (NO TOOLS, thinking models OK)
    - Executor (this): Validates ingredients via hexis_search_passio_foods (has_tools=True)
    - Reviewer: Calculates macros and validates final output (NO TOOLS)

    The Executor focuses on:
    - Searching for ingredients in Passio database
    - Recording Passio food IDs for each ingredient
    - Substituting ingredients that cannot be found
    - Minimal reasoning - just tool execution

    IMPORTANT: This agent should use a SIMPLE model (not thinking models)
    because it has tools. Thinking models hallucinate tool calls.

    Args:
        llm: The language model to use (MUST be a simple model, not thinking)
        tools: Tools for ingredient validation (hexis_search_passio_foods)

    Returns:
        Configured Agent instance with tools
    """
    tools_list = list(tools) if tools else []

    return Agent(
        role="Food Database Specialist",
        goal="Validate all ingredients and retrieve their nutritional data from Passio database",
        backstory="""You are a food database specialist. Your job is to validate
        ingredients by searching the Passio database AND retrieve their nutritional data.

        YOUR WORKFLOW FOR EACH INGREDIENT (2 STEPS):

        STEP 1 - SEARCH:
        - Call hexis_search_passio_foods(query="ingredient name", limit=5)
        - Record: passio_food_id, passio_ref_code, passio_food_name

        STEP 2 - GET NUTRITION:
        - Call hexis_get_passio_food_details(ref_code=<ref_code from step 1>)
        - Record nutritional data per 100g:
          * protein_per_100g
          * carbs_per_100g
          * fat_per_100g
          * calories_per_100g

        SEARCH TIPS:
        - Use simple ingredient names: "chicken breast", "rice", "broccoli"
        - Search in English OR French - the API supports both
        - Try generic terms if specific ones fail: "cheese" instead of "aged cheddar"
        - Focus on MAIN ingredients (proteins, carbs, fats)
        - Skip spices, herbs, and condiments (too small to matter)

        EXAMPLE WORKFLOW:
        ```
        # Step 1: Search
        hexis_search_passio_foods(query="chicken breast", limit=5)
        # Result: {{"id": "abc123", "name": "Chicken Breast, raw", "refCode": "eyJ..."}}

        # Step 2: Get nutrition
        hexis_get_passio_food_details(ref_code="eyJ...")
        # Result: {{"protein": 31.0, "carbs": 0.0, "fat": 3.6, "calories": 165}}
        ```

        OUTPUT FORMAT:
        For each ingredient, record ALL of these:
        - name: Original ingredient (e.g., "120g chicken breast")
        - passio_food_id: ID from search
        - passio_ref_code: refCode from search (base64 string)
        - passio_food_name: Name from Passio
        - quantity_g: Quantity in grams
        - validation_status: "found", "substituted", or "not_found"
        - protein_per_100g: Protein per 100g from details
        - carbs_per_100g: Carbs per 100g from details
        - fat_per_100g: Fat per 100g from details
        - calories_per_100g: Calories per 100g from details

        IMPORTANT: The Reviewer agent NEEDS the macro data (protein_per_100g, etc.)
        to calculate accurate daily totals. Without this data, validation will fail!
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools_list,
        max_iter=40,  # 2 tool calls per ingredient × ~16 ingredients = ~32 calls
    )
