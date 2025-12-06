"""Agents for the Strava Description Crew and Meal Planning Crew."""
from .description_agent import create_description_agent
from .music_agent import create_music_agent
from .privacy_agent import create_privacy_agent
from .translation_agent import create_translation_agent
from .lyrics_agent import create_lyrics_agent

# Meal planning agents
from .hexis_analysis_agent import create_hexis_analysis_agent
from .weekly_structure_agent import create_weekly_structure_agent
from .meal_generation_agent import create_meal_generation_agent
from .meal_compilation_agent import create_meal_compilation_agent
from .nutritional_validation_agent import create_nutritional_validation_agent
from .mealy_integration_agent import create_mealy_integration_agent

# Supervisor/Executor/Reviewer pattern for MEAL_GENERATION
from .meal_planning_supervisor_agent import create_meal_planning_supervisor_agent
from .ingredient_validation_executor_agent import create_ingredient_validation_executor_agent
from .meal_recipe_reviewer_agent import create_meal_recipe_reviewer_agent

# Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS
from .hexis_data_supervisor_agent import create_hexis_data_supervisor_agent
from .hexis_data_executor_agent import create_hexis_data_executor_agent
from .hexis_analysis_reviewer_agent import create_hexis_analysis_reviewer_agent

# Supervisor/Executor/Reviewer pattern for DESCRIPTION
from .description_supervisor_agent import create_description_supervisor_agent
from .data_retrieval_executor_agent import create_data_retrieval_executor_agent
from .description_reviewer_agent import create_description_reviewer_agent

__all__ = [
    "create_description_agent",
    "create_music_agent",
    "create_privacy_agent",
    "create_translation_agent",
    "create_lyrics_agent",
    "create_hexis_analysis_agent",
    "create_weekly_structure_agent",
    "create_meal_generation_agent",
    "create_meal_compilation_agent",
    "create_nutritional_validation_agent",
    "create_mealy_integration_agent",
    # Supervisor/Executor/Reviewer pattern for MEAL_GENERATION
    "create_meal_planning_supervisor_agent",
    "create_ingredient_validation_executor_agent",
    "create_meal_recipe_reviewer_agent",
    # Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS
    "create_hexis_data_supervisor_agent",
    "create_hexis_data_executor_agent",
    "create_hexis_analysis_reviewer_agent",
    # Supervisor/Executor/Reviewer pattern for DESCRIPTION
    "create_description_supervisor_agent",
    "create_data_retrieval_executor_agent",
    "create_description_reviewer_agent",
]
