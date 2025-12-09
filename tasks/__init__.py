"""Tasks for the Strava Description Crew and Meal Planning Crew."""
from .description_task import create_description_task
from .music_task import create_music_task
from .privacy_task import create_privacy_task
from .translation_task import create_translation_task
from .lyrics_task import create_lyrics_task

# Meal planning tasks
from .hexis_analysis_task import create_hexis_analysis_task
from .weekly_structure_task import create_weekly_structure_task
from .meal_generation_task import create_meal_generation_task
from .meal_compilation_task import create_meal_compilation_task
from .nutritional_validation_task import create_nutritional_validation_task
from .mealy_integration_task import create_mealy_integration_task

# Supervisor/Executor/Reviewer pattern for MEAL_GENERATION
from .meal_planning_supervisor_task import create_meal_planning_supervisor_task
from .ingredient_validation_executor_task import create_ingredient_validation_executor_task
from .meal_recipe_reviewer_task import create_meal_recipe_reviewer_task

# Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS
from .hexis_data_supervisor_task import create_hexis_data_supervisor_task
from .hexis_data_executor_task import create_hexis_data_executor_task
from .hexis_analysis_reviewer_task import (
    create_hexis_analysis_reviewer_task,
    _extract_daily_meal_targets,  # Deterministic extraction for reliability
)

# Supervisor/Executor/Reviewer pattern for DESCRIPTION
from .description_supervisor_task import create_description_supervisor_task
from .data_retrieval_executor_task import create_data_retrieval_executor_task
from .description_reviewer_task import create_description_reviewer_task

__all__ = [
    "create_description_task",
    "create_music_task",
    "create_privacy_task",
    "create_translation_task",
    "create_lyrics_task",
    "create_hexis_analysis_task",
    "create_weekly_structure_task",
    "create_meal_generation_task",
    "create_meal_compilation_task",
    "create_nutritional_validation_task",
    "create_mealy_integration_task",
    # Supervisor/Executor/Reviewer pattern for MEAL_GENERATION
    "create_meal_planning_supervisor_task",
    "create_ingredient_validation_executor_task",
    "create_meal_recipe_reviewer_task",
    # Supervisor/Executor/Reviewer pattern for HEXIS_ANALYSIS
    "create_hexis_data_supervisor_task",
    "create_hexis_data_executor_task",
    "create_hexis_analysis_reviewer_task",
    "_extract_daily_meal_targets",
    # Supervisor/Executor/Reviewer pattern for DESCRIPTION
    "create_description_supervisor_task",
    "create_data_retrieval_executor_task",
    "create_description_reviewer_task",
]
