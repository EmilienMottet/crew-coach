"""Tasks for the Strava Description Crew and Meal Planning Crew."""
from .description_task import create_description_task
from .music_task import create_music_task
from .privacy_task import create_privacy_task
from .translation_task import create_translation_task

# Meal planning tasks
from .hexis_analysis_task import create_hexis_analysis_task
from .weekly_structure_task import create_weekly_structure_task
from .meal_generation_task import create_meal_generation_task
from .meal_compilation_task import create_meal_compilation_task
from .nutritional_validation_task import create_nutritional_validation_task
from .mealy_integration_task import create_mealy_integration_task

__all__ = [
    "create_description_task",
    "create_music_task",
    "create_privacy_task",
    "create_translation_task",
    "create_hexis_analysis_task",
    "create_weekly_structure_task",
    "create_meal_generation_task",
    "create_meal_compilation_task",
    "create_nutritional_validation_task",
    "create_mealy_integration_task",
]
