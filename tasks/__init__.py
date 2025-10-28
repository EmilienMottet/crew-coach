"""Tasks for the Strava Description Crew."""
from .description_task import create_description_task
from .privacy_task import create_privacy_task
from .translation_task import create_translation_task

__all__ = [
    "create_description_task",
    "create_privacy_task",
    "create_translation_task"
]
