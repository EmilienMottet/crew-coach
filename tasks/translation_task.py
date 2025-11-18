"""Task for translating activity titles and descriptions."""
from __future__ import annotations

import os

from crewai import Task

from schemas import TranslationPayload


def create_translation_task(agent, content_to_translate: str) -> Task:
    """
    Create a task for translating activity title and description.
    
    Args:
        agent: The agent responsible for this task
        content_to_translate: The JSON string containing title and description to translate
        
    Returns:
        Configured Task instance
    """
    # Get target language from environment (default: English)
    target_language = os.getenv("TRANSLATION_TARGET_LANGUAGE", "English")
    
    # Determine source language hint
    source_language = os.getenv("TRANSLATION_SOURCE_LANGUAGE", "French")

    description = f"""
    Translate the activity summary below to {target_language}.

    SOURCE LANGUAGE: {source_language}
    TARGET LANGUAGE: {target_language}

    CONTENT TO TRANSLATE (JSON):
    {content_to_translate}

    TRANSLATION CHECKLIST:

    1. TITLE (≤ 50 characters)
        - Preserve emoji placement exactly
        - Keep the punchy, motivational tone
        - Adapt running terminology idiomatically

    2. DESCRIPTION (≤ 500 characters)
        - Maintain line breaks and formatting cues
        - Translate running jargon precisely (tempo, intervals, recovery, etc.)
        - Keep metric units untouched (km, /km, bpm)
        - Convey the original emotional tone in {target_language}

    3. PROTECT STRUCTURED FIELDS
        - Do not translate `workout_type`
        - Do not translate `key_metrics`
        - Only the natural-language fields should change

    OUTPUT FORMAT EXAMPLE (French):
    Input:
    {{
        "title": "🏃 Morning Run - 12.3K",
        "description": "Easy recovery run...",
        "workout_type": "Recovery Run",
        "key_metrics": {{"distance": "12.3 km"}}
    }}

    Expected Output (translate ONLY title and description):
    {{
        "title": "🏃 Sortie matinale - 12.3K",
        "description": "Sortie récupération tranquille...",
        "workout_type": "Recovery Run",
        "key_metrics": {{"distance": "12.3 km"}}
    }}

    CRITICAL REQUIREMENTS:
    - Return ONLY the translated DATA, NOT a schema definition
    - Do NOT include "properties", "required", "type", or "additionalProperties" fields
    - No markdown fenced blocks (```json)
    - Ensure emojis render correctly
    - Title must be ≤50 characters
    - Description must be ≤500 characters
    """
    
    return Task(
        description=description,
        agent=agent,
        expected_output=(
            f"Valid JSON adhering to the TranslationPayload schema with title and description translated to {target_language}"
        ),
        output_json=TranslationPayload,
    )
