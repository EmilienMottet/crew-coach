"""Task for translating activity titles and descriptions."""
from crewai import Task
from typing import Any
import os


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
    Translate the following activity content to {target_language}.
    
    SOURCE LANGUAGE: {source_language}
    TARGET LANGUAGE: {target_language}
    
    CONTENT TO TRANSLATE:
    {content_to_translate}
    
    YOUR TASK:
    
    1. TRANSLATE THE TITLE:
       - Preserve all emojis in their original positions
       - Keep the title concise and impactful (max 50 characters)
       - Maintain the same tone and energy
       - Adapt running terminology appropriately
       - Ensure it sounds natural in {target_language}
    
    2. TRANSLATE THE DESCRIPTION:
       - Preserve all emojis and formatting (line breaks, etc.)
       - Keep the description informative (max 500 characters)
       - Translate technical terms accurately:
         * Tempo run, interval session, easy run, long run, etc.
         * Pace, heart rate, recovery, warm-up, cool-down
         * Keep units as-is (km, /km, bpm, etc.)
       - Maintain the motivational and personal tone
       - Adapt idioms and expressions naturally
    
    3. PRESERVE METADATA:
       - Keep workout_type and key_metrics exactly as they are
       - These are technical fields that should not be translated
       - Only translate the human-readable title and description
    
    TRANSLATION QUALITY GUIDELINES:
    
    - **Natural Language**: The translation should sound like it was originally written in {target_language}
    - **Sports Context**: Use terminology that athletes in {target_language} would actually use
    - **Character Limits**: 
      * Title: Maximum 50 characters (including emojis)
      * Description: Maximum 500 characters (including emojis)
    - **Tone Consistency**: Match the original's tone (motivational, analytical, casual, etc.)
    - **Formatting**: Preserve line breaks, bullet points, and structure
    
    COMMON TERMINOLOGY REFERENCE:
    
    French → English:
    - Sortie = Run/outing
    - Fractionné = Intervals
    - Tempo/Seuil = Tempo/Threshold
    - Récupération = Recovery
    - Échauffement = Warm-up
    - Retour au calme = Cool-down
    - Allure = Pace
    - Fréquence cardiaque = Heart rate
    - J'ai ressenti = I felt / Felt
    
    English → French:
    - Run = Sortie/Course
    - Intervals = Fractionné
    - Tempo = Tempo/Seuil
    - Recovery = Récupération
    - Warm-up = Échauffement
    - Cool-down = Retour au calme
    - Pace = Allure
    - Heart rate = Fréquence cardiaque
    - Felt = J'ai ressenti / Ressenti
    
    OUTPUT FORMAT:
    Return a JSON object with the same structure as the input, but with translated content:
    {{
        "title": "Translated title in {target_language}",
        "description": "Translated description in {target_language}",
        "workout_type": "Keep original (not translated)",
        "key_metrics": {{
            "Keep all metrics exactly as they are (not translated)"
        }}
    }}
    
    IMPORTANT:
    - Only translate the "title" and "description" fields
    - Keep "workout_type" and "key_metrics" unchanged
    - Ensure character limits are respected
    - Verify that emojis are preserved correctly
    """
    
    return Task(
        description=description,
        agent=agent,
        expected_output=f"A JSON object with title and description translated to {target_language}, preserving emojis and formatting"
    )
