"""Pydantic models for structured task outputs."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class GeneratedActivityContent(BaseModel):
    """Structured payload for generated activity content."""

    title: str = Field(..., max_length=200, description="English activity title")
    description: str = Field(
        ..., max_length=1000, description="English activity description"
    )
    workout_type: str = Field(..., description="Workout classification label")
    key_metrics: Dict[str, str] = Field(
        default_factory=dict,
        description="Key metrics captured as label -> value pairs",
    )


class ActivityMusicSelection(BaseModel):
    """Soundtrack enrichment appended to the generated description."""

    updated_description: str = Field(
        ...,
        max_length=1000,
        description="English activity description with soundtrack details appended",
    )
    music_tracks: List[str] = Field(
        default_factory=list,
        description="Ordered list of '<artist> – <track>' entries included in the summary",
    )


class PrivacyRecommendations(BaseModel):
    """Recommended sanitized content from the privacy agent."""

    title: Optional[str] = Field(
        default=None, description="Optional sanitized title to publish"
    )
    description: Optional[str] = Field(
        default=None, description="Optional sanitized description to publish"
    )


class PrivacyAssessment(BaseModel):
    """Privacy and compliance assessment outcome."""

    privacy_approved: bool = Field(..., description="Whether content is privacy-safe")
    during_work_hours: bool = Field(
        ..., description="Whether the activity occurred during work hours"
    )
    should_be_private: bool = Field(
        ..., description="Recommended Strava visibility (True = private)"
    )
    issues_found: List[str] = Field(
        default_factory=list, description="List of privacy issues detected"
    )
    recommended_changes: PrivacyRecommendations = Field(
        default_factory=PrivacyRecommendations,
        description="Suggested sanitized content when issues are found",
    )
    reasoning: str = Field(..., description="Plain-language explanation")


class TranslationPayload(GeneratedActivityContent):
    """Translated activity content (title/description in target language)."""

    pass


# ============================================================================
# Meal Planning Schemas (for crew_mealy.py)
# ============================================================================


class MacroTargets(BaseModel):
    """Daily macronutrient targets in grams."""

    protein_g: float = Field(..., description="Protein target in grams")
    carbs_g: float = Field(..., description="Carbohydrate target in grams")
    fat_g: float = Field(..., description="Fat target in grams")
    calories: int = Field(..., description="Total calorie target")


class HexisWeeklyAnalysis(BaseModel):
    """Analysis output from Hexis training data."""

    week_start_date: str = Field(..., description="ISO format start date (YYYY-MM-DD)")
    week_end_date: str = Field(..., description="ISO format end date (YYYY-MM-DD)")
    training_load_summary: str = Field(
        ..., description="Summary of weekly training load (TSS, hours, intensity)"
    )
    recovery_status: str = Field(
        ..., description="Current recovery status and recommendations"
    )
    daily_energy_needs: Dict[str, int] = Field(
        ...,
        description="Calorie needs per day (key=day_name, value=calories)",
    )
    daily_macro_targets: Dict[str, MacroTargets] = Field(
        ...,
        description="Macro targets per day (key=day_name, value=MacroTargets)",
    )
    nutritional_priorities: List[str] = Field(
        ...,
        description="Key nutritional priorities for the week (e.g., 'High carbs pre-race', 'Recovery focus')",
    )


class DailyNutritionTarget(BaseModel):
    """Nutritional targets for a single day."""

    day_name: str = Field(..., description="Day of week (e.g., 'Monday')")
    date: str = Field(..., description="ISO format date (YYYY-MM-DD)")
    calories: int = Field(..., description="Total calorie target")
    macros: MacroTargets = Field(..., description="Macro targets for the day")
    training_context: str = Field(
        ..., description="Training context for the day (e.g., 'Hard interval session', 'Rest day')"
    )
    meal_timing_notes: str = Field(
        ..., description="Guidance on meal timing (e.g., 'High-carb breakfast pre-workout')"
    )


class WeeklyNutritionPlan(BaseModel):
    """Structured weekly nutrition plan with daily targets."""

    week_start_date: str = Field(..., description="ISO format start date")
    week_end_date: str = Field(..., description="ISO format end date")
    daily_targets: List[DailyNutritionTarget] = Field(
        ..., description="Nutrition targets for each day of the week"
    )
    weekly_summary: str = Field(
        ..., description="High-level summary of the week's nutrition strategy"
    )


class MealItem(BaseModel):
    """A single meal with nutritional information."""

    meal_type: str = Field(
        ..., description="Type of meal (Breakfast, Lunch, Dinner, Snack)"
    )
    meal_name: str = Field(..., description="Descriptive name of the meal")
    description: str = Field(
        ..., description="Detailed description with ingredients and preparation notes"
    )
    calories: int = Field(..., description="Estimated calories")
    protein_g: float = Field(..., description="Protein in grams")
    carbs_g: float = Field(..., description="Carbohydrates in grams")
    fat_g: float = Field(..., description="Fat in grams")
    preparation_time_min: int = Field(
        ..., description="Estimated preparation time in minutes"
    )
    ingredients: List[str] = Field(..., description="List of ingredients")
    recipe_notes: Optional[str] = Field(
        default=None, description="Optional cooking instructions or tips"
    )


class DailyMealPlan(BaseModel):
    """Complete meal plan for a single day."""

    day_name: str = Field(..., description="Day of week")
    date: str = Field(..., description="ISO format date")
    meals: List[MealItem] = Field(..., description="All meals for the day")
    daily_totals: MacroTargets = Field(
        ..., description="Total macros/calories for the day"
    )
    notes: Optional[str] = Field(
        default=None, description="Any special notes for the day"
    )


class WeeklyMealPlan(BaseModel):
    """Complete meal plan for the entire week."""

    week_start_date: str = Field(..., description="ISO format start date")
    week_end_date: str = Field(..., description="ISO format end date")
    daily_plans: List[DailyMealPlan] = Field(
        ..., description="Meal plans for each day of the week"
    )
    shopping_list: List[str] = Field(
        ..., description="Consolidated shopping list for the week"
    )
    meal_prep_tips: List[str] = Field(
        ..., description="Batch cooking and meal prep recommendations"
    )


class NutritionalValidation(BaseModel):
    """Validation result from the nutritional validation agent."""

    approved: bool = Field(
        ..., description="Whether the meal plan meets nutritional standards"
    )
    validation_summary: str = Field(
        ..., description="Summary of the validation assessment"
    )
    macro_accuracy: Dict[str, str] = Field(
        ...,
        description="Assessment of macro accuracy per day (key=day_name, value=assessment)",
    )
    variety_score: str = Field(
        ..., description="Assessment of meal variety throughout the week"
    )
    practicality_score: str = Field(
        ..., description="Assessment of meal prep practicality"
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="List of issues found (if any)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement",
    )


class MealySyncStatus(BaseModel):
    """Status of a single meal sync to Mealy."""

    day_name: str = Field(..., description="Day of week")
    date: str = Field(..., description="ISO format date")
    meal_type: str = Field(..., description="Meal type (Breakfast, Lunch, etc.)")
    success: bool = Field(..., description="Whether sync was successful")
    mealy_id: Optional[str] = Field(
        default=None, description="Mealy ID if successfully created"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if sync failed"
    )


class MealyIntegrationResult(BaseModel):
    """Result of integrating the weekly meal plan into Mealy."""

    week_start_date: str = Field(..., description="ISO format start date")
    week_end_date: str = Field(..., description="ISO format end date")
    total_meals_created: int = Field(..., description="Total number of meals synced")
    sync_statuses: List[MealySyncStatus] = Field(
        ..., description="Detailed sync status for each meal"
    )
    mealy_week_url: Optional[str] = Field(
        default=None, description="URL to view the week in Mealy (if available)"
    )
    summary: str = Field(..., description="Human-readable summary of the integration")
