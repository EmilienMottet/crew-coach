"""Pydantic models for structured task outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


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
    """Soundtrack candidates selected from Spotify data."""

    original_description: str = Field(
        ...,
        description="The original activity description passed as input",
    )
    candidate_tracks: List[str] = Field(
        default_factory=list,
        description="Ordered list of '<artist> – <track>' candidates found in Spotify data",
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


class HexisMealTarget(BaseModel):
    """Nutritional target for a single meal (from Hexis mealRecommendation)."""

    meal_type: str = Field(
        ..., description="Type of meal: Breakfast, Lunch, PM Snack, Dinner"
    )
    time: str = Field(..., description="Meal time in HH:MM:SS.000Z format")
    carb_code: str = Field(..., description="Carb level guidance: LOW, MEDIUM, or HIGH")
    calories: int = Field(..., description="Target calories for this meal")
    protein_g: float = Field(..., description="Target protein in grams")
    carbs_g: float = Field(..., description="Target carbohydrates in grams")
    fat_g: float = Field(..., description="Target fat in grams")


class HexisDailyMealTargets(BaseModel):
    """Per-meal targets for a single day (from Hexis)."""

    date: str = Field(..., description="ISO format date (YYYY-MM-DD)")
    meals: List[HexisMealTarget] = Field(
        ..., description="List of meal targets for the day"
    )


class HexisWeeklyAnalysis(BaseModel):
    """Analysis output from Hexis training data."""

    model_config = {"extra": "allow"}  # Allow additional fields from LLM

    week_start_date: str = Field(..., description="ISO format start date (YYYY-MM-DD)")
    week_end_date: str = Field(..., description="ISO format end date (YYYY-MM-DD)")
    training_load_summary: str | Dict = Field(
        ...,
        description="Summary of weekly training load (TSS, hours, intensity) - can be string or object",
    )
    recovery_status: str | Dict = Field(
        ...,
        description="Current recovery status and recommendations - can be string or object",
    )
    daily_energy_needs: Dict = Field(
        ...,
        description="Calorie needs per day (key=date, value=calories or detailed object)",
    )
    daily_macro_targets: Dict = Field(
        ...,
        description="Macro targets per day (key=date, value=MacroTargets object)",
    )
    # Accept both simple strings and rich objects with priority/rationale/implementation
    nutritional_priorities: List[Union[str, Dict[str, Any]]] = Field(
        ...,
        description="Key nutritional priorities for the week - can be simple strings or objects with priority/rationale/implementation",
    )
    # Per-meal targets from Hexis (extracted from mealRecommendation)
    daily_meal_targets: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-meal targets from Hexis (key=date, value=HexisDailyMealTargets)",
    )


class DailyNutritionTarget(BaseModel):
    """Nutritional targets for a single day."""

    day_name: str = Field(..., description="Day of week (e.g., 'Monday')")
    date: str = Field(..., description="ISO format date (YYYY-MM-DD)")
    calories: int = Field(..., description="Total calorie target")
    macros: MacroTargets = Field(..., description="Macro targets for the day")
    training_context: str = Field(
        ...,
        description="Training context for the day (e.g., 'Hard interval session', 'Rest day')",
    )
    meal_timing_notes: str = Field(
        ...,
        description="Guidance on meal timing (e.g., 'High-carb breakfast pre-workout')",
    )
    # Per-meal targets from Hexis (with carbCode for food choice guidance)
    meal_targets: Optional[List[HexisMealTarget]] = Field(
        default=None,
        description="Per-meal targets from Hexis with exact calories, macros, and carbCode",
    )


class WeeklyNutritionPlan(BaseModel):
    """Structured weekly nutrition plan with daily targets."""

    model_config = {"extra": "allow"}

    week_start_date: str = Field(..., description="ISO format start date")
    week_end_date: str = Field(..., description="ISO format end date")
    daily_targets: List[DailyNutritionTarget] = Field(
        ..., description="Nutrition targets for each day of the week"
    )
    weekly_summary: str = Field(
        ..., description="High-level summary of the week's nutrition strategy"
    )


class ValidatedIngredient(BaseModel):
    """An ingredient validated against Passio food database."""

    name: str = Field(
        ..., description="Ingredient name as displayed (e.g., '120g chicken breast')"
    )
    passio_food_id: Optional[str] = Field(
        default=None, description="Passio Food ID from hexis_search_passio_foods"
    )
    passio_food_name: Optional[str] = Field(
        default=None, description="Exact food name as found in Passio database"
    )
    passio_ref_code: Optional[str] = Field(
        default=None,
        description="Base64-encoded refCode from hexis_search_passio_foods (REQUIRED for Hexis API)",
    )
    quantity_g: Optional[float] = Field(
        default=None, description="Quantity in grams for macro calculation"
    )
    adjusted_quantity_g: Optional[float] = Field(
        default=None,
        description="Python-adjusted quantity to match macro targets (overrides quantity_g if set)",
    )
    validation_status: Optional[str] = Field(
        default="found", description="Status: 'found', 'substituted', or 'not_found'"
    )
    substitution_note: Optional[str] = Field(
        default=None, description="Note if ingredient was substituted"
    )
    # Nutritional data from Passio (per 100g) - populated by hexis_get_passio_food_details
    protein_per_100g: Optional[float] = Field(
        default=None, description="Protein in grams per 100g from Passio"
    )
    carbs_per_100g: Optional[float] = Field(
        default=None, description="Carbohydrates in grams per 100g from Passio"
    )
    fat_per_100g: Optional[float] = Field(
        default=None, description="Fat in grams per 100g from Passio"
    )
    calories_per_100g: Optional[float] = Field(
        default=None, description="Calories per 100g from Passio"
    )


# ============================================================================
# Supervisor/Executor/Reviewer Pattern Schemas (Inter-agent communication)
# ============================================================================


# ----------------------------------------------------------------------------
# HEXIS_ANALYSIS Pattern: Data Supervisor → Data Executor → Analysis Reviewer
# ----------------------------------------------------------------------------


class HexisToolCall(BaseModel):
    """A planned Hexis tool call from the Supervisor."""

    tool_name: str = Field(
        ...,
        description="Name of the Hexis tool to call (e.g., 'hexis_get_weekly_plan')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the tool call (e.g., {'start_date': '2025-01-20', 'end_date': '2025-01-26'})",
    )
    purpose: str = Field(..., description="Why this data is needed for the analysis")
    priority: int = Field(default=1, description="Execution priority (1=highest)")


class HexisDataRetrievalPlan(BaseModel):
    """Output of HEXIS_DATA_SUPERVISOR → Input for HEXIS_DATA_EXECUTOR.

    Contains the data retrieval plan for the Executor to execute.
    """

    model_config = {"extra": "allow"}

    week_start_date: str = Field(..., description="ISO format start date (YYYY-MM-DD)")
    week_end_date: str = Field(..., description="ISO format end date (YYYY-MM-DD)")
    tool_calls: List[HexisToolCall] = Field(
        ..., description="Ordered list of Hexis tool calls to execute"
    )
    analysis_focus: List[str] = Field(
        ...,
        description="Key areas to focus on in the analysis (e.g., 'training_load', 'recovery', 'nutrition')",
    )
    special_considerations: Optional[str] = Field(
        default=None, description="Any special considerations for data retrieval"
    )


class HexisToolResult(BaseModel):
    """Result of a single Hexis tool execution."""

    tool_name: str = Field(..., description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether the tool call succeeded")
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw result from the tool (if successful)"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message (if failed)"
    )


class RawHexisData(BaseModel):
    """Output of HEXIS_DATA_EXECUTOR → Input for HEXIS_ANALYSIS_REVIEWER.

    Contains all raw data retrieved from Hexis tools.
    """

    model_config = {"extra": "allow"}

    week_start_date: str = Field(..., description="ISO format start date")
    week_end_date: str = Field(..., description="ISO format end date")
    tool_results: List[HexisToolResult] = Field(
        ..., description="Results from all tool calls"
    )
    total_calls: int = Field(..., description="Total number of tool calls made")
    successful_calls: int = Field(..., description="Number of successful calls")
    weekly_plan_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Extracted weekly plan data (the main data payload)"
    )
    retrieval_notes: Optional[str] = Field(
        default=None, description="Notes from the Executor about the data retrieval"
    )


# ----------------------------------------------------------------------------
# MEAL_GENERATION Pattern: Meal Supervisor → Ingredient Executor → Recipe Reviewer
# ----------------------------------------------------------------------------


class MealTemplate(BaseModel):
    """Template for a single meal from the Supervisor (before ingredient validation)."""

    meal_type: str = Field(
        ..., description="Type of meal (Breakfast, Lunch, Dinner, Afternoon Snack)"
    )
    meal_name: str = Field(..., description="Descriptive name of the meal")
    description: str = Field(
        ..., description="Detailed description with cooking technique and flavors"
    )
    ingredients_to_validate: List[str] = Field(
        ...,
        description="List of ingredients with quantities to validate (e.g., '120g chicken breast')",
    )
    estimated_calories: float = Field(
        ..., description="Estimated calories for the meal"
    )
    estimated_protein: float = Field(..., description="Estimated protein in grams")
    estimated_carbs: float = Field(..., description="Estimated carbohydrates in grams")
    estimated_fat: float = Field(..., description="Estimated fat in grams")
    preparation_time_min: int = Field(
        ..., description="Estimated preparation time in minutes"
    )
    recipe_notes: str = Field(..., description="Short, actionable cooking steps")


class MealPlanTemplate(BaseModel):
    """Output of Supervisor → Input for Executor. Contains meal plan before ingredient validation."""

    model_config = {"extra": "allow"}

    day_name: str = Field(..., description="Day of week (e.g., 'Monday')")
    date: str = Field(..., description="ISO format date (YYYY-MM-DD)")
    meals: List[MealTemplate] = Field(..., description="List of 4 meals for the day")
    target_calories: int = Field(..., description="Target calories for the day")
    target_protein: float = Field(..., description="Target protein in grams")
    target_carbs: float = Field(..., description="Target carbs in grams")
    target_fat: float = Field(..., description="Target fat in grams")
    training_context: str = Field(
        ..., description="Training context for timing optimization"
    )


class ValidatedMealIngredients(BaseModel):
    """Validated ingredients for a single meal from the Executor."""

    meal_type: str = Field(
        ..., description="Type of meal (matches MealTemplate.meal_type)"
    )
    meal_name: str = Field(
        ..., description="Name of the meal (matches MealTemplate.meal_name)"
    )
    validated_ingredients: List[ValidatedIngredient] = Field(
        ..., description="List of validated ingredients with Passio IDs"
    )
    validation_success: bool = Field(
        default=True, description="Whether all critical ingredients were validated"
    )
    validation_notes: Optional[str] = Field(
        default=None, description="Notes about substitutions or missing ingredients"
    )


class ValidatedIngredientsList(BaseModel):
    """Output of Executor → Input for Reviewer. Contains all validated ingredients."""

    model_config = {"extra": "allow"}

    day_name: str = Field(..., description="Day of week")
    date: str = Field(..., description="ISO format date")
    validated_meals: List[ValidatedMealIngredients] = Field(
        ..., description="Validated ingredients for each meal"
    )
    total_validations: int = Field(
        ..., description="Total number of ingredients validated"
    )
    successful_validations: int = Field(
        ..., description="Number of successful validations"
    )
    substitutions_made: int = Field(
        default=0, description="Number of ingredient substitutions"
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
    validated_ingredients: Optional[List[ValidatedIngredient]] = Field(
        default=None,
        description="Ingredients validated via hexis_search_passio_foods with Passio IDs for Hexis integration",
    )
    recipe_notes: Optional[str] = Field(
        default=None, description="Optional cooking instructions or tips"
    )
    hexis_food_id: Optional[str] = Field(
        default=None, description="Hexis Food ID if sourced from Hexis database"
    )
    data_origin: Optional[str] = Field(
        default=None,
        description="Data origin (e.g., 'HEXIS_DATABASE') if sourced from Hexis",
    )

    @field_validator("calories", mode="before")
    @classmethod
    def round_calories(cls, v):
        """Round float calories to nearest integer."""
        if v is None:
            return None
        return round(float(v))

    @field_validator("recipe_notes", mode="before")
    @classmethod
    def normalize_recipe_notes(cls, v):
        """Convert recipe_notes from list to string if needed."""
        if v is None:
            return None
        if isinstance(v, list):
            # Join list items with newlines for better readability
            return "\n".join(str(item) for item in v if item)
        return str(v) if v else None


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

    model_config = {"extra": "allow"}

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

    model_config = {"extra": "allow"}

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


class HexisSyncStatus(BaseModel):
    """Status of a single meal sync to Hexis."""

    day_name: str = Field(..., description="Day of week")
    date: str = Field(..., description="ISO format date")
    meal_type: str = Field(..., description="Meal type (Breakfast, Lunch, etc.)")
    success: bool = Field(..., description="Whether sync was successful")
    hexis_id: Optional[str] = Field(
        default=None, description="Hexis ID if successfully created"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if sync failed"
    )


# Keep old name for backwards compatibility
MealySyncStatus = HexisSyncStatus


class HexisIntegrationResult(BaseModel):
    """Result of integrating the weekly meal plan into Hexis."""

    model_config = {"extra": "allow"}

    week_start_date: str = Field(..., description="ISO format start date")
    week_end_date: str = Field(..., description="ISO format end date")
    total_meals_created: int = Field(..., description="Total number of meals synced")
    sync_statuses: List[HexisSyncStatus] = Field(
        ..., description="Detailed sync status for each meal"
    )
    summary: str = Field(..., description="Human-readable summary of the integration")


# Keep old name for backwards compatibility
MealyIntegrationResult = HexisIntegrationResult


class LyricsVerificationResult(BaseModel):
    """Result of lyrics verification and quote selection."""

    final_description: str = Field(
        ...,
        description="The complete updated description text with quote and music",
    )
    accepted_tracks: List[str] = Field(
        default_factory=list,
        description="List of tracks that passed verification",
    )
    rejected_tracks: List[str] = Field(
        default_factory=list,
        description="List of tracks rejected due to content",
    )
    selected_quote: str = Field(
        ...,
        description="The selected quote text",
    )
    quote_source: str = Field(
        ...,
        description="Source of the quote",
    )


# ----------------------------------------------------------------------------
# DESCRIPTION Pattern: Description Supervisor → Data Retrieval Executor → Description Reviewer
# Separates reasoning from tool execution for Strava activity description generation
# ----------------------------------------------------------------------------


class ActivityToolCall(BaseModel):
    """A planned tool call for activity data retrieval."""

    tool_name: str = Field(
        ...,
        description="Name of the tool to call (e.g., 'IntervalsIcu__get_activities')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the tool call"
    )
    purpose: str = Field(
        ..., description="Why this data is needed for description generation"
    )
    priority: int = Field(default=1, description="Execution priority (1=highest)")


class ActivityDataRetrievalPlan(BaseModel):
    """Output of DESCRIPTION_SUPERVISOR → Input for DATA_RETRIEVAL_EXECUTOR.

    Contains the data retrieval plan for fetching activity data from Intervals.icu.
    """

    model_config = {"extra": "allow"}

    activity_id: str = Field(..., description="Strava activity ID")
    activity_date: str = Field(
        ..., description="Activity date in ISO format (YYYY-MM-DD)"
    )
    activity_type: str = Field(..., description="Type of activity (Run, Ride, etc.)")
    tool_calls: List[ActivityToolCall] = Field(
        ..., description="Ordered list of tool calls to execute"
    )
    data_focus: List[str] = Field(
        ...,
        description="Key data points to focus on (e.g., 'pace', 'heart_rate', 'power', 'core_temperature')",
    )
    description_style: str = Field(
        default="engaging",
        description="Style guidance for the final description (e.g., 'technical', 'motivational', 'engaging')",
    )


class ActivityToolResult(BaseModel):
    """Result of a single activity data tool execution."""

    tool_name: str = Field(..., description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether the tool call succeeded")
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw result from the tool (if successful)"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message (if failed)"
    )


class RawActivityData(BaseModel):
    """Output of DATA_RETRIEVAL_EXECUTOR → Input for DESCRIPTION_REVIEWER.

    Contains all raw activity data retrieved from Intervals.icu tools.
    """

    model_config = {"extra": "allow"}

    activity_id: str = Field(..., description="Strava activity ID")
    activity_date: str = Field(..., description="Activity date")
    activity_type: str = Field(..., description="Type of activity")
    tool_results: List[ActivityToolResult] = Field(
        ..., description="Results from all tool calls"
    )
    total_calls: int = Field(..., description="Total number of tool calls made")
    successful_calls: int = Field(..., description="Number of successful calls")
    intervals_activity_id: Optional[str] = Field(
        default=None, description="Intervals.icu activity ID (if found)"
    )
    activity_details: Optional[Dict[str, Any]] = Field(
        default=None, description="Full activity details from Intervals.icu"
    )
    activity_streams: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Activity streams data (heart_rate, power, core_temperature, etc.)",
    )
    retrieval_notes: Optional[str] = Field(
        default=None, description="Notes from the Executor about data retrieval"
    )
