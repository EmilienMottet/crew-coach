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
