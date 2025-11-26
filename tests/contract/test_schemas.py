"""Contract tests for input/output schema validation."""
import pytest
from schemas import (
    GeneratedActivityContent,
    PrivacyAssessment,
    TranslationPayload,
    ActivityMusicSelection,
    PrivacyRecommendations
)


@pytest.mark.priority_medium
@pytest.mark.contract
class TestSchemaValidation:
    """Validate that outputs conform to Pydantic schemas."""

    @pytest.mark.timeout(120)
    def test_output_conforms_to_contract(self, crew_instance, sample_activity_outside_work):
        """Final output should match expected contract."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        # Required fields
        required_fields = [
            "activity_id",
            "title",
            "description",
            "should_be_private",
            "privacy_check"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Validate types
        assert isinstance(result["activity_id"], int), "activity_id should be int"
        assert isinstance(result["title"], str), "title should be str"
        assert isinstance(result["description"], str), "description should be str"
        assert isinstance(result["should_be_private"], bool), "should_be_private should be bool"
        assert isinstance(result["privacy_check"], dict), "privacy_check should be dict"

    @pytest.mark.timeout(120)
    def test_privacy_assessment_schema(self, crew_instance, sample_activity_work_hours):
        """Privacy check output should validate against PrivacyAssessment."""
        result = crew_instance.process_activity(sample_activity_work_hours)

        privacy_data = result["privacy_check"]

        # Should be valid PrivacyAssessment
        assessment = PrivacyAssessment(
            privacy_approved=privacy_data["approved"],
            during_work_hours=privacy_data["during_work_hours"],
            should_be_private=result["should_be_private"],
            issues_found=privacy_data.get("issues", []),
            reasoning=privacy_data["reasoning"]
        )

        assert assessment is not None
        assert assessment.should_be_private is True
        assert isinstance(assessment.reasoning, str)
        assert len(assessment.reasoning) > 0

    @pytest.mark.timeout(120)
    def test_character_limits_enforced(self, crew_instance, sample_activity_outside_work):
        """Strava limits (≤50 title, ≤500 description) must be enforced."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        from tests.conftest import assert_character_limits
        assert_character_limits(result["title"], result["description"])

    @pytest.mark.timeout(120)
    def test_privacy_check_structure(self, crew_instance, sample_activity_outside_work):
        """Privacy check should have expected structure."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        privacy_check = result["privacy_check"]

        # Required fields in privacy_check
        assert "approved" in privacy_check, "Missing 'approved' field"
        assert "during_work_hours" in privacy_check, "Missing 'during_work_hours' field"
        assert "reasoning" in privacy_check, "Missing 'reasoning' field"

        # Validate types
        assert isinstance(privacy_check["approved"], bool)
        assert isinstance(privacy_check["during_work_hours"], bool)
        assert isinstance(privacy_check["reasoning"], str)
        assert len(privacy_check["reasoning"]) > 0

    def test_privacy_recommendations_schema(self):
        """Test PrivacyRecommendations schema."""
        # Valid with no recommendations
        rec1 = PrivacyRecommendations()
        assert rec1.title is None
        assert rec1.description is None

        # Valid with recommendations
        rec2 = PrivacyRecommendations(
            title="Sanitized Title",
            description="Sanitized Description"
        )
        assert rec2.title == "Sanitized Title"
        assert rec2.description == "Sanitized Description"

    def test_generated_activity_content_schema(self):
        """Test GeneratedActivityContent schema validation."""
        # Valid content
        content = GeneratedActivityContent(
            title="Test Activity",
            description="A great workout session",
            workout_type="Tempo Run",
            key_metrics={"pace": "4:30/km", "hr": "145 bpm"}
        )

        assert content.title == "Test Activity"
        assert content.workout_type == "Tempo Run"
        assert "pace" in content.key_metrics

    def test_activity_music_selection_schema(self):
        """Test ActivityMusicSelection schema validation."""
        music = ActivityMusicSelection(
            original_description="Original description",
            candidate_tracks=["Artist – Track 1", "Artist – Track 2"]
        )

        assert music.original_description == "Original description"
        assert len(music.candidate_tracks) == 2
