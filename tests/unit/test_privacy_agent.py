"""Unit tests for Privacy Agent (work hours detection, PII validation)."""
import pytest
from schemas import PrivacyAssessment


@pytest.mark.priority_high
@pytest.mark.unit
class TestPrivacyAgent:
    """Test privacy detection and work hours compliance."""

    @pytest.mark.timeout(120)
    def test_work_hours_morning_slot(self, crew_instance, sample_activity_work_hours):
        """Activity at 11:00 CET (work hours) should be private."""
        result = crew_instance.process_activity(sample_activity_work_hours)

        assert result["should_be_private"] is True, \
            f"Activity during work hours should be private. Got: {result}"
        assert result["privacy_check"]["during_work_hours"] is True, \
            "Privacy check should detect work hours"
        reasoning_lower = result["privacy_check"]["reasoning"].lower()
        assert "work" in reasoning_lower or "heures" in reasoning_lower, \
            f"Reasoning should mention work hours. Got: {result['privacy_check']['reasoning']}"

    @pytest.mark.timeout(120)
    def test_outside_work_hours_evening(self, crew_instance, sample_activity_outside_work):
        """Activity at 18:30 CET (outside work) should be public."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        assert result["should_be_private"] is False, \
            f"Activity outside work hours should be public. Got: {result}"
        assert result["privacy_check"]["during_work_hours"] is False, \
            "Privacy check should detect outside work hours"

    @pytest.mark.timeout(120)
    def test_work_hours_boundary_08_29(self, crew_instance):
        """Activity at 08:29 CET (1 min before work) should be public."""
        from tests.fixtures.activities import ACTIVITIES
        activity = ACTIVITIES["boundary_08_29"]

        result = crew_instance.process_activity(activity)
        assert result["should_be_private"] is False, \
            "Activity at 08:29 CET (1 min before work start) should be public"

    @pytest.mark.timeout(120)
    def test_work_hours_boundary_08_30(self, crew_instance):
        """Activity at 08:30 CET (exactly start of work) should be private."""
        from tests.fixtures.activities import ACTIVITIES
        activity = ACTIVITIES["boundary_08_30"]

        result = crew_instance.process_activity(activity)
        assert result["should_be_private"] is True, \
            "Activity at 08:30 CET (exactly work start) should be private"

    @pytest.mark.timeout(120)
    def test_weekend_activity_public(self, crew_instance):
        """Sunday morning activity should be public."""
        from tests.fixtures.activities import ACTIVITIES
        activity = ACTIVITIES["weekend_long_run"]

        result = crew_instance.process_activity(activity)
        assert result["should_be_private"] is False, \
            "Weekend activities should be public"

    @pytest.mark.timeout(120)
    def test_privacy_schema_validation(self, crew_instance, sample_activity_work_hours):
        """Privacy check result should match PrivacyAssessment schema."""
        result = crew_instance.process_activity(sample_activity_work_hours)

        # Validate schema compliance
        privacy_data = result["privacy_check"]
        assessment = PrivacyAssessment(
            privacy_approved=privacy_data["approved"],
            during_work_hours=privacy_data["during_work_hours"],
            should_be_private=result["should_be_private"],
            issues_found=privacy_data.get("issues", []),
            reasoning=privacy_data["reasoning"]
        )
        assert assessment is not None
        assert assessment.should_be_private is True
