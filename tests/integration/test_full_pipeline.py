"""Integration tests for the complete 4-agent pipeline."""
import pytest


@pytest.mark.priority_medium
@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Test the complete Description → Music → Privacy → Translation workflow."""

    @pytest.mark.timeout(180)
    def test_complete_workflow_outside_work_hours(self, crew_instance, sample_activity_outside_work):
        """Full workflow for public activity."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        # Validate complete output structure
        assert "activity_id" in result, "Missing activity_id in result"
        assert result["activity_id"] == 87654321
        assert "title" in result, "Missing title in result"
        assert "description" in result, "Missing description in result"
        assert "should_be_private" in result, "Missing should_be_private in result"
        assert "privacy_check" in result, "Missing privacy_check in result"

        # Validate French translation
        from tests.conftest import assert_french_text, assert_character_limits
        assert_french_text(result["title"], "Title")
        assert_french_text(result["description"], "Description")
        assert_character_limits(result["title"], result["description"])

        # Validate privacy decision
        assert result["should_be_private"] is False, \
            "Evening activity should be public"

    @pytest.mark.timeout(180)
    def test_complete_workflow_work_hours(self, crew_instance, sample_activity_work_hours):
        """Full workflow for private activity (work hours)."""
        result = crew_instance.process_activity(sample_activity_work_hours)

        # Should be marked private
        assert result["should_be_private"] is True, \
            "Work hours activity should be private"
        assert result["privacy_check"]["during_work_hours"] is True, \
            "Privacy check should detect work hours"

        # Still should have French content
        from tests.conftest import assert_french_text
        assert_french_text(result["title"], "Title")

    @pytest.mark.timeout(180)
    def test_music_integration_with_spotify_data(self, crew_instance):
        """Music agent should process Spotify data when available."""
        from tests.fixtures.activities import ACTIVITIES
        activity = ACTIVITIES["evening_ride"]

        result = crew_instance.process_activity(activity)

        # Description might mention music (not guaranteed but likely)
        # Just verify it doesn't crash with Spotify data
        assert "title" in result
        assert "description" in result
        assert len(result["description"]) > 0

    @pytest.mark.timeout(240)
    def test_end_to_end_run_vs_ride(self, crew_instance):
        """Test pipeline handles both Run and Ride activities."""
        from tests.fixtures.activities import ACTIVITIES

        run_result = crew_instance.process_activity(ACTIVITIES["weekend_long_run"])
        ride_result = crew_instance.process_activity(ACTIVITIES["evening_ride"])

        # Both should succeed
        assert "title" in run_result, "Run activity should have title"
        assert "title" in ride_result, "Ride activity should have title"

        # Validate both are in French
        from tests.conftest import assert_french_text
        assert_french_text(run_result["title"], "Run title")
        assert_french_text(ride_result["title"], "Ride title")

        # Titles should differ (different activity types)
        assert run_result["title"] != ride_result["title"], \
            "Different activity types should have different titles"

    @pytest.mark.timeout(180)
    def test_output_json_serializability(self, crew_instance, sample_activity_outside_work):
        """Final output should be JSON-serializable."""
        import json

        result = crew_instance.process_activity(sample_activity_outside_work)

        # Should be able to serialize to JSON
        json_str = json.dumps(result)
        assert json_str is not None

        # Should be able to deserialize back
        deserialized = json.loads(json_str)
        assert deserialized["activity_id"] == result["activity_id"]
        assert deserialized["title"] == result["title"]
