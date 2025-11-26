"""Robustness tests for MCP tool failures."""
import pytest


@pytest.mark.priority_medium
@pytest.mark.robustness
class TestMCPFailures:
    """Test graceful degradation when MCP tools fail."""

    @pytest.mark.timeout(120)
    def test_activity_with_minimal_data(self, crew_instance):
        """Process activity with minimal required data."""
        minimal_activity = {
            "object_type": "activity",
            "object_id": 8888,
            "object_data": {
                "id": 8888,
                "name": "Test Run",
                "type": "Run",
                "distance": 10000,
                "moving_time": 3000,
                "start_date_local": "2025-11-26T18:00:00Z"
            },
            "spotify_recently_played": {"items": []}
        }

        result = crew_instance.process_activity(minimal_activity)

        # Should still generate valid output
        assert "title" in result, "Should generate title with minimal data"
        assert "description" in result, "Should generate description with minimal data"
        assert len(result["title"]) > 0
        assert len(result["description"]) > 0

    @pytest.mark.timeout(120)
    def test_activity_without_spotify_data(self, crew_instance, sample_activity_work_hours):
        """Activity without Spotify data should process successfully."""
        # sample_activity_work_hours has empty Spotify data
        result = crew_instance.process_activity(sample_activity_work_hours)

        # Should succeed without Spotify data
        assert "title" in result
        assert "description" in result
        assert result["should_be_private"] is True

    @pytest.mark.timeout(120)
    def test_activity_with_incomplete_metrics(self, crew_instance):
        """Activity with some missing metrics should still process."""
        incomplete_activity = {
            "object_type": "activity",
            "object_id": 7777,
            "object_data": {
                "id": 7777,
                "name": "Incomplete Activity",
                "type": "Ride",
                "distance": 45000,
                "moving_time": 4800,
                "start_date_local": "2025-11-26T17:00:00Z",
                # Missing heart rate and power
            },
            "spotify_recently_played": {"items": []}
        }

        result = crew_instance.process_activity(incomplete_activity)

        # Should handle missing metrics gracefully
        assert "title" in result
        assert "description" in result
        assert result["should_be_private"] is False  # Outside work hours
