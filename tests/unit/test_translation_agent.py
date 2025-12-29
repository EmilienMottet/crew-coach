"""Unit tests for Translation Agent (French translation)."""

import pytest
import re
from tests.conftest import assert_french_text, assert_character_limits, assert_has_emoji


@pytest.mark.priority_high
@pytest.mark.unit
@pytest.mark.mcp_required
class TestTranslationAgent:
    """Test French translation accuracy and constraints."""

    @pytest.mark.timeout(120)
    def test_title_translated_to_french(
        self, crew_instance, sample_activity_outside_work
    ):
        """Title should be in French."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        assert_french_text(result["title"], "Title")

    @pytest.mark.timeout(120)
    def test_description_translated_to_french(
        self, crew_instance, sample_activity_outside_work
    ):
        """Description should be in French."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        assert_french_text(result["description"], "Description")

    @pytest.mark.timeout(120)
    def test_character_limits_respected(
        self, crew_instance, sample_activity_outside_work
    ):
        """Title ≤50 chars, description ≤500 chars."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        assert_character_limits(result["title"], result["description"])

    @pytest.mark.timeout(120)
    def test_emoji_preserved_in_translation(
        self, crew_instance, sample_activity_outside_work
    ):
        """Emojis should be preserved after translation."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        # Title or description should contain emoji
        combined = result["title"] + " " + result["description"]
        assert_has_emoji(combined)

    @pytest.mark.timeout(120)
    def test_metrics_preserved_format(
        self, crew_instance, sample_activity_outside_work
    ):
        """Metric formats (pace, HR, power) should be preserved."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        # Check for metric patterns (flexible regex)
        desc = result["description"]

        # Should contain at least one metric pattern
        metric_patterns = [
            r"\d+:\d+\s*/km",  # Pace: 4:30 /km
            r"\d+\s*bpm",  # Heart rate: 145 bpm
            r"\d+\s*W",  # Power: 280 W
            r"\d+[,\.]?\d*\s*km",  # Distance: 50 km or 50.5 km
        ]

        has_metric = any(
            re.search(pattern, desc, re.IGNORECASE) for pattern in metric_patterns
        )
        assert has_metric, f"No metrics found in description: {desc}"

    @pytest.mark.timeout(120)
    def test_run_activity_french_terminology(self, crew_instance):
        """Run activities should use French running terminology."""
        from tests.fixtures.activities import ACTIVITIES

        activity = ACTIVITIES["weekend_long_run"]

        result = crew_instance.process_activity(activity)

        # Check for French running terms
        combined = result["title"] + " " + result["description"]
        combined_lower = combined.lower()

        # Should have running-related French terms
        running_terms = [
            "course",
            "sortie",
            "entraînement",
            "allure",
            "tempo",
            "foulée",
        ]
        has_running_term = any(term in combined_lower for term in running_terms)

        assert (
            has_running_term
        ), f"Expected French running terminology in: {result['title']} | {result['description'][:100]}"

    @pytest.mark.timeout(120)
    def test_ride_activity_french_terminology(self, crew_instance):
        """Ride activities should use French cycling terminology."""
        from tests.fixtures.activities import ACTIVITIES

        activity = ACTIVITIES["evening_ride"]

        result = crew_instance.process_activity(activity)

        # Check for French cycling terms
        combined = result["title"] + " " + result["description"]
        combined_lower = combined.lower()

        # Should have cycling-related French terms
        cycling_terms = ["vélo", "sortie", "puissance", "watts", "allure", "pédalage"]
        has_cycling_term = any(term in combined_lower for term in cycling_terms)

        assert (
            has_cycling_term
        ), f"Expected French cycling terminology in: {result['title']} | {result['description'][:100]}"
