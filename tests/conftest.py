"""Shared test fixtures for Strava Description Crew tests."""
import os
import pytest
from datetime import datetime, timezone
from crew import StravaDescriptionCrew


def pytest_collection_modifyitems(config, items):
    """Skip MCP-required tests in CI environment."""
    is_ci = (
        os.getenv("CI", "false").lower() == "true"
        or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
    )

    if is_ci:
        skip_mcp = pytest.mark.skip(reason="MCP servers not available in CI")
        for item in items:
            if "mcp_required" in item.keywords:
                item.add_marker(skip_mcp)


@pytest.fixture(scope="session")
def test_env_vars():
    """Ensure required environment variables are set for tests."""
    required_vars = [
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "MCP_API_KEY",
        "STRAVA_MCP_SERVER_URL",
        "INTERVALS_MCP_SERVER_URL"
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        pytest.skip(f"Missing required env vars: {', '.join(missing)}")


@pytest.fixture(scope="function")
def crew_instance(test_env_vars):
    """Create a fresh StravaDescriptionCrew instance for each test."""
    crew = StravaDescriptionCrew()
    yield crew
    # Cleanup MCP connections
    for adapter in crew.mcp_adapters:
        adapter.stop()


@pytest.fixture
def sample_activity_work_hours():
    """Activity during work hours (should be private)."""
    return {
        "object_type": "activity",
        "object_id": 12345678,
        "object_data": {
            "id": 12345678,
            "name": "Lunch Run",
            "type": "Run",
            "distance": 10000,
            "moving_time": 2400,
            "start_date_local": "2025-11-26T11:00:00Z",  # 11:00 CET = work hours
            "average_heartrate": 145,
            "average_watts": None
        },
        "spotify_recently_played": {"items": []}
    }


@pytest.fixture
def sample_activity_outside_work():
    """Activity outside work hours (should be public)."""
    return {
        "object_type": "activity",
        "object_id": 87654321,
        "object_data": {
            "id": 87654321,
            "name": "Evening Ride",
            "type": "Ride",
            "distance": 50000,
            "moving_time": 5400,
            "start_date_local": "2025-11-26T18:30:00Z",  # 18:30 CET = outside work
            "average_heartrate": 150,
            "average_watts": 280
        },
        "spotify_recently_played": {"items": []}
    }


def assert_french_text(text: str, field_name: str):
    """Validate that text is in French (heuristic-based)."""
    # French indicators
    french_keywords = ["sortie", "entraînement", "séance", "allure", "vitesse", "vélo", "course"]
    french_chars = ["é", "è", "ê", "à", "ù", "ç", "œ"]

    text_lower = text.lower()
    has_keyword = any(kw in text_lower for kw in french_keywords)
    has_accent = any(char in text for char in french_chars)

    assert has_keyword or has_accent, \
        f"{field_name} should be in French but got: {text[:100]}"


def assert_character_limits(title: str, description: str):
    """Validate Strava character limits."""
    assert len(title) <= 50, f"Title exceeds 50 chars: {len(title)} chars"
    assert len(description) <= 500, f"Description exceeds 500 chars: {len(description)} chars"


def assert_has_emoji(text: str):
    """Check if text contains emoji (heuristic)."""
    # Simple check for common emoji ranges
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc symbols
        (0x1F680, 0x1F6FF),  # Transport
        (0x2600, 0x26FF),    # Misc symbols
    ]
    has_emoji = any(
        any(start <= ord(char) <= end for start, end in emoji_ranges)
        for char in text
    )
    assert has_emoji, f"Expected emoji in text: {text[:50]}"
