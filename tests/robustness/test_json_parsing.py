"""Robustness tests for JSON parsing edge cases."""
import pytest
import json


@pytest.mark.priority_high
@pytest.mark.robustness
class TestJSONParsing:
    """Test robust JSON parsing from LLM responses."""

    @pytest.mark.timeout(120)
    @pytest.mark.mcp_required
    def test_standard_json_output(self, crew_instance, sample_activity_outside_work):
        """Standard case: clean JSON output."""
        result = crew_instance.process_activity(sample_activity_outside_work)

        # Should be valid JSON-serializable
        json_str = json.dumps(result)
        assert json_str is not None
        assert "title" in result
        assert "description" in result

    @pytest.mark.timeout(120)
    @pytest.mark.mcp_required
    def test_activity_with_null_fields(self, crew_instance):
        """Handle activity with null/missing fields gracefully."""
        from tests.fixtures.activities import ACTIVITIES
        activity = ACTIVITIES["null_fields_activity"]

        # Should not crash
        result = crew_instance.process_activity(activity)
        assert "title" in result, "Result should have title even with null fields"
        assert "description" in result, "Result should have description even with null fields"
        assert len(result["title"]) > 0, "Title should not be empty"
        assert len(result["description"]) > 0, "Description should not be empty"

    @pytest.mark.timeout(30)
    @pytest.mark.mcp_required
    def test_malformed_input_error_handling(self, crew_instance):
        """Malformed input should return error dict or handle gracefully."""
        malformed_input = {"invalid": "structure"}

        result = crew_instance.process_activity(malformed_input)

        # Should return error response or valid fallback
        assert "error" in result or "title" in result, \
            "Should return error dict or valid result"

    def test_pydantic_schema_detection(self):
        """Test detection of Pydantic schema in LLM response."""
        from crew import StravaDescriptionCrew

        # Valid schema response (has "type" and "properties" keys)
        schema_response = {
            "title": {"type": "string", "description": "Activity title"},
            "properties": {"field": "string"}
        }
        assert StravaDescriptionCrew._is_pydantic_schema(schema_response) is True

        # Valid data response (no "type" or "properties")
        data_response = {
            "title": "My Activity",
            "description": "A nice run"
        }
        assert StravaDescriptionCrew._is_pydantic_schema(data_response) is False

    def test_json_extraction_from_markdown(self):
        """Test extraction of JSON from markdown-wrapped responses."""
        from crew import StravaDescriptionCrew

        # Markdown-wrapped JSON
        markdown_json = '''```json
        {"title": "Test", "description": "Test desc"}
        ```'''

        result = StravaDescriptionCrew._extract_json_from_text(markdown_json)
        assert result is not None
        assert result.get("title") == "Test"
        assert result.get("description") == "Test desc"

    def test_json_extraction_with_backticks(self):
        """Test extraction with various markdown formats."""
        from crew import StravaDescriptionCrew

        # Just backticks without "json" tag
        simple_markdown = '''```
        {"title": "Test 2", "description": "Another test"}
        ```'''

        result = StravaDescriptionCrew._extract_json_from_text(simple_markdown)
        assert result is not None
        assert result.get("title") == "Test 2"

    def test_json_unnesting(self):
        """Test recursive JSON string unnesting."""
        from crew import StravaDescriptionCrew

        # Nested JSON strings
        nested = {
            "title": "Test",
            "description": '{"nested": "value"}',
            "normal": "plain value"
        }

        result = StravaDescriptionCrew._unnest_json_strings(nested)
        assert isinstance(result["description"], dict), "Should unnest JSON string"
        assert result["description"]["nested"] == "value"
        assert result["normal"] == "plain value", "Should preserve normal strings"

    def test_deeply_nested_json_strings(self):
        """Test deeply nested JSON strings."""
        from crew import StravaDescriptionCrew

        deeply_nested = {
            "level1": '{"level2": "{\\"level3\\": \\"value\\"}"}'
        }

        result = StravaDescriptionCrew._unnest_json_strings(deeply_nested)

        # Should unnest at least one level
        assert isinstance(result["level1"], dict), "Should unnest first level"
        assert "level2" in result["level1"]
