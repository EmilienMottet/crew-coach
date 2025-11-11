"""Test the MCP tool wrapper with various malformed inputs."""

from mcp_tool_wrapper import validate_tool_input
import sys


def test_valid_dict():
    """Test with a valid dictionary input."""
    print("\n🧪 Test 1: Valid dictionary input")
    input_data = {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 10}
    result = validate_tool_input(input_data)
    assert result == input_data
    print("✅ PASSED: Valid dict returned as-is")


def test_list_with_params():
    """Test with a list containing parameter dicts."""
    print("\n🧪 Test 2: List with parameter dicts (malformed)")
    input_data = [
        {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 50},
        {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 50},
    ]
    result = validate_tool_input(input_data)
    assert result == {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 50}
    print("✅ PASSED: First parameter dict extracted from list")


def test_list_with_mocked_response():
    """Test with a list containing params and mocked response."""
    print("\n🧪 Test 3: List with mocked response (malformed)")
    input_data = [
        {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 50},
        {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 50},
        {"status": "success", "data": [{"id": "A-123"}]},  # Mocked response
    ]
    result = validate_tool_input(input_data)
    assert result == {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 50}
    print("✅ PASSED: Parameter dict extracted, mocked response ignored")


def test_mocked_response_only():
    """Test with a mocked response dict (should fail)."""
    print("\n🧪 Test 4: Mocked response dict only (should fail)")
    input_data = {"status": "success", "data": [{"id": "A-123"}]}
    try:
        result = validate_tool_input(input_data)
        print("❌ FAILED: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"✅ PASSED: Correctly rejected mocked response: {e}")


def test_json_string():
    """Test with a JSON string input."""
    print("\n🧪 Test 5: JSON string input")
    input_data = '{"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 10}'
    result = validate_tool_input(input_data)
    assert result == {"start_date": "2025-11-11", "end_date": "2025-11-11", "limit": 10}
    print("✅ PASSED: JSON string parsed correctly")


def test_invalid_type():
    """Test with an invalid type (should fail)."""
    print("\n🧪 Test 6: Invalid type (should fail)")
    input_data = 12345
    try:
        result = validate_tool_input(input_data)
        print("❌ FAILED: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"✅ PASSED: Correctly rejected invalid type: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing MCP Tool Input Validation")
    print("=" * 60)

    try:
        test_valid_dict()
        test_list_with_params()
        test_list_with_mocked_response()
        test_mocked_response_only()
        test_json_string()
        test_invalid_type()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
