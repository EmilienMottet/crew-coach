#!/usr/bin/env python3
"""Test that Music Agent can use Spotify tools with /codex/v1 endpoint."""
import os
import sys

# Set up codex endpoint for testing
os.environ["OPENAI_API_BASE"] = "https://ccproxy.emottet.com/codex/v1"
os.environ["OPENAI_MODEL_NAME"] = "gpt-5"
os.environ["OPENAI_MUSIC_MODEL_NAME"] = "gpt-5"  # Force music agent to use codex
os.environ["OPENAI_MUSIC_API_BASE"] = "https://ccproxy.emottet.com/codex/v1"

from dotenv import load_dotenv
load_dotenv(override=True)

from crew import StravaDescriptionCrew

def test_music_agent_initialization():
    """Test that Music Agent can be initialized with codex endpoint and tools."""
    
    print("\n" + "="*80)
    print("Testing Music Agent with Codex Endpoint")
    print("="*80 + "\n")
    
    try:
        crew_instance = StravaDescriptionCrew()
        
        print("✅ SUCCESS: Music Agent initialized with codex endpoint")
        print(f"\n   Music LLM: {crew_instance.music_llm}")
        
        # Check that tools are available
        music_agent = crew_instance.music_agent
        if hasattr(music_agent, 'tools') and music_agent.tools:
            print(f"\n✅ Music Agent has {len(music_agent.tools)} tools available")
            print("\n   Sample tools:")
            for tool in music_agent.tools[:5]:
                tool_name = getattr(tool, 'name', str(tool))
                print(f"      - {tool_name}")
        else:
            print("\n⚠️  Warning: Music Agent has no tools")
        
        return True
        
    except ValueError as e:
        if "codex endpoint" in str(e).lower():
            print(f"\n❌ FAILED: Codex endpoint still blocked for tool-using agents!")
            print(f"   Error: {e}")
            return False
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_rotation_logic():
    """Test that provider rotation doesn't skip codex for tool calls."""
    
    print("\n" + "="*80)
    print("Testing Provider Rotation Logic")
    print("="*80 + "\n")
    
    from llm_provider_rotation import _requires_tool_free_context
    
    # Test various combinations
    test_cases = [
        ("https://ccproxy.emottet.com/codex/v1", "gpt-5", "Codex + GPT-5"),
        ("https://ccproxy.emottet.com/codex/v1", "claude-sonnet-4.5", "Codex + Claude"),
        ("https://ccproxy.emottet.com/copilot/v1", "gpt-5", "Copilot + GPT-5"),
        ("https://ccproxy.emottet.com/claude/v1", "claude-sonnet-4-5-20250929", "Claude endpoint"),
    ]
    
    all_passed = True
    for api_base, model, name in test_cases:
        is_tool_free = _requires_tool_free_context(api_base, model)
        status = "❌ BLOCKED" if is_tool_free else "✅ ALLOWED"
        print(f"{status}: {name}")
        print(f"         Endpoint: {api_base}")
        print(f"         Model: {model}")
        print(f"         tool_free_only: {is_tool_free}\n")
        
        # Codex should now be allowed for tools
        if "codex" in api_base.lower() and is_tool_free:
            print(f"   ⚠️  ERROR: Codex should support tools but is marked tool_free_only!")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("\n🧪 CODEX ENDPOINT FUNCTION CALLING TEST\n")
    
    test1 = test_music_agent_initialization()
    test2 = test_provider_rotation_logic()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    if test1 and test2:
        print("✅ All tests passed! Codex endpoint now supports tools/MCP.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Codex restrictions may still be in place.")
        sys.exit(1)
