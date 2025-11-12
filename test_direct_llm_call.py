#!/usr/bin/env python3
"""Direct test: Call music agent's LLM manually to see what happens."""
import os
import sys

os.environ["OPENAI_API_BASE"] = "https://ccproxy.emottet.com/copilot/v1"
os.environ["OPENAI_MODEL_NAME"] = "gpt-5-mini"

from dotenv import load_dotenv
load_dotenv(override=True)

# CRITICAL: Import llm_auth_init BEFORE any CrewAI imports to apply patches
import llm_auth_init  # noqa: F401 - Side effects only

from crew import StravaDescriptionCrew

print("Creating crew...")
crew_instance = StravaDescriptionCrew()

print(f"\nMusic Agent LLM type: {type(crew_instance.music_llm)}")
print(f"Music Agent tools: {len(getattr(crew_instance.music_agent, 'tools', []))}")

# Try to call the LLM directly with a simple message
print("\nAttempting direct LLM call...")

try:
    result = crew_instance.music_llm.call(
        messages="Hello, can you list available tools?",
        from_agent=crew_instance.music_agent,
    )
    print(f"\n✅ LLM responded: {result[:200] if result else 'No response'}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
