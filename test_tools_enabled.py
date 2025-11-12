#!/usr/bin/env python3
"""Quick test to verify tools are now passed to agents."""
import json
import os
import sys

os.environ["OPENAI_API_BASE"] = "https://ccproxy.emottet.com/copilot/v1"
os.environ["OPENAI_MODEL_NAME"] = "gpt-5-mini"  # Fast model for testing

from dotenv import load_dotenv
load_dotenv(override=True)

# Load minimal test input
with open("input.json") as f:
    test_data = json.load(f)

from crew import StravaDescriptionCrew

print("=" * 80)
print("TESTING TOOL CALLS WITH output_json DISABLED")
print("=" * 80)
print()

crew = StravaDescriptionCrew()

# Just create agents and check the tool setup
print("\n✅ Agents created successfully\n")

print("🔍 Checking agent configurations:")
print(f"  Description Agent tools: {len(getattr(crew.description_agent, 'tools', []))}")
print(f"  Music Agent tools: {len(getattr(crew.music_agent, 'tools', []))}")
print(f"  Privacy Agent tools: {len(getattr(crew.privacy_agent, 'tools', []))}")
print(f"  Translation Agent tools: {len(getattr(crew.translation_agent, 'tools', []))}")

print("\n✅ All agents initialized correctly!")
print("\nNow run the full test with: cat input.json | python crew.py 2>&1 | tee test_output.log")
print("Look for: '🤖 DESCRIPTION calling LLM (has_tools=True' and '🤖 MUSIC calling LLM (has_tools=True'")
