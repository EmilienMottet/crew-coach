#!/usr/bin/env python3
"""Test if the Music agent LLM accepts tool calls."""

import os
import sys
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Load environment
load_dotenv()

# Create a simple test tool
from crewai.tools import tool

@tool
def test_music_tool(query: str) -> str:
    """Test tool that always returns a success message."""
    return f"Tool called successfully with query: {query}"

# Test different LLM configurations
configs = [
    {
        "name": "Copilot (claude-haiku-4.5)",
        "model": "claude-haiku-4.5",
        "base": "https://ccproxy.emottet.com/copilot/v1",
    },
    {
        "name": "Claude endpoint (claude-sonnet-4-5-20250929)",
        "model": "claude-sonnet-4-5-20250929",
        "base": "https://ccproxy.emottet.com/claude/v1",
    },
    {
        "name": "Codex (gpt-5) - SHOULD FAIL",
        "model": "gpt-5",
        "base": "https://ccproxy.emottet.com/codex/v1",
    },
]

auth_token = os.getenv("OPENAI_API_AUTH_TOKEN")
if auth_token and not auth_token.startswith("Basic "):
    auth_token = f"Basic {auth_token}"

print("=" * 80)
print("Testing Music Agent LLM configurations with tool calling")
print("=" * 80)

for config in configs:
    print(f"\n🧪 Testing: {config['name']}")
    print(f"   Model: {config['model']}")
    print(f"   Endpoint: {config['base']}")

    try:
        # Create LLM
        llm = LLM(
            model=config['model'],
            api_base=config['base'],
            api_key=auth_token
        )

        # Create agent with tool
        agent = Agent(
            role="Test Agent",
            goal="Call the test_music_tool with query 'hello'",
            backstory="You are a test agent. You MUST call test_music_tool with the query parameter set to 'hello'.",
            llm=llm,
            tools=[test_music_tool],
            verbose=False,
            allow_delegation=False,
        )

        # Create task
        task = Task(
            description="Call the test_music_tool with query='hello' and return the result.",
            agent=agent,
            expected_output="The result from test_music_tool",
        )

        # Execute
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        # Check result
        output = str(result.raw if hasattr(result, 'raw') else result)
        if "Tool called successfully" in output:
            print(f"   ✅ SUCCESS: Tool was called!")
            print(f"   Result: {output[:100]}")
        else:
            print(f"   ❌ FAILED: Tool was NOT called")
            print(f"   Result: {output[:200]}")

    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:200]}")

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
