#!/usr/bin/env python3
"""Test script to verify privacy agent handles generation failures correctly."""
import json
import sys
from tasks.privacy_task import create_privacy_task
from agents.privacy_agent import create_privacy_agent
from crewai import LLM, Crew, Process
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Mock LLM configuration
base_url = os.getenv("OPENAI_API_BASE", "https://ghcopilot.emottet.com/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5mini")
api_key = os.getenv("OPENAI_API_KEY", "dummy-key")

llm = LLM(
    model=f"openai/{model_name}",
    api_base=base_url,
    api_key=api_key,
    drop_params=True,
    additional_drop_params=["stop"]
)

# Test cases: good content vs failed generation
test_cases = [
    {
        "name": "Failed Generation - Error Message (Lunch time)",
        "activity_data": {
            "object_data": {
                "id": 16284886069,
                "distance": 12337,
                "type": "Run",
                "start_date_local": "2025-10-27T11:54:41Z"  # ~13:54 Paris time = Lunch Run
            }
        },
        "generated_content": json.dumps({
            "title": "Activity completed",
            "description": "I'm unable to complete the task due to issues retrieving data from Intervals.icu. Please verify the connection and try again.",
            "workout_type": "Unknown",
            "key_metrics": {}
        }, indent=2),
        "expected_time": "Lunch Run"
    },
    {
        "name": "Failed Generation - Generic Title (Morning)",
        "activity_data": {
            "object_data": {
                "id": 16284886070,
                "distance": 8500,
                "type": "Run",
                "start_date_local": "2025-10-27T06:30:00Z"  # ~08:30 Paris time = Morning Run
            }
        },
        "generated_content": json.dumps({
            "title": "Activity",
            "description": "Unable to retrieve workout details. Connection error.",
            "workout_type": "Unknown",
            "key_metrics": {}
        }, indent=2),
        "expected_time": "Morning Run"
    },
    {
        "name": "Failed Generation - Evening Long Run",
        "activity_data": {
            "object_data": {
                "id": 16284886072,
                "distance": 25000,
                "type": "Run",
                "start_date_local": "2025-10-27T17:00:00Z"  # ~19:00 Paris time, but 25K = Long Run
            }
        },
        "generated_content": json.dumps({
            "title": "Activity completed",
            "description": "Error connecting to Intervals.icu",
            "workout_type": "Unknown",
            "key_metrics": {}
        }, indent=2),
        "expected_time": "Long Run"  # Distance ≥ 20K takes precedence
    },
    {
        "name": "Valid Generation - Should Pass",
        "activity_data": {
            "object_data": {
                "id": 16284886071,
                "distance": 15000,
                "type": "Run",
                "start_date_local": "2025-10-27T05:30:00Z"  # ~07:30 Paris time
            }
        },
        "generated_content": json.dumps({
            "title": "🏃 15K Long Run - Strong Effort",
            "description": "Great long run this morning. Started easy, built up pace gradually. Avg pace 4:45/km, avg HR 148 bpm. Felt strong throughout! 💪",
            "workout_type": "Long Run",
            "key_metrics": {
                "average_pace": "4:45 /km",
                "average_hr": "148 bpm"
            }
        }, indent=2),
        "expected_time": None  # Valid content, no fallback
    }
]

# Run tests
agent = create_privacy_agent(llm)

for test_case in test_cases:
    print(f"\n{'='*70}")
    print(f"TEST CASE: {test_case['name']}")
    print(f"{'='*70}\n")
    
    print(f"Input generated content:\n{test_case['generated_content']}\n")
    
    # Create and run task
    task = create_privacy_task(
        agent,
        test_case['activity_data'],
        test_case['generated_content']
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        
        # Parse result
        result_str = str(result)
        if result_str.startswith("```json"):
            result_str = result_str[7:]
        if result_str.startswith("```"):
            result_str = result_str[3:]
        if result_str.endswith("```"):
            result_str = result_str[:-3]
        result_str = result_str.strip()
        
        privacy_check = json.loads(result_str)
        
        print(f"\n{'='*70}")
        print(f"RESULT for {test_case['name']}:")
        print(f"{'='*70}")
        print(json.dumps(privacy_check, indent=2))
        print("\n")
        
        # Validate expectations
        if "unable to complete" in test_case['generated_content'].lower() or \
           "Activity completed" in test_case['generated_content'] or \
           test_case['generated_content'].count('"title": "Activity"') > 0:
            # Should have fallback title
            recommended_title = privacy_check.get('recommended_changes', {}).get('title', '')
            recommended_desc = privacy_check.get('recommended_changes', {}).get('description', '')
            expected_time = test_case.get('expected_time', '')
            
            print(f"✅ Expected fallback with '{expected_time}', got: {recommended_title}")
            print(f"✅ Expected empty description, got: '{recommended_desc}'")
            
            # Check if the expected time phrase is in the title
            if expected_time and expected_time in recommended_title and "K" in recommended_title:
                print(f"✅ PASS: Fallback title contains correct time '{expected_time}'")
            elif not expected_time:
                print(f"⚠️  No expected_time specified for test case")
            else:
                print(f"❌ FAIL: Expected '{expected_time}' in title, got '{recommended_title}'")
                
            if recommended_desc == "" or recommended_desc is None:
                print("✅ PASS: Description is empty as expected")
            else:
                print(f"❌ FAIL: Description should be empty, got: '{recommended_desc}'")
        else:
            # Valid content - should pass through or have minimal changes
            print("✅ Valid content - privacy check should approve or suggest minor changes")
        
    except Exception as e:
        print(f"\n❌ ERROR in test case: {str(e)}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("ALL TESTS COMPLETED")
print(f"{'='*70}\n")
