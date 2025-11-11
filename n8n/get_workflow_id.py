#!/usr/bin/env python3
"""
Helper script to retrieve n8n workflow IDs from the configuration file.

Usage:
    python get_workflow_id.py update_strava_activity
    python get_workflow_id.py weekly_meal_plan
    python get_workflow_id.py --list
"""

import json
import sys
from pathlib import Path

WORKFLOW_IDS_FILE = Path(__file__).parent / "workflows" / "workflow_ids.json"


def load_workflow_ids():
    """Load workflow IDs from configuration file."""
    with open(WORKFLOW_IDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_workflow_id(workflow_key: str) -> str:
    """Get workflow ID by key."""
    data = load_workflow_ids()
    if workflow_key in data["workflows"]:
        return data["workflows"][workflow_key]["id"]
    raise ValueError(f"Workflow key '{workflow_key}' not found")


def list_workflows():
    """List all available workflows."""
    data = load_workflow_ids()
    print("Available workflows:")
    print("-" * 60)
    for key, info in data["workflows"].items():
        print(f"Key: {key}")
        print(f"  ID: {info['id']}")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_workflow_id.py <workflow_key>")
        print("       python get_workflow_id.py --list")
        sys.exit(1)

    if sys.argv[1] == "--list":
        list_workflows()
    else:
        workflow_key = sys.argv[1]
        try:
            workflow_id = get_workflow_id(workflow_key)
            print(workflow_id)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("\nUse --list to see available workflow keys", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
