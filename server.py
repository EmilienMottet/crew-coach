"""HTTP server exposing the Strava description crew and meal planning crew as REST APIs."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from crew import StravaDescriptionCrew
from crew_mealy import MealPlanningCrew

app = FastAPI(
    title="Strava Activity & Meal Planning Crew",
    description="Multi-agent CrewAI system for Strava activities and weekly meal planning"
)

# Reuse crew instances to avoid reloading models for every request.
strava_crew_instance = StravaDescriptionCrew()
meal_planning_crew_instance = MealPlanningCrew()


class MealPlanRequest(BaseModel):
    """Request model for meal planning endpoint."""
    week_start_date: Optional[str] = Field(
        default=None,
        description="Start date of the week (YYYY-MM-DD). Defaults to next Monday if not provided."
    )


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Ensure the input matches the structure expected by the crew."""
    if isinstance(payload, dict):
        return payload

    if isinstance(payload, list):
        if not payload:
            raise ValueError("Payload list is empty.")
        first_item = payload[0]
        if isinstance(first_item, dict):
            return first_item
        raise ValueError("List payload elements must be JSON objects.")

    raise ValueError("Payload must be a JSON object or an array of objects.")


def _build_error_response(message: str) -> Dict[str, Any]:
    """Produce the same safe fallback used by the CLI entrypoint."""
    return {
        "error": message,
        "activity_id": None,
        "title": "Error processing activity",
        "description": "An error occurred while generating the description",
        "should_be_private": True,
        "privacy_check": {
            "approved": False,
            "during_work_hours": False,
            "issues": [message],
            "reasoning": "Error during processing",
        },
    }


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    """Simple readiness probe."""
    return {"status": "ok"}


def _calculate_next_monday() -> str:
    """Calculate the date of the next Monday."""
    today = datetime.now()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # If today is Monday, get next Monday
    next_monday = today + timedelta(days=days_until_monday)
    return next_monday.strftime("%Y-%m-%d")


@app.post("/process")
async def process_activity(request: Request) -> JSONResponse:
    """Run the Strava crew against the request body."""
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - FastAPI handles parsing
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc

    try:
        activity_payload = _normalize_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = strava_crew_instance.process_activity(activity_payload)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Error while processing request: {exc}\n", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        fallback = _build_error_response(str(exc))
        return JSONResponse(status_code=500, content=fallback)

    return JSONResponse(content=result)


@app.post("/meal-plan")
async def generate_meal_plan(request_body: MealPlanRequest) -> JSONResponse:
    """
    Generate a weekly meal plan based on Hexis training data.

    Args:
        request_body: Optional week_start_date (YYYY-MM-DD). Defaults to next Monday.

    Returns:
        Complete meal planning result with integration status
    """
    # Determine week start date
    week_start_date = request_body.week_start_date
    if not week_start_date:
        week_start_date = _calculate_next_monday()

    # Validate date format
    try:
        datetime.strptime(week_start_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Expected YYYY-MM-DD, got: {week_start_date}"
        ) from exc

    print(f"\n🍽️  Generating meal plan for week starting: {week_start_date}\n", file=sys.stderr)

    try:
        result = meal_planning_crew_instance.generate_meal_plan(week_start_date)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Error while generating meal plan: {exc}\n", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        error_result = {
            "error": str(exc),
            "week_start_date": week_start_date,
            "summary": "Meal planning failed due to error",
        }
        return JSONResponse(status_code=500, content=error_result)

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
