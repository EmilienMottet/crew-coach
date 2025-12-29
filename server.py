"""HTTP server exposing the Strava description crew and meal planning crew as REST APIs."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from crew import StravaDescriptionCrew
from crew_mealy import MealPlanningCrew

app = FastAPI(
    title="Strava Activity & Meal Planning Crew",
    description="Multi-agent CrewAI system for Strava activities and weekly meal planning",
)

# Reuse crew instances to avoid reloading models for every request.
strava_crew_instance = StravaDescriptionCrew()
meal_planning_crew_instance = MealPlanningCrew()


class MealPlanRequest(BaseModel):
    """Request model for meal planning endpoint."""

    week_start_date: Optional[str] = Field(
        default=None,
        description="Start date of the week (YYYY-MM-DD). Defaults to next Monday if not provided.",
    )
    days: Optional[int] = Field(
        default=None,
        ge=1,
        le=7,
        description="Number of consecutive days to generate (1-7). Defaults to 7 when omitted.",
    )


class MealPlanAsyncRequest(BaseModel):
    """Request model for async meal planning endpoint with callback."""

    week_start_date: Optional[str] = Field(
        default=None,
        description="Start date of the week (YYYY-MM-DD). Defaults to next Monday if not provided.",
    )
    days: Optional[int] = Field(
        default=None,
        ge=1,
        le=7,
        description="Number of consecutive days to generate (1-7). Defaults to 7 when omitted.",
    )
    callback_url: str = Field(
        ...,
        description="URL to POST the result when meal planning completes (n8n $execution.resumeUrl)",
    )


# Simple in-memory job tracking for async meal planning
# Key: job_id, Value: {"status": str, "started_at": datetime, "result": Optional[dict]}
_meal_plan_jobs: Dict[str, Dict[str, Any]] = {}


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


async def _run_meal_plan_and_callback(
    job_id: str,
    week_start_date: str,
    days_to_generate: int,
    callback_url: str,
) -> None:
    """Background task: Generate meal plan and POST result to callback URL."""
    print(
        f"\n🚀 [Job {job_id}] Starting async meal plan generation...\n",
        file=sys.stderr,
    )

    _meal_plan_jobs[job_id]["status"] = "running"

    try:
        # Run the synchronous crew in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: meal_planning_crew_instance.generate_meal_plan(
                week_start_date,
                days_to_generate=days_to_generate,
            ),
        )

        _meal_plan_jobs[job_id]["status"] = "completed"
        _meal_plan_jobs[job_id]["result"] = result

        print(
            f"\n✅ [Job {job_id}] Meal plan generation completed. Sending callback...\n",
            file=sys.stderr,
        )

    except Exception as exc:
        print(f"\n❌ [Job {job_id}] Error: {exc}\n", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)

        result = {
            "error": str(exc),
            "job_id": job_id,
            "week_start_date": week_start_date,
            "summary": "Meal planning failed due to error",
        }
        _meal_plan_jobs[job_id]["status"] = "failed"
        _meal_plan_jobs[job_id]["result"] = result

    # Send callback to n8n resumeUrl
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                callback_url,
                json=result,
                headers={"Content-Type": "application/json"},
            )
            print(
                f"\n📤 [Job {job_id}] Callback sent to {callback_url}: HTTP {response.status_code}\n",
                file=sys.stderr,
            )
    except Exception as callback_exc:
        print(
            f"\n❌ [Job {job_id}] Failed to send callback: {callback_exc}\n",
            file=sys.stderr,
        )


@app.post("/meal-plan-async")
async def generate_meal_plan_async(
    request_body: MealPlanAsyncRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Start async meal plan generation and return immediately with a job ID.

    The result will be POSTed to the callback_url when complete.
    Use this endpoint with n8n Wait node (On Webhook Call) for non-blocking execution.

    Args:
        request_body: week_start_date, days, and callback_url (required)

    Returns:
        Job ID for tracking (optional - result comes via callback)
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
            detail=f"Invalid date format. Expected YYYY-MM-DD, got: {week_start_date}",
        ) from exc

    days_to_generate = request_body.days or 7
    callback_url = request_body.callback_url

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Track job
    _meal_plan_jobs[job_id] = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "week_start_date": week_start_date,
        "days": days_to_generate,
        "callback_url": callback_url,
        "result": None,
    }

    print(
        f"\n🍽️  [Job {job_id}] Queued async meal plan: week={week_start_date}, days={days_to_generate}\n"
        f"   Callback URL: {callback_url}\n",
        file=sys.stderr,
    )

    # Schedule background task
    background_tasks.add_task(
        _run_meal_plan_and_callback,
        job_id,
        week_start_date,
        days_to_generate,
        callback_url,
    )

    # Return immediately
    return JSONResponse(
        status_code=202,  # Accepted
        content={
            "job_id": job_id,
            "status": "accepted",
            "week_start_date": week_start_date,
            "days": days_to_generate,
            "message": "Meal plan generation started. Result will be sent to callback URL.",
        },
    )


@app.get("/meal-plan-status/{job_id}")
async def get_meal_plan_status(job_id: str) -> JSONResponse:
    """
    Check the status of an async meal plan job.

    Args:
        job_id: The job ID returned by /meal-plan-async

    Returns:
        Job status and result (if completed)
    """
    if job_id not in _meal_plan_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _meal_plan_jobs[job_id]
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": job["status"],
            "started_at": job["started_at"],
            "week_start_date": job.get("week_start_date"),
            "has_result": job["result"] is not None,
        }
    )


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
            detail=f"Invalid date format. Expected YYYY-MM-DD, got: {week_start_date}",
        ) from exc

    days_to_generate = request_body.days or 7

    print(
        f"\n🍽️  Generating meal plan for week starting: {week_start_date} (days={days_to_generate})\n",
        file=sys.stderr,
    )

    try:
        result = meal_planning_crew_instance.generate_meal_plan(
            week_start_date,
            days_to_generate=days_to_generate,
        )
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
