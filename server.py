"""HTTP server exposing the Strava description crew as a REST API."""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from crew import StravaDescriptionCrew

app = FastAPI(title="Strava Activity Description Crew")

# Reuse a single crew instance to avoid reloading models for every request.
crew_instance = StravaDescriptionCrew()


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


@app.post("/process")
async def process_activity(request: Request) -> JSONResponse | Dict[str, Any]:
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
        result = crew_instance.process_activity(activity_payload)
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Error while processing request: {exc}\n", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        fallback = _build_error_response(str(exc))
        return JSONResponse(status_code=500, content=fallback)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
