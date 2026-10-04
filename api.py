# =============================================================================
# api.py  —  FastAPI REST interface
# =============================================================================
# Runs inside the main thread; the video loop runs in a background thread.
#
# Endpoints
#   GET /counts          → live in/out totals from the DB
#   GET /events?limit=20 → most recent crossing events
#   GET /stats           → live FPS, frame count, running status
#
# Interview Q: "Why FastAPI over Flask?"
# A: FastAPI is async-native, auto-generates OpenAPI docs at /docs,
#    uses Pydantic for validation, and is much faster under load.
#    It's becoming the default for ML inference services.

from fastapi import FastAPI, Query
import database
import tracker_state

app = FastAPI(
    title="People Counter API",
    description="Real-time object counting from CCTV/video stream.",
    version="1.0.0",
)


@app.get("/counts", summary="Total crossing counts per class and direction")
def counts():
    """
    Returns accumulated in/out totals from the SQLite database,
    grouped by object class (person, bicycle, car).
    """
    return database.get_counts()


@app.get("/events", summary="Recent line-crossing events")
def events(limit: int = Query(default=20, ge=1, le=200)):
    """
    Returns the most recent `limit` crossing events in reverse
    chronological order.
    """
    return database.get_recent_events(limit)


@app.get("/stats", summary="Live pipeline metrics")
def stats():
    """
    Returns live FPS, total frames processed, and whether the
    video loop is still running.  Useful for monitoring dashboards.
    """
    return tracker_state.snapshot()
