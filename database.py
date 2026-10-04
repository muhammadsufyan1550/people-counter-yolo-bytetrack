# =============================================================================
# database.py  —  thin SQLite wrapper
# =============================================================================
# We use Python's built-in sqlite3 — no extra dependency.
# In production you'd swap this for PostgreSQL + SQLAlchemy / asyncpg.
#
# Interview Q: "Why SQLite here instead of PostgreSQL?"
# A: SQLite is zero-config and perfect for a single-node prototype.
#    For multi-camera / multi-service deployments you'd use PostgreSQL
#    because it handles concurrent writers and network access.

import sqlite3
import threading
from datetime import datetime
import config

# One lock so the video thread and the API thread don't corrupt the DB.
_lock = threading.Lock()

# Unique identifier for this process run — set once when the module is first imported.
# Using an ISO timestamp means it's human-readable in the DB and needs no extra library.
# ByteTrack resets track IDs to 1 on every run, so without this, track_id=1 from run A
# and track_id=1 from run B are indistinguishable in the events table.
RUN_ID: str = datetime.utcnow().isoformat(timespec="seconds")


def init_db() -> None:
    """Create the events table if it doesn't already exist."""
    with _lock, sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crossing_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id    TEXT    NOT NULL,   -- ISO timestamp of process start; groups events by run
                timestamp TEXT    NOT NULL,   -- ISO-8601 UTC string of the crossing moment
                track_id  INTEGER NOT NULL,   -- persistent ByteTrack ID (resets to 1 each run)
                class_id  INTEGER NOT NULL,   -- 0=person, 1=bicycle, 2=car
                direction TEXT    NOT NULL    -- 'in' or 'out'
            )
        """)
        # Migrate existing DBs that pre-date the run_id column — safe to ignore if already present.
        try:
            conn.execute("ALTER TABLE crossing_events ADD COLUMN run_id TEXT NOT NULL DEFAULT 'legacy'")
        except sqlite3.OperationalError:
            pass  # Column already exists — nothing to do

        # Optional: wipe all old events so /counts starts clean each run.
        # Controlled by CLEAR_DB_ON_STARTUP in config.py.
        if config.CLEAR_DB_ON_STARTUP:
            conn.execute("DELETE FROM crossing_events")
            print(f"[INFO] DB cleared (CLEAR_DB_ON_STARTUP=True). Fresh start for run {RUN_ID}.")

        conn.commit()


def log_event(track_id: int, class_id: int, direction: str) -> None:
    """Insert one crossing event row, tagged with the current RUN_ID."""
    ts = datetime.utcnow().isoformat(timespec="seconds")
    with _lock, sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            "INSERT INTO crossing_events (run_id, timestamp, track_id, class_id, direction) VALUES (?,?,?,?,?)",
            (RUN_ID, ts, track_id, class_id, direction),
        )
        conn.commit()


def get_counts() -> dict:
    """Return in/out counts for the CURRENT run only, broken down by class."""
    with _lock, sqlite3.connect(config.DB_PATH) as conn:
        rows = conn.execute(
            # WHERE run_id = RUN_ID ensures we never mix events from previous runs.
            "SELECT class_id, direction, COUNT(*) FROM crossing_events WHERE run_id=? GROUP BY class_id, direction",
            (RUN_ID,),
        ).fetchall()

    # Build a nested dict: { class_label: { "in": N, "out": N } }
    result = {}
    for class_id, direction, count in rows:
        label = config.CLASS_LABELS.get(class_id, f"class_{class_id}")
        result.setdefault(label, {"in": 0, "out": 0})
        result[label][direction] = count
    return result


def get_recent_events(limit: int = 20) -> list[dict]:
    """Return the most recent crossing events for the CURRENT run only."""
    with _lock, sqlite3.connect(config.DB_PATH) as conn:
        rows = conn.execute(
            """SELECT timestamp, track_id, class_id, direction
               FROM crossing_events
               WHERE run_id=?
               ORDER BY id DESC LIMIT ?""",
            (RUN_ID, limit),
        ).fetchall()

    return [
        {
            "run_id":    RUN_ID,
            "timestamp": r[0],
            "track_id":  r[1],
            "class":     config.CLASS_LABELS.get(r[2], f"class_{r[2]}"),
            "direction": r[3],
        }
        for r in rows
    ]
