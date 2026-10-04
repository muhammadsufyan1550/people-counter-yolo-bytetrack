# =============================================================================
# tracker_state.py  —  shared in-memory state between video thread & API
# =============================================================================
# Both the video processing loop and the FastAPI handlers need to read the
# current FPS, counts, and running status.  We store them here as module-level
# variables protected by a threading.Lock.
#
# Interview Q: "How do you share state between threads safely?"
# A: Use a lock (mutex). Python's GIL helps for simple reads, but a Lock
#    makes intent explicit and prevents partial writes on composite values.

import threading

_lock = threading.Lock()

# Live metrics updated every frame by the video loop
_state = {
    "fps":          0.0,
    "frame_count":  0,
    "running":      False,
    "finished":     False,   # True after EOF stop (LOOP_VIDEO=False)
    "reconnecting": False,   # True while RTSP reconnect is in progress
    "in_counts":    {},      # { class_label: count }
    "out_counts":   {},
}


def update(fps: float, frame_count: int, in_counts: dict, out_counts: dict) -> None:
    with _lock:
        _state["fps"]         = round(fps, 1)
        _state["frame_count"] = frame_count
        _state["in_counts"]   = dict(in_counts)
        _state["out_counts"]  = dict(out_counts)


def set_running(val: bool) -> None:
    with _lock:
        _state["running"] = val


def set_finished(val: bool) -> None:
    with _lock:
        _state["finished"] = val


def set_reconnecting(val: bool) -> None:
    with _lock:
        _state["reconnecting"] = val


def snapshot() -> dict:
    """Return a safe copy of the current state."""
    with _lock:
        return dict(_state)
