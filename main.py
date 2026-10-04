# =============================================================================
# main.py  —  entry point: starts the video loop + FastAPI server
# =============================================================================
# Run with:  python main.py
#
# What happens:
#   1. Initialise the SQLite database (creates tables if missing).
#   2. Start the video processing loop in a daemon background thread.
#   3. Start the FastAPI/Uvicorn server in the main thread.
#   4. When you press Q in the OpenCV window, the stop_event fires,
#      the loop exits, and Uvicorn shuts down automatically (daemon thread).
#
# Interview Q: "Why run the video loop in a background thread?"
# A: Uvicorn's event loop blocks the main thread.  Running video processing
#    in a daemon thread lets both run concurrently without asyncio complexity.
#    For production you'd use multiprocessing or separate Docker services to
#    get true CPU isolation and fault boundaries per camera.

import threading
import uvicorn

import config
import database
from video_loop import run_video_loop
from api import app

def main():
    # Step 1 — ensure DB schema exists
    database.init_db()

    # Step 2 — start video loop in background thread
    stop_event = threading.Event()
    video_thread = threading.Thread(
        target=run_video_loop,
        args=(stop_event,),
        daemon=True,   # killed automatically when main thread exits
        name="VideoLoop",
    )
    video_thread.start()
    print("[INFO] Video loop thread started.")

    # Step 3 — start API server (blocks until Ctrl-C or process exit)
    print(f"[INFO] API available at http://localhost:{config.API_PORT}/docs")
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="warning",   # reduce Uvicorn noise in the terminal
    )

if __name__ == "__main__":
    main()
