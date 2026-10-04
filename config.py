# =============================================================================
# config.py  —  single place to change every tunable parameter
# =============================================================================
# Interview note: centralising config like this mirrors production practice;
# in deployment you'd swap these values via environment variables or a YAML.

# ---------------------------------------------------------------------------
# Video source
#   0            → default webcam
#   "path/to/file.mp4"  → local video file  (default for testing)
#   "rtsp://user:pass@host:554/stream"  → IP camera
# ---------------------------------------------------------------------------
VIDEO_SOURCE = "pedestrians.mp4"
LOOP_VIDEO   = False   # False → stop cleanly at EOF and print a summary; True → loop (old behaviour)

# ---------------------------------------------------------------------------
# Frame skipping (item 2)
# ---------------------------------------------------------------------------
# 1 = process every frame (default, most accurate)
# 2 = process every 2nd frame (roughly doubles FPS, slight accuracy loss)
# 3 = process every 3rd frame (higher throughput, more ID switches in crowds)
FRAME_SKIP = 1

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
MODEL_PATH   = "yolov8n.pt"          # nano model — fast; swap for yolov8s.pt for better accuracy
CONF_THRESH  = 0.40                  # only keep detections above this confidence
# COCO class IDs we care about: 0=person, 2=car, 1=bicycle
TARGET_CLASSES = [0, 1, 2]
CLASS_LABELS   = {0: "person", 1: "bicycle", 2: "car"}

# Colour per class (B, G, R)
CLASS_COLORS   = {0: (0, 255, 0), 1: (255, 165, 0), 2: (0, 165, 255)}

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
# ByteTrack is built into Ultralytics; model.track(persist=True) enables it.
# persist=True keeps the same track IDs across consecutive frames.
TRACKER = "bytetrack.yaml"           # bundled tracker config

# ---------------------------------------------------------------------------
# RTSP reconnect  (only used when VIDEO_SOURCE starts with "rtsp://")
# ---------------------------------------------------------------------------
RTSP_MAX_RETRIES   = 5    # number of reconnect attempts before giving up
RTSP_RETRY_DELAY_S = 2.0  # seconds to wait between each attempt

# ---------------------------------------------------------------------------
# Counting line
#   Expressed as a fraction of frame height so it works at any resolution.
#   0.67 → line sits at ~67 % of the frame height (lower third of upper half)
# ---------------------------------------------------------------------------
LINE_Y_FRACTION = 0.67               # tweak this if the crosswalk moves

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = "events.db"                # SQLite file written next to this script
CLEAR_DB_ON_STARTUP = False          # True → DELETE all old events on every run (clean slate)
