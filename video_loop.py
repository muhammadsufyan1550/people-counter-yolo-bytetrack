# =============================================================================
# video_loop.py  —  detection, tracking, line-crossing logic, OpenCV display
# =============================================================================
# This is the "heart" of the pipeline:
#   Camera/file → YOLO detect → ByteTrack assign IDs → line-cross check
#   → draw overlay → update shared state → log event to DB
#
# Design note: "How does ByteTrack work at a high level?"
# A: ByteTrack separates detections into high-confidence and low-confidence
#    bins.  High-confidence ones are matched to existing tracks via IoU.
#    Low-confidence ones get a second chance — matched to unmatched tracks
#    before being discarded.  This recovers occluded objects that YOLO
#    temporarily loses, giving stable IDs through crowds.
#
# Interview Q: "How do you detect a line crossing without missing events?"
# A: Store each track's Y-position from the *previous* frame.  On the current
#    frame, if the track transitions from one side of the line to the other,
#    it crossed.  Using previous vs. current avoids duplicate triggers.

import time
import threading
import collections

import cv2
from ultralytics import YOLO

import config
import database
import tracker_state

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_line_y(frame_h: int) -> int:
    """Convert the fractional line position to an absolute pixel Y coordinate."""
    return int(frame_h * config.LINE_Y_FRACTION)


def _draw_overlay(frame, line_y: int, in_counts: dict, out_counts: dict, fps: float):
    """
    Draw the counting line, per-class counts, and FPS onto the frame.
    Modifies the frame in-place (OpenCV draws in-place on numpy arrays).
    """
    h, w = frame.shape[:2]

    # --- Counting line ---
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
    cv2.putText(frame, "COUNT LINE", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    # --- Arrows hinting direction ---
    cv2.putText(frame, "IN  v", (w - 100, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(frame, "OUT ^", (w - 100, line_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # --- Counts panel (top-left) ---
    panel_lines = ["=== COUNTS ==="]
    all_classes = sorted(set(list(in_counts.keys()) + list(out_counts.keys())))
    for cls in all_classes:
        panel_lines.append(f"{cls:>8s}  IN:{in_counts.get(cls,0):>3d}  OUT:{out_counts.get(cls,0):>3d}")
    panel_lines.append(f"FPS: {fps:>5.1f}")

    for i, line in enumerate(panel_lines):
        y_pos = 28 + i * 22
        # Dark background for readability
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        cv2.rectangle(frame, (6, y_pos - th - 4), (6 + tw + 4, y_pos + 4), (0, 0, 0), -1)
        cv2.putText(frame, line, (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)


def _draw_track(frame, x1, y1, x2, y2, track_id: int, class_id: int, conf: float):
    """Draw bounding box, label, and track ID for one detection."""
    color = config.CLASS_COLORS.get(class_id, (200, 200, 200))
    label = config.CLASS_LABELS.get(class_id, "?")
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"{label} #{track_id} {conf:.2f}"
    cv2.putText(frame, tag, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ---------------------------------------------------------------------------
# Main loop — meant to run in a background thread
# ---------------------------------------------------------------------------

def _is_rtsp(source) -> bool:
    """Return True if the source looks like an RTSP URL (not a file or webcam index)."""
    return isinstance(source, str) and source.lower().startswith("rtsp://")


def _open_capture(source, max_retries: int, delay: float):
    """
    Try to open a VideoCapture, retrying on failure.
    For file sources, max_retries=1 (no retry needed; failure = file missing).
    For RTSP sources, retries allow recovery from transient network drops.

    Design note: "How do you handle an RTSP stream that drops mid-session?"
    A: On cap.read() failure, release the old handle, set a 'reconnecting' flag
       visible in /stats, sleep briefly, then re-open — up to a max retry count.
       Exponential backoff would be better in production; here we use a fixed
       delay for simplicity.
    """
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"[WARN] Cannot open source (attempt {attempt}/{max_retries}): {source}")
        if attempt < max_retries:
            time.sleep(delay)
    return None


def run_video_loop(stop_event: threading.Event) -> None:
    """
    Opens the video source, runs detection+tracking every frame,
    checks for line crossings, updates shared state and DB.

    stop_event: set this from outside to gracefully stop the loop.
    """
    # --- Model ---
    # verbose=False suppresses per-frame console spam from Ultralytics
    model = YOLO(config.MODEL_PATH)

    # --- Reconnect settings ---
    is_rtsp      = _is_rtsp(config.VIDEO_SOURCE)
    max_retries  = config.RTSP_MAX_RETRIES  if is_rtsp else 1
    retry_delay  = config.RTSP_RETRY_DELAY_S if is_rtsp else 0

    # --- Initial open ---
    cap = _open_capture(config.VIDEO_SOURCE, max_retries, retry_delay)
    if cap is None:
        print(f"[ERROR] Cannot open video source: {config.VIDEO_SOURCE}")
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_y  = _compute_line_y(frame_h)

    print(f"[INFO] Source: {config.VIDEO_SOURCE}  |  Resolution: {frame_w}x{frame_h}  |  Line Y: {line_y}px")

    # --- Per-class crossing counts (live, in memory) ---
    in_counts:  dict = collections.defaultdict(int)   # { class_label: count }
    out_counts: dict = collections.defaultdict(int)

    # --- Track history: maps track_id → last known centre_y ---
    # Used to detect the frame-to-frame transition across the line.
    prev_cy: dict[int, float] = {}

    # --- Set to track which IDs have already been counted ---
    # Prevents double-counting if a track hovers exactly on the line.
    counted_ids: set[int] = set()

    # --- FPS calculation ---
    fps        = 0.0
    frame_count = 0
    t_start    = time.perf_counter()

    tracker_state.set_running(True)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if is_rtsp:
                # RTSP drop — attempt reconnect before giving up.
                print(f"[WARN] Stream read failed. Reconnecting ({config.RTSP_MAX_RETRIES} attempts) …")
                tracker_state.set_reconnecting(True)
                cap.release()
                cap = _open_capture(config.VIDEO_SOURCE, max_retries, retry_delay)
                tracker_state.set_reconnecting(False)
                if cap is None:
                    print("[ERROR] Reconnect failed. Stopping.")
                    break
                # Reset tracking state — track IDs from before the drop are stale
                prev_cy.clear()
                counted_ids.clear()
                print("[INFO] Reconnected successfully.")
                continue

            elif config.LOOP_VIDEO:
                # File loop mode — seek to frame 0 and keep running
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_cy.clear()
                counted_ids.clear()
                continue
            else:
                # File stop mode — print summary and exit cleanly.
                # Counts remain in tracker_state so /counts and /stats stay readable.
                elapsed_total = time.perf_counter() - t_start
                avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0.0
                print("\n[DONE] Video finished.")
                print(f"       Average FPS : {avg_fps:.1f}")
                print(f"       Total frames: {frame_count}")
                print("       Crossing summary:")
                all_cls = sorted(set(list(in_counts.keys()) + list(out_counts.keys())))
                for cls in all_cls:
                    print(f"         {cls:>10s}  IN={in_counts.get(cls,0):>3d}  OUT={out_counts.get(cls,0):>3d}")
                tracker_state.set_finished(True)
                stop_event.set()
                break

        frame_count += 1

        # ----------------------------------------------------------------
        # Frame skipping — run inference only every FRAME_SKIP frames.
        # On skipped frames we still show the window (smooth playback) but
        # reuse the last overlay-drawn frame to avoid a black/stale display.
        # ----------------------------------------------------------------
        if config.FRAME_SKIP > 1 and frame_count % config.FRAME_SKIP != 0:
            # Show previous annotated frame so the window doesn't go black
            if hasattr(run_video_loop, "_last_annotated"):
                cv2.imshow("People Counter — press Q to quit", run_video_loop._last_annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
            # Still update FPS (based on all frames, not just inferred ones)
            elapsed = time.perf_counter() - t_start
            if elapsed > 0:
                fps = frame_count / elapsed
            tracker_state.update(fps, frame_count, dict(in_counts), dict(out_counts))
            continue

        # ----------------------------------------------------------------
        # YOLO inference + ByteTrack tracking
        #   classes=TARGET_CLASSES  → only detect our objects (faster)
        #   persist=True            → ByteTrack keeps IDs across frames
        #   verbose=False           → no per-frame stdout noise
        # ----------------------------------------------------------------
        results = model.track(
            frame,
            classes=config.TARGET_CLASSES,
            conf=config.CONF_THRESH,
            tracker=config.TRACKER,
            persist=True,
            verbose=False,
        )

        # results[0].boxes contains all detections for this frame
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            # xyxy: [[x1,y1,x2,y2], ...]  — absolute pixel coords
            xyxy     = boxes.xyxy.cpu().numpy().astype(int)
            ids      = boxes.id.cpu().numpy().astype(int)
            cls_ids  = boxes.cls.cpu().numpy().astype(int)
            confs    = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, cid, conf in zip(xyxy, ids, cls_ids, confs):
                # Vertical centre of the box is a stable anchor point
                cy = (y1 + y2) / 2.0

                # --- Line-crossing detection ---
                if tid in prev_cy and tid not in counted_ids:
                    prev = prev_cy[tid]
                    cls_label = config.CLASS_LABELS.get(cid, f"class_{cid}")

                    # Crossed from above-line → below-line  → moving toward camera → IN
                    if prev < line_y <= cy:
                        in_counts[cls_label] += 1
                        counted_ids.add(tid)
                        database.log_event(int(tid), int(cid), "in")

                    # Crossed from below-line → above-line  → moving away → OUT
                    elif prev > line_y >= cy:
                        out_counts[cls_label] += 1
                        counted_ids.add(tid)
                        database.log_event(int(tid), int(cid), "out")

                # Update last-known Y for next frame
                prev_cy[tid] = cy

                # Draw bounding box + label
                _draw_track(frame, x1, y1, x2, y2, tid, cid, float(conf))

        # ----------------------------------------------------------------
        # FPS — cumulative average since start (or last 500-frame reset)
        # ----------------------------------------------------------------
        elapsed = time.perf_counter() - t_start
        if elapsed > 0:
            fps = frame_count / elapsed

        # Reset counters every 500 frames to keep rolling average recent
        if frame_count >= 500:
            frame_count = 0
            t_start = time.perf_counter()

        # ----------------------------------------------------------------
        # Draw overlay and update shared state
        # ----------------------------------------------------------------
        _draw_overlay(frame, line_y, dict(in_counts), dict(out_counts), fps)
        tracker_state.update(fps, frame_count, dict(in_counts), dict(out_counts))

        # Cache the fully annotated frame for frame-skip display on skipped frames
        run_video_loop._last_annotated = frame.copy()

        # ----------------------------------------------------------------
        # Show frame
        # ----------------------------------------------------------------
        cv2.imshow("People Counter — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracker_state.set_running(False)
    print("[INFO] Video loop stopped.")
