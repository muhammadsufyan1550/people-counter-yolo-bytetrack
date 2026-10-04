# =============================================================================
# benchmark.py  —  compare PyTorch vs ONNX Runtime average FPS (CPU, same video)
# =============================================================================
# Run:  python benchmark.py
# Requires: yolov8n.onnx (run export_onnx.py first)
#
# Interview Q: "Why is ONNX Runtime sometimes faster than PyTorch on CPU?"
# A: PyTorch carries the full autograd + training infrastructure at runtime.
#    ONNX Runtime strips that away and uses optimised execution providers
#    (e.g. MLAS on CPU, CUDA EP on GPU) with graph-level optimisations like
#    operator fusion and constant folding.  On small models the overhead is
#    modest; on GPU with the CUDA EP or TensorRT EP the gap is much larger.

import time
import cv2
import numpy as np
from ultralytics import YOLO

N_FRAMES = 200        # frames used for both benchmarks — keep equal for fairness
IMGSZ    = 640        # must match export imgsz

# ---------------------------------------------------------------------------
# Helper: collect N frames from the video into a list once, so I/O is
# identical for both backends and doesn't skew the timing.
# ---------------------------------------------------------------------------
def load_frames(video_path: str, n: int) -> list:
    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# PyTorch backend  (Ultralytics model.predict)
# ---------------------------------------------------------------------------
def bench_pytorch(frames: list) -> float:
    model = YOLO("yolov8n.pt")
    # Warm-up: one inference to load weights into CPU cache
    model.predict(frames[0], imgsz=IMGSZ, verbose=False)

    t0 = time.perf_counter()
    for frame in frames:
        model.predict(frame, imgsz=IMGSZ, classes=[0, 1, 2], conf=0.4, verbose=False)
    elapsed = time.perf_counter() - t0
    return len(frames) / elapsed


# ---------------------------------------------------------------------------
# ONNX Runtime backend  (direct onnxruntime session, no Ultralytics wrapper)
# ---------------------------------------------------------------------------
def _preprocess(frame: np.ndarray, imgsz: int) -> np.ndarray:
    """Resize → BGR→RGB → HWC→CHW → normalise → add batch dim."""
    img = cv2.resize(frame, (imgsz, imgsz))
    img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
    img = np.transpose(img, (2, 0, 1))                  # HWC → CHW
    return np.expand_dims(img, 0)                        # add batch dim → (1,3,H,W)


def bench_onnx(frames: list) -> float:
    import onnxruntime as ort

    # Use CPUExecutionProvider explicitly (no CUDA here)
    sess = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warm-up
    sess.run(None, {input_name: _preprocess(frames[0], IMGSZ)})

    t0 = time.perf_counter()
    for frame in frames:
        blob = _preprocess(frame, IMGSZ)
        sess.run(None, {input_name: blob})
    elapsed = time.perf_counter() - t0
    return len(frames) / elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    if not os.path.exists("yolov8n.onnx"):
        print("[ERROR] yolov8n.onnx not found — run  python export_onnx.py  first.")
        raise SystemExit(1)

    print(f"Loading {N_FRAMES} frames from video …")
    frames = load_frames("pedestrians.mp4", N_FRAMES)
    print(f"[OK] {len(frames)} frames loaded. Starting benchmarks …\n")

    print("--- PyTorch (CPU) ---")
    pt_fps = bench_pytorch(frames)
    print(f"    Average FPS: {pt_fps:.1f}\n")

    print("--- ONNX Runtime (CPUExecutionProvider) ---")
    ort_fps = bench_onnx(frames)
    print(f"    Average FPS: {ort_fps:.1f}\n")

    speedup = ort_fps / pt_fps
    print("=" * 45)
    print(f"  PyTorch FPS      : {pt_fps:>6.1f}")
    print(f"  ONNX Runtime FPS : {ort_fps:>6.1f}")
    print(f"  Speedup          : {speedup:>6.2f}x")
    print("=" * 45)
    print("""
Note: ONNX Runtime on CPU typically gives a modest speedup (1.2–2x) over
PyTorch because it skips autograd overhead and applies graph optimisations.
The real win comes on GPU: with the CUDA or TensorRT EP you'd expect 3–6x.
On this Windows CPU-only setup the difference reflects ORT's MLAS kernel vs.
PyTorch's default CPU backend.
""")
