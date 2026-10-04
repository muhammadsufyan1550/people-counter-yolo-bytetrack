# =============================================================================
# export_onnx.py  —  export YOLOv8n to ONNX format
# =============================================================================
# Run once:  python export_onnx.py
# Produces:  yolov8n.onnx  (in the project directory)
#
# Why ONNX?
#   ONNX (Open Neural Network Exchange) is a vendor-neutral model format.
#   Once exported you can run it with ONNX Runtime (CPU/GPU), TensorRT,
#   OpenVINO, or CoreML — without keeping PyTorch installed at inference time.
#   This is important for lean Docker images and edge deployments.
#
# Interview Q: "What is the difference between exporting to ONNX and to TensorRT?"
# A: ONNX is portable across hardware; TensorRT is NVIDIA-specific but applies
#    aggressive kernel fusion and FP16/INT8 quantisation for maximum GPU speed.
#    The typical path is: PyTorch → ONNX → trtexec → TensorRT engine.

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# imgsz must match what you'll use at inference time.
# opset=12 is broadly compatible with ONNX Runtime ≥ 1.8.
path = model.export(format="onnx", imgsz=640, opset=18, simplify=True)
print(f"[OK] Exported to: {path}")
