"""
export_onnx.py  ─  Phase 2: Edge Conversion + FP16 Quantization
================================================================
Converts the trained FP32 YOLOv8n (.pt) weights into an ONNX model
with FP16 (half-precision) quantization applied.

Why ONNX + FP16?
  • ONNX is runtime-agnostic: runs on ONNX Runtime (CPU/GPU/NPU),
    TensorRT, OpenVINO, CoreML, and most edge accelerators.
  • FP16 halves model size vs FP32 with < 1% mAP drop on detection tasks.
  • INT8 would shrink further but requires a calibration dataset and
    introduces larger accuracy loss on small-object classes (no-hardhat).
  • FP16 is natively accelerated on NVIDIA Ampere+ (RTX 30xx/40xx),
    Apple Silicon, and Qualcomm Hexagon DSP — the primary edge targets.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights runs/train/yolov8n_ppe_fp32/weights/best.pt
    python scripts/export_onnx.py --quant int8 --calib-images data/ppe_dataset/images/val

Requirements:
    pip install ultralytics onnx onnxruntime onnxsim
    (Optional for INT8): pip install onnxruntime-tools
"""

import argparse
import os
import sys
import time
import shutil
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 → ONNX (FP16/INT8)")
    p.add_argument("--weights",  type=str,
                   default="runs/train/yolov8n_ppe_fp32/weights/best.pt",
                   help="Path to trained .pt weights")
    p.add_argument("--imgsz",    type=int, default=640,
                   help="Inference image size (default: 640)")
    p.add_argument("--quant",    choices=["fp16", "int8"], default="fp16",
                   help="Quantization precision (default: fp16)")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="Run onnx-simplifier to fold constants (default: True)")
    p.add_argument("--dynamic",  action="store_true", default=False,
                   help="Export with dynamic batch axis (default: static batch=1)")
    p.add_argument("--output-dir", type=str, default="models",
                   help="Where to copy final .onnx (default: models/)")
    p.add_argument("--calib-images", type=str, default="",
                   help="[INT8 only] Path to calibration images folder")
    p.add_argument("--benchmark", action="store_true", default=True,
                   help="Run latency benchmark after export (default: True)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Size helper
# ─────────────────────────────────────────────────────────────────────────────
def file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# FP16 Export via Ultralytics
# ─────────────────────────────────────────────────────────────────────────────
def export_fp16(weights: str, imgsz: int, simplify: bool, dynamic: bool) -> str:
    from ultralytics import YOLO

    print("\n[INFO] Exporting YOLOv8 → ONNX (FP16) …")
    model = YOLO(weights)

    # Ultralytics handles half=True → stores weights in FP16 within ONNX graph
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=True,           # ← FP16 quantization
        dynamic=dynamic,
        simplify=simplify,
        opset=17,            # ONNX opset 17 is broadly supported
        batch=1,
    )
    print(f"[OK]  ONNX (FP16) written → {exported_path}")
    return str(exported_path)


# ─────────────────────────────────────────────────────────────────────────────
# INT8 Post-Training Quantization via ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────
def export_int8(weights: str, imgsz: int, simplify: bool,
                dynamic: bool, calib_images: str) -> str:
    """
    INT8 quantization workflow:
      1. Export FP32 ONNX from Ultralytics
      2. Collect calibration statistics on representative images
      3. Apply static INT8 quantization via onnxruntime.quantization
    """
    from ultralytics import YOLO
    import numpy as np
    import cv2

    # Step 1 — Export FP32 ONNX first
    print("\n[INFO] Step 1/3 — Exporting FP32 ONNX (pre-quantization)…")
    model = YOLO(weights)
    fp32_path = str(model.export(
        format="onnx", imgsz=imgsz, half=False,
        dynamic=dynamic, simplify=simplify, opset=17, batch=1,
    ))
    print(f"[OK]  FP32 ONNX → {fp32_path}")

    # Step 2 — Calibration data reader
    print("\n[INFO] Step 2/3 — Collecting calibration statistics…")
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader,
            QuantType, QuantFormat
        )
    except ImportError:
        print("[ERROR] onnxruntime-tools not found.")
        print("        Run: pip install onnxruntime-tools")
        sys.exit(1)

    class PPECalibReader(CalibrationDataReader):
        def __init__(self, images_dir: str, img_size: int = 640, n_images: int = 200):
            self.img_size = img_size
            self.paths = []
            if images_dir and Path(images_dir).exists():
                exts = {".jpg", ".jpeg", ".png", ".bmp"}
                self.paths = [
                    p for p in Path(images_dir).rglob("*") if p.suffix.lower() in exts
                ][:n_images]
            if not self.paths:
                print("[WARN] No calibration images found — using random noise (suboptimal).")
                self._use_random = True
                self._count = 0
                self._max = 100
            else:
                print(f"[INFO] Using {len(self.paths)} calibration images from {images_dir}")
                self._use_random = False
            self._iter = iter(self.paths)

        def get_next(self):
            if self._use_random:
                if self._count >= self._max:
                    return None
                self._count += 1
                noise = np.random.randint(0, 256,
                    (1, 3, self.img_size, self.img_size), dtype=np.float32) / 255.0
                return {"images": noise}
            try:
                path = next(self._iter)
                img = cv2.imread(str(path))
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[np.newaxis]  # NCHW
                return {"images": img}
            except StopIteration:
                return None

    # Step 3 — Quantize
    print("\n[INFO] Step 3/3 — Applying INT8 static quantization…")
    int8_path = fp32_path.replace(".onnx", "_int8.onnx")
    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=PPECalibReader(calib_images, imgsz),
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"[OK]  INT8 ONNX → {int8_path}")
    return int8_path


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark the exported model
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_onnx(onnx_path: str, imgsz: int, n_warmup: int = 10, n_runs: int = 100):
    import numpy as np
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    # Replace dynamic dims with concrete values
    shape = [d if isinstance(d, int) else 1 for d in inp_shape]
    dtype = np.float16 if sess.get_inputs()[0].type == "tensor(float16)" else np.float32
    dummy = np.random.rand(*shape).astype(dtype)

    # Warmup
    for _ in range(n_warmup):
        sess.run(None, {inp_name: dummy})

    import time
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {inp_name: dummy})
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms per inference

    return elapsed, 1000 / elapsed  # latency_ms, fps


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 65)
    print("  PPE Edge Detection — Phase 2: ONNX Conversion + Quantization")
    print("=" * 65)
    print(f"  Source weights : {args.weights}")
    print(f"  Quantization   : {args.quant.upper()}")
    print(f"  Image size     : {args.imgsz}×{args.imgsz}")
    print(f"  Simplify       : {args.simplify}")
    print("=" * 65)

    if not Path(args.weights).exists():
        print(f"\n[ERROR] Weights not found: {args.weights}")
        print("        Run train.py first, or adjust --weights path.")
        sys.exit(1)

    # ── FP32 size for comparison ──────────────────────────────────────────────
    fp32_size_mb = file_mb(args.weights)
    print(f"\n[INFO] FP32 .pt size : {fp32_size_mb:.2f} MB")

    # ── Export ────────────────────────────────────────────────────────────────
    t_export_start = time.time()
    if args.quant == "fp16":
        onnx_path = export_fp16(args.weights, args.imgsz, args.simplify, args.dynamic)
    else:
        onnx_path = export_int8(
            args.weights, args.imgsz, args.simplify, args.dynamic, args.calib_images
        )
    export_time = time.time() - t_export_start

    # ── Copy to output dir ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(onnx_path).name
    shutil.copy2(onnx_path, dest)

    # ── Size comparison ───────────────────────────────────────────────────────
    quant_size_mb = file_mb(str(dest))
    compression   = (1 - quant_size_mb / fp32_size_mb) * 100

    print(f"\n  ┌─ Conversion Summary ─────────────────────────────────┐")
    print(f"  │  FP32 .pt size        : {fp32_size_mb:>8.2f} MB               │")
    print(f"  │  {args.quant.upper()} ONNX size       : {quant_size_mb:>8.2f} MB               │")
    print(f"  │  Size reduction       : {compression:>7.1f} %                │")
    print(f"  │  Export time          : {export_time:>7.1f} s                │")
    print(f"  │  Saved to             : {str(dest):<30}│")
    print(f"  └──────────────────────────────────────────────────────┘")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if args.benchmark:
        print(f"\n[INFO] Benchmarking {args.quant.upper()} ONNX (100 runs, batch=1)…")
        try:
            lat_ms, fps = benchmark_onnx(str(dest), args.imgsz)
            print(f"  Inference latency : {lat_ms:.2f} ms / frame")
            print(f"  Throughput        : {fps:.1f} FPS")
        except Exception as e:
            print(f"[WARN] Benchmark failed: {e}")

    print(f"\n  Next step → Run:  python live_inference.py --model {dest}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
