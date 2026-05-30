"""
live_inference.py  ─  Phase 3: Edge Inference with Real-Time Metrics
=====================================================================
Loads the quantized ONNX model and runs inference on a test video
or live webcam feed, overlaying:

  ✓ Bounding boxes with class labels and confidence scores
  ✓ Live Inference FPS  (pure model time, excluding render)
  ✓ Pre-processing latency  (ms)
  ✓ Post-processing / NMS latency  (ms)
  ✓ Device used  (CPU / CUDA)
  ✓ Per-class detection count

Usage:
    # Webcam (device 0)
    python live_inference.py --model models/best_fp16.onnx

    # Test video file
    python live_inference.py --model models/best_fp16.onnx --source test_video.mp4

    # Save output
    python live_inference.py --model models/best_fp16.onnx --source video.mp4 --save

    # Adjust thresholds
    python live_inference.py --model models/best_fp16.onnx --conf 0.45 --iou 0.5

Requirements:
    pip install onnxruntime-gpu opencv-python numpy
    (CPU-only): pip install onnxruntime opencv-python numpy
"""

import argparse
import sys
import time
import collections
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants — PPE Dataset Classes & Colors
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "head",       # no helmet — violation
    1: "helmet",     # wearing helmet — safe
}

# BGR color palette per class
CLASS_COLORS = {
    0: (0,   0, 230),    # head (no helmet) → red   (violation)
    1: (0,  210,  0),    # helmet           → green (safe)
}

# Violation classes (triggers red alert bar)
VIOLATION_CLASSES = {0}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PPE Edge Inference (ONNX)")
    p.add_argument("--model",  type=str, required=True,
                   help="Path to quantized .onnx model file")
    p.add_argument("--source", type=str, default="0",
                   help="Video source: '0' = webcam, or path to .mp4/.avi")
    p.add_argument("--conf",   type=float, default=0.40,
                   help="Confidence threshold (default: 0.40)")
    p.add_argument("--iou",    type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    p.add_argument("--imgsz",  type=int,   default=640,
                   help="Model input size (default: 640)")
    p.add_argument("--save",   action="store_true",
                   help="Save annotated output to output_<source>.mp4")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress live window (useful for headless servers)")
    p.add_argument("--fp16",   action="store_true", default=True,
                   help="Use FP16 inference (default: True, auto-detected)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime Session
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed.")
        print("        pip install onnxruntime-gpu  (or onnxruntime for CPU-only)")
        sys.exit(1)

    available = ort.get_available_providers()
    providers  = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    device_label = "CUDA" if "CUDAExecutionProvider" in providers else "CPU"

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    inp  = sess.get_inputs()[0]
    dtype = np.float16 if inp.type == "tensor(float16)" else np.float32

    print(f"[OK]  Model loaded  : {model_path}")
    print(f"      Input name    : {inp.name}  shape={inp.shape}  dtype={inp.type}")
    print(f"      Device        : {device_label}  ({providers[0]})")
    return sess, inp.name, dtype, device_label


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray, imgsz: int, dtype) -> tuple:
    """
    Letterbox-resize → normalize → NCHW → cast.
    Returns (blob, scale, pad_x, pad_y) for coordinate remapping.
    """
    h, w = frame.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top    = (imgsz - new_h) // 2
    pad_bottom =  imgsz - new_h - pad_top
    pad_left   = (imgsz - new_w) // 2
    pad_right  =  imgsz - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    rgb   = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    norm  = rgb.astype(np.float32) / 255.0
    nchw  = np.transpose(norm, (2, 0, 1))[np.newaxis]   # 1×3×H×W
    blob  = nchw.astype(dtype)

    return blob, scale, pad_left, pad_top


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing: decode YOLOv8 raw output + NMS
# ─────────────────────────────────────────────────────────────────────────────
def postprocess(
    output: np.ndarray,
    orig_h: int, orig_w: int,
    scale: float, pad_x: int, pad_y: int,
    conf_thresh: float, iou_thresh: float
) -> list[dict]:
    """
    YOLOv8 ONNX output shape: [1, 4+nc, num_anchors]
    Each column = [cx, cy, w, h, cls0_score, cls1_score, …]
    """
    preds = output[0]          # [4+nc, num_anchors]
    preds = preds.T            # [num_anchors, 4+nc]

    boxes_xywh = preds[:, :4]
    scores     = preds[:, 4:]  # [num_anchors, nc]

    cls_ids = np.argmax(scores, axis=1)
    confs   = scores[np.arange(len(scores)), cls_ids]

    mask = confs >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    confs       = confs[mask]
    cls_ids     = cls_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2 (in padded-image space)
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Remap to original frame coordinates
    x1 = np.clip((x1 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_y) / scale, 0, orig_h)

    # OpenCV NMS (operates on [x,y,w,h] format, per class)
    detections = []
    unique_cls  = np.unique(cls_ids)
    for cls in unique_cls:
        m   = cls_ids == cls
        bxs = np.stack([x1[m], y1[m], x2[m]-x1[m], y2[m]-y1[m]], axis=1)
        cfs = confs[m].tolist()

        indices = cv2.dnn.NMSBoxes(
            bxs.tolist(), cfs, conf_thresh, iou_thresh
        )
        if len(indices) == 0:
            continue
        for i in (indices.flatten() if isinstance(indices, np.ndarray) else indices):
            detections.append({
                "x1":   int(x1[m][i]), "y1": int(y1[m][i]),
                "x2":   int(x2[m][i]), "y2": int(y2[m][i]),
                "conf": float(cfs[i]),
                "cls":  int(cls),
            })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Drawing / HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_box(frame, det: dict):
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    cls   = det["cls"]
    conf  = det["conf"]
    color = CLASS_COLORS.get(cls, (200, 200, 200))
    label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"

    # Bounding box
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(
    frame,
    fps_inf: float,
    pre_ms: float,
    post_ms: float,
    device: str,
    detections: list,
    conf_thresh: float,
    model_name: str,
    violation: bool,
):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Top metrics bar ───────────────────────────────────────────────────────
    bar_h = 110
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    alpha = 0.78
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Title
    cv2.putText(frame, "PPE SAFETY MONITOR  [EDGE]",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # Row 1
    fps_color = (0, 255, 80) if fps_inf >= 25 else (0, 180, 255) if fps_inf >= 12 else (0, 60, 255)
    cv2.putText(frame, f"INF FPS: {fps_inf:6.1f}",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, fps_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"PRE: {pre_ms:.1f} ms",
                (190, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"POST/NMS: {post_ms:.1f} ms",
                (330, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    # Row 2
    cv2.putText(frame, f"Device: {device}",
                (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Model: {Path(model_name).stem}",
                (140, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {conf_thresh:.2f}  |  Dets: {len(detections)}",
                (370, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1, cv2.LINE_AA)

    # ── Per-class count sidebar ───────────────────────────────────────────────
    counts = collections.Counter(d["cls"] for d in detections)
    y_off  = bar_h + 20
    for cls_id, name in CLASS_NAMES.items():
        cnt   = counts.get(cls_id, 0)
        color = CLASS_COLORS[cls_id]
        cv2.putText(frame, f"{name}: {cnt}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y_off += 22

    # ── Violation alert banner ────────────────────────────────────────────────
    if violation:
        banner = "⚠  SAFETY VIOLATION DETECTED  ⚠"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        bx = (w - bw) // 2
        by = h - 40
        cv2.rectangle(frame, (bx - 12, by - bh - 8), (bx + bw + 12, by + 10),
                      (0, 0, 200), -1)
        cv2.putText(frame, banner, (bx, by),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Rolling average helper
# ─────────────────────────────────────────────────────────────────────────────
class RollingAvg:
    def __init__(self, window: int = 30):
        self._q = collections.deque(maxlen=window)

    def update(self, v: float):
        self._q.append(v)

    @property
    def value(self) -> float:
        return sum(self._q) / len(self._q) if self._q else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    # ── Load model ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PPE Edge Detection — Phase 3: Live Inference")
    print("=" * 65)
    sess, inp_name, dtype, device = load_model(args.model)

    # ── Open video source ─────────────────────────────────────────────────────
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        sys.exit(1)

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[OK]  Source: {args.source}  ({frame_w}×{frame_h} @ {src_fps:.0f} fps)")

    # ── Optional video writer ─────────────────────────────────────────────────
    writer = None
    if args.save:
        out_name = f"output_{Path(str(args.source)).stem}.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_name, fourcc, src_fps, (frame_w, frame_h))
        print(f"[OK]  Saving to: {out_name}")

    # ── Metric trackers ───────────────────────────────────────────────────────
    avg_fps  = RollingAvg(30)
    avg_pre  = RollingAvg(30)
    avg_post = RollingAvg(30)
    frame_count = 0

    print(f"\n[INFO] Running inference  (conf={args.conf}, iou={args.iou}) …")
    print("       Press  Q  or  ESC  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str):        # end of video file → loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame_count += 1

        # ── Pre-processing ────────────────────────────────────────────────────
        t_pre0 = time.perf_counter()
        blob, scale, pad_x, pad_y = preprocess(frame, args.imgsz, dtype)
        t_pre1 = time.perf_counter()
        pre_ms = (t_pre1 - t_pre0) * 1000

        # ── Inference (pure model time — no rendering) ────────────────────────
        t_inf0 = time.perf_counter()
        raw_out = sess.run(None, {inp_name: blob})
        t_inf1  = time.perf_counter()
        inf_ms  = (t_inf1 - t_inf0) * 1000

        # ── Post-processing / NMS ─────────────────────────────────────────────
        t_post0 = time.perf_counter()
        dets = postprocess(
            raw_out[0],
            frame_h, frame_w,
            scale, pad_x, pad_y,
            args.conf, args.iou
        )
        t_post1 = time.perf_counter()
        post_ms = (t_post1 - t_post0) * 1000

        # ── Update rolling averages ───────────────────────────────────────────
        inf_fps = 1000 / inf_ms if inf_ms > 0 else 0
        avg_fps.update(inf_fps)
        avg_pre.update(pre_ms)
        avg_post.update(post_ms)

        # ── Draw bounding boxes ───────────────────────────────────────────────
        violation = any(d["cls"] in VIOLATION_CLASSES for d in dets)
        for det in dets:
            draw_box(frame, det)

        # ── Draw HUD ──────────────────────────────────────────────────────────
        frame = draw_hud(
            frame,
            fps_inf=avg_fps.value,
            pre_ms=avg_pre.value,
            post_ms=avg_post.value,
            device=device,
            detections=dets,
            conf_thresh=args.conf,
            model_name=args.model,
            violation=violation,
        )

        # ── Console log (every 30 frames) ─────────────────────────────────────
        if frame_count % 30 == 0:
            print(
                f"  Frame {frame_count:5d} | "
                f"FPS: {avg_fps.value:5.1f} | "
                f"Pre: {avg_pre.value:5.1f} ms | "
                f"Post: {avg_post.value:5.1f} ms | "
                f"Dets: {len(dets)}"
            )

        # ── Display & save ────────────────────────────────────────────────────
        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("PPE Edge Inference  [Q / ESC = quit]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\n[INFO] Quit by user.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Session Summary")
    print("=" * 65)
    print(f"  Frames processed  : {frame_count}")
    print(f"  Avg Inference FPS : {avg_fps.value:.1f}   (model time only)")
    print(f"  Avg Pre-proc      : {avg_pre.value:.2f} ms")
    print(f"  Avg Post/NMS      : {avg_post.value:.2f} ms")
    print(f"  Total latency/frm : {1000/avg_fps.value + avg_pre.value + avg_post.value:.2f} ms")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run(args)
