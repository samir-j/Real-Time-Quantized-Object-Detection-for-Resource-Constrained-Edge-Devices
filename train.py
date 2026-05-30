"""
train.py  ─  Phase 1: Baseline FP32 Training
=============================================
Trains YOLOv8n on the PPE / Hard Hat Workers dataset at full FP32 precision.

Usage:
    python train.py                        # GPU training (recommended)
    python train.py --device cpu           # CPU fallback
    python train.py --epochs 100 --batch 32

Requirements:
    pip install ultralytics torch torchvision
"""

import argparse
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLOv8n (FP32) on PPE detection dataset"
    )
    p.add_argument("--data",    type=str, default="ppe.yaml",
                   help="Path to dataset YAML (default: data/ppe.yaml)")
    p.add_argument("--epochs",  type=int, default=50,
                   help="Number of training epochs (default: 50)")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Input image size in pixels (default: 640)")
    p.add_argument("--batch",   type=int, default=16,
                   help="Batch size (default: 16; reduce to 8 if OOM)")
    p.add_argument("--device",  type=str, default="0",
                   help="Device: '0' for GPU 0, 'cpu', '0,1' for multi-GPU")
    p.add_argument("--workers", type=int, default=8,
                   help="Dataloader workers (default: 8)")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience — stop if no mAP improvement for N epochs (default: 10)")
    p.add_argument("--project", type=str, default="runs/train",
                   help="Output project directory")
    p.add_argument("--name",    type=str, default="yolov8n_ppe_fp32",
                   help="Run name (subfolder inside --project)")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("        Run: pip install ultralytics torch torchvision")
        sys.exit(1)

    # ── Auto-detect device (graceful GPU → CPU fallback) ─────────────────────
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU.")
        device = "cpu"

    # ── Device report ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PPE Edge Detection — Phase 1: FP32 Baseline Training")
    print("=" * 65)
    if device != "cpu":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU  : {gpu}  ({vram:.1f} GB VRAM)")
    else:
        print("  GPU  : NOT AVAILABLE — running on CPU (slow!)")
    print(f"  Early stop : patience={args.patience} epochs")
    print(f"  Data : {args.data}")
    print(f"  Run  : {args.project}/{args.name}")
    print("=" * 65 + "\n")

    # ── Load pretrained YOLOv8n backbone ─────────────────────────────────────
    # 'yolov8n.pt' is the nano variant (~6 MB) — perfect for edge devices
    model = YOLO("yolov8n.pt")
    print(f"[INFO] Model parameters : {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"[INFO] Precision        : FP32 (baseline)\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Early stopping
        patience=args.patience,  # stop if mAP doesn't improve for N epochs
        # Augmentation (standard Mosaic + flips)
        mosaic=1.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # Regularisation
        weight_decay=0.0005,
        dropout=0.0,
        # Logging
        plots=True,
        save=True,
        save_period=10,          # checkpoint every 10 epochs
        val=True,
        verbose=True,
        # Precision — keep FP32 for baseline
        half=False,
    )
    elapsed = time.time() - t0

    # ── Post-training report ──────────────────────────────────────────────────
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    last_weights = Path(args.project) / args.name / "weights" / "last.pt"

    print("\n" + "=" * 65)
    print("  Training Complete")
    print("=" * 65)
    print(f"  Total time   : {elapsed/3600:.2f} h")
    print(f"  Best weights : {best_weights}")
    print(f"  Last weights : {last_weights}")

    # Key metrics from final validation
    metrics = results.results_dict
    map50    = metrics.get("metrics/mAP50(B)",    0.0)
    map5095  = metrics.get("metrics/mAP50-95(B)", 0.0)
    prec     = metrics.get("metrics/precision(B)", 0.0)
    rec      = metrics.get("metrics/recall(B)",    0.0)

    print(f"\n  ┌─ Final Validation Metrics (FP32 Baseline) ─────────┐")
    print(f"  │  Precision  : {prec:.4f}                              │")
    print(f"  │  Recall     : {rec:.4f}                              │")
    print(f"  │  mAP@0.5    : {map50:.4f}                              │")
    print(f"  │  mAP@.5:.95 : {map5095:.4f}  ← benchmark this!         │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print("\n  Next step → Run: python scripts/export_onnx.py")
    print("=" * 65 + "\n")

    return str(best_weights)


# ─────────────────────────────────────────────────────────────────────────────
# Validation-only mode (re-evaluate a saved model)
# ─────────────────────────────────────────────────────────────────────────────
def validate_only(weights: str, data: str, imgsz: int = 640, device: str = "0"):
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device, half=False)
    print(f"\n  mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  mAP@.5:.95  : {metrics.box.map:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    best = train(args)
    print(f"\n[DONE] Best weights saved → {best}")
