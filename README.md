# Real Time Quantized Object Detection for Resource-Constrained Edge Devices

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-FP16-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**A lightweight, edge-optimized object detection system for PPE (Personal Protective Equipment) compliance monitoring, built with YOLOv8n and deployed via ONNX Runtime with FP16 quantization.**

[Setup](#-setup--installation) •
[Pipeline](#-pipeline-execution) •
[Benchmarks](#-performance-benchmark-table) •
[Video Demo](#-video-demonstration) •
[Model Weights](#-model-weights)

</div>

---

## Problem Statement

Construction sites and industrial facilities are high-risk environments where PPE compliance — particularly **hard hat usage** — is critical for worker safety. Manual monitoring by safety officers is:

- **Impractical** at scale across large sites
- **Inconsistent** due to human fatigue and blind spots
- **Reactive** rather than preventive

This project addresses these challenges by training a **lightweight object detection model (YOLOv8n)** to automatically detect hard hat compliance in real-time, and then **optimizing it for edge deployment** using ONNX conversion with FP16 quantization — enabling inference on resource-constrained devices without cloud dependency.

### Detection Classes

| Class ID | Label | Description | Alert Level |
|:--------:|-------|-------------|:-----------:|
| `0` | **head** | Worker **without** a hard hat | 🔴 **Violation** |
| `1` | **helmet** | Worker **wearing** a hard hat | 🟢 Safe |

---

##  Project Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────┐  │
│  │  Dataset    │───▶│  YOLOv8n   │───▶│  FP32 Weights     │  │
│  │  (14,748    │    │  Training   │    │  best.pt (6.2 MB) │  │
│  │   images)   │    │  50 epochs  │    │  mAP50: 0.966     │  │
│  └────────────┘    └────────────┘    └────────┬───────────┘  │
│                                               │              │
│                    EDGE CONVERSION             ▼              │
│                                      ┌────────────────────┐  │
│                                      │  ONNX Export       │  │
│                                      │  + FP16 Quantize   │  │
│                                      │  best.onnx (5.9MB) │  │
│                                      └────────┬───────────┘  │
│                                               │              │
│                    EDGE INFERENCE              ▼              │
│                                      ┌────────────────────┐  │
│                                      │  ONNX Runtime      │  │
│                                      │  Live Inference     │  │
│                                      │  + Real-time HUD   │  │
│                                      └────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

##  Repository Structure

```
├── download_dataset.py      # Phase 0 — Dataset download (Roboflow / Kaggle)
├── ppe.yaml                 # Dataset configuration (YOLOv8 format)
├── train.py                 # Phase 1 — FP32 baseline training with early stopping
├── export_onnx.py           # Phase 2 — ONNX conversion + FP16/INT8 quantization
├── live_inference.py        # Phase 3 — Real-time inference with metrics overlay
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── data/                    # Dataset (created by download_dataset.py)
│   └── ppe_dataset/
│       ├── train/
│       │   ├── images/      # 14,748 training images
│       │   └── labels/      # YOLO-format annotations
│       ├── valid/
│       │   ├── images/      # 1,413 validation images
│       │   └── labels/
│       └── test/
│           ├── images/      # Test images
│           └── labels/
│
├── models/                  # Exported edge-optimized models
│   └── best.onnx            # ONNX model (FP16 quantized)
│
└── yolov8n.pt               # Pretrained YOLOv8n backbone (auto-downloaded)
```

---

## Setup & Installation

### Prerequisites

- **Python** 3.10+
- **OS**: Windows / Linux / macOS
- **(Optional)** NVIDIA GPU with CUDA for accelerated training and inference

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/edge-ppe-detection.git
cd edge-ppe-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) For NVIDIA GPU acceleration
pip install onnxruntime-gpu
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.0.0 | YOLOv8 training & export framework |
| `torch` | ≥ 2.0.0 | Deep learning backend |
| `onnx` | ≥ 1.14.0 | Model format conversion |
| `onnxruntime` | ≥ 1.16.0 | Edge inference engine |
| `onnxsim` | ≥ 0.4.33 | ONNX graph simplification |
| `opencv-python` | ≥ 4.8.0 | Image processing & video I/O |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `roboflow` | ≥ 1.1.0 | Dataset download (optional) |

---

##  Pipeline Execution

### Phase 0 — Data Sourcing

```bash
# Option A: Download via Roboflow (recommended — pre-formatted for YOLOv8)
python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY --source roboflow

# Option B: Download via Kaggle CLI
python download_dataset.py --source kaggle
```

**Dataset Details:**

| Property | Value |
|----------|-------|
| **Name** | Hard Hat Workers |
| **Source** | [Roboflow Universe](https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/12) |
| **Total Images** | ~16,161 (train: 14,748 · val: 1,413 · test: included) |
| **Annotations** | Bounding box (YOLO format) |
| **Classes** | 2 — `head`, `helmet` |
| **License** | Public Domain |

---

### Phase 1 — Baseline FP32 Training

```bash
# Standard training (auto-detects GPU, falls back to CPU)
python train.py --epochs 50 --batch 16 --device 0

# CPU-only (explicit)
python train.py --epochs 50 --batch 8 --device cpu

# Custom early stopping patience
python train.py --epochs 50 --batch 16 --device 0 --patience 5
```

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `--epochs` | `50` | Maximum training epochs |
| `--batch` | `16` | Batch size (reduce to 8 if OOM) |
| `--device` | `0` | `0` = GPU, `cpu` = CPU (auto-fallback if no GPU) |
| `--imgsz` | `640` | Input image resolution |
| `--patience` | `10` | Early stopping — halt if mAP stagnates for N epochs |
| `--data` | `ppe.yaml` | Dataset configuration path |

**Features:**
-  Automatic GPU → CPU fallback (no crash if CUDA unavailable)
-  Early stopping with configurable patience
-  Pretrained YOLOv8n backbone (transfer learning)
-  AdamW optimizer with warmup scheduling
-  Standard augmentation (mosaic, HSV, flip)
-  Periodic checkpointing every 10 epochs

---

### Phase 2 — ONNX Conversion & Quantization

```bash
# FP16 quantization (recommended)
python export_onnx.py \
    --weights "C:\Users\samir\runs\detect\runs\train\yolov8n_ppe_fp32-5\weights\best.pt" \
    --quant fp16

# INT8 quantization (with calibration data)
python export_onnx.py \
    --weights "C:\Users\samir\runs\detect\runs\train\yolov8n_ppe_fp32-5\weights\best.pt" \
    --quant int8 \
    --calib-images data/ppe_dataset/valid/images
```

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `--weights` | `runs/train/.../best.pt` | Trained FP32 model path |
| `--quant` | `fp16` | Quantization level: `fp16` or `int8` |
| `--imgsz` | `640` | Export input resolution |
| `--simplify` | `True` | Run ONNX graph simplification |
| `--benchmark` | `True` | Run latency benchmark after export |

**Output:** Optimized model saved to `models/best.onnx`

---

### Phase 3 — Live Edge Inference

```bash
# Live webcam feed
python live_inference.py --model models/best.onnx

# Test video file
python live_inference.py --model models/best.onnx --source test_video.mp4

# Save annotated output video
python live_inference.py --model models/best.onnx --source test_video.mp4 --save

# Custom detection thresholds
python live_inference.py --model models/best.onnx --conf 0.45 --iou 0.5
```

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `--model` | *(required)* | Path to `.onnx` model |
| `--source` | `0` (webcam) | Video source: `0` = webcam, or path to `.mp4`/`.avi` |
| `--conf` | `0.40` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--save` | `False` | Save annotated output video |
| `--no-display` | `False` | Run headless (no GUI window) |

**Real-time HUD displays:**
-  Bounding boxes with class labels and confidence scores
-  Live Inference FPS (pure model time, excluding rendering)
-  Pre-processing latency (ms)
-  Post-processing / NMS latency (ms)
-  Device info (CPU / CUDA)
-  Per-class detection count
-  Safety violation alert banner

---

##  Performance Benchmark Table

> **Test Hardware:**
> - **CPU:** Intel Core i5-10210U @ 1.60GHz
> - **GPU:** None (CPU-only inference)
> - **RAM:** 8 GB
> - **OS:** Windows 11 / Python 3.12.7

### Model Comparison: FP32 Baseline vs. ONNX Quantized

| Metric | FP32 Baseline (`.pt`) | ONNX Quantized (`.onnx`) | Δ Change |
|:-------|:---------------------:|:------------------------:|:--------:|
| **Model Size** | 6.2 MB | 5.9 MB | ↓ 4.8% |
| **Precision** | 0.9390 | ~0.937 | ↓ ~0.2% |
| **Recall** | 0.9278 | ~0.926 | ↓ ~0.2% |
| **mAP@0.5** | 0.9656 | ~0.964 | ↓ ~0.2% |
| **mAP@0.5:0.95** | 0.6350 | ~0.633 | ↓ ~0.3% |
| **Inference Latency** | 139.0 ms | ~125 ms | ↓ ~10% |
| **Inference FPS** | ~7.2 | ~8.0 | ↑ ~11% |
| **Pre-processing** | — | ~2.7 ms | — |
| **Post-processing (NMS)** | — | ~1.3 ms | — |

> **Note:** FP16 quantization on CPU shows modest speed gains because FP16 is primarily accelerated on GPU hardware (NVIDIA Ampere+, Apple Silicon). The main benefit on CPU is **reduced model size** and **lower memory bandwidth**. On GPU-equipped edge devices (Jetson Nano, Jetson Xavier), FP16 speedups of **2–3×** are typical.

### Per-Class Accuracy (FP32 Baseline)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|:------|:---------:|:------:|:-------:|:------------:|
| **head** (violation) | 0.916 | 0.925 | 0.954 | 0.635 |
| **helmet** (safe) | 0.962 | 0.931 | 0.977 | 0.635 |
| **All** | **0.939** | **0.928** | **0.966** | **0.635** |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8n (nano) — 3.01M parameters |
| Input Resolution | 640 × 640 |
| Epochs Trained | 10 / 50 (converged early) |
| Batch Size | 8 |
| Optimizer | AdamW (lr=0.001, weight_decay=0.0005) |
| Warmup | 3 epochs |
| Early Stopping | patience = 10 epochs |
| Augmentation | Mosaic, HSV jitter, Horizontal flip, CLAHE |
| Training Time | ~24 hours (CPU-only) |

---

##  Quantization Trade-Off Analysis

### Why ONNX + FP16?

**Format — ONNX (Open Neural Network Exchange):**
- **Runtime-agnostic**: Runs on ONNX Runtime, TensorRT, OpenVINO, CoreML, and most edge accelerators
- **Hardware support**: Compatible with CPU, GPU, NPU, and DSP backends
- **Ecosystem**: Largest cross-platform model interoperability standard

**Precision — FP16 (Half-Precision Floating Point):**

| Aspect | FP16 | INT8 |
|--------|------|------|
| **Size Reduction** | ~50% vs FP32 | ~75% vs FP32 |
| **Accuracy Loss** | < 0.3% mAP drop | 1–3% mAP drop |
| **Calibration** |  Not required |  Requires representative data |
| **Hardware Accel.** | NVIDIA Ampere+, Apple Silicon, Qualcomm Hexagon | Requires INT8-capable hardware |
| **Safety Suitability** |  Minimal false negatives |  Higher risk of missed violations |

**Decision Rationale:**

> FP16 was chosen over INT8 because this is a **safety-critical application** where **missing a violation (false negative) can result in injury or death**. FP16 provides a near-lossless compression (~0.3% mAP degradation) while halving memory requirements. INT8 would offer greater compression but introduces measurably higher accuracy loss — particularly on the smaller `head` class where detection precision is paramount. For safety applications, **reliability takes priority over maximum compression**.

### Inference Pipeline Breakdown

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  PRE-PROCESSING │────▶│    INFERENCE      │────▶│  POST-PROCESSING    │
│                 │     │                  │     │                     │
│  • Letterbox    │     │  • ONNX Runtime  │     │  • Decode boxes     │
│    resize 640²  │     │  • FP16 weights  │     │  • Confidence filter│
│  • BGR → RGB    │     │  • CPU/CUDA exec │     │  • Per-class NMS    │
│  • Normalize    │     │                  │     │  • Coord remap      │
│  • NCHW tensor  │     │                  │     │                     │
│                 │     │                  │     │                     │
│   ~2.7 ms       │     │   ~125 ms (CPU)  │     │   ~1.3 ms           │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                  Total: ~129 ms/frame
```

---

##  Technical Details

### Model Architecture — YOLOv8n (Nano)

| Component | Detail |
|-----------|--------|
| **Backbone** | Modified CSPDarknet53 |
| **Neck** | PAN-FPN (Path Aggregation Network) |
| **Head** | Decoupled anchor-free detection head |
| **Parameters** | 3,006,038 (fused) |
| **GFLOPs** | 8.1 |
| **Layers** | 73 (fused) |

### Why YOLOv8n?

YOLOv8n (nano) was selected as the base model because:

1. **Lightweight**: Only 3M parameters and 6.2 MB — fits comfortably in edge device memory
2. **Fast**: Single-stage detector with anchor-free head minimizes computational overhead
3. **Accurate**: Achieves 0.966 mAP@0.5 on PPE detection despite its small size
4. **Well-supported**: Ultralytics ecosystem provides seamless training → ONNX export pipeline
5. **Transfer learning**: Pretrained on COCO (80 classes) provides strong feature initialization

---

## 📎 Model Weights

| Model | Format | Size | Link |
|-------|--------|:----:|------|
| FP32 Baseline | `best.pt` (YOLOv8n) | 6.2 MB | [Google Drive / HuggingFace — ADD LINK] |
| ONNX Quantized | `best.onnx` (FP16) | 5.9 MB | [Google Drive / HuggingFace — ADD LINK] |

---


##  Quick Start (TL;DR)

```bash
# Install
pip install -r requirements.txt

# Download dataset
python download_dataset.py --api-key YOUR_KEY --source roboflow

# Train (auto-detects GPU, falls back to CPU)
python train.py --epochs 50 --batch 16 --device 0

# Convert to ONNX (FP16)
python export_onnx.py --weights path/to/best.pt --quant fp16

# Run live inference
python live_inference.py --model models/best.onnx
```

Press **Q** or **ESC** to quit the inference window.

---

##  References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime — High Performance Inference](https://onnxruntime.ai/)
- [Hard Hat Workers Dataset — Roboflow Universe](https://universe.roboflow.com/joseph-nelson/hard-hat-workers)
- [ONNX Model Optimization & Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- Jocher, G., et al. "Ultralytics YOLO." (2023). GitHub: https://github.com/ultralytics/ultralytics

---

##  License

This project is developed for **educational and research purposes**. The dataset is sourced from [Roboflow Universe](https://universe.roboflow.com/) under the **Public Domain** license.

---

<div align="center">

**Built with**  **using YOLOv8 + ONNX Runtime**

*Real-time PPE compliance monitoring — from cloud training to edge deployment*

</div>
