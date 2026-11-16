# Multi-Object Tracking & Directional Counting Pipeline (YOLO11 + ByteTrack / BoT-SORT / DeepSORT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-blueviola.svg)](https://github.com/ultralytics/ultralytics)

An end-to-end production-grade computer vision pipeline designed for real-time multi-object detection, multi-object tracking (MOT), and directional line-crossing counting. This project benchmarks modern tracking frameworks combined with state-of-the-art YOLO11 detectors, featuring model optimization via ONNX Runtime serialization.

---

## 🛠️ Pipeline Architecture

```
  [ Input Stream ] 
         │ 
         ▼
┌──────────────────┐      ┌────────────────────────┐      ┌──────────────────────────┐
│  YOLO11 Detector │ ───> │ Multi-Object Tracker   │ ───> │  Line-Crossing Counter   │
│ (PyTorch / ONNX) │      │ (ByteTrack / BoT-SORT) │      │ (Vector Cross-Product)   │
└──────────────────┘      └────────────────────────┘      └──────────────────────────┘
                                                                       │
                                                                       ▼
                                                             [ Rendered Output MP4 ]
```

1. **Detection Stage**: Classifies and localizes target classes (pedestrians, vehicles) on a frame-by-frame basis.
2. **Tracking & Association Stage**:
   - **ByteTrack**: Utilizes detection boxes and low-confidence boxes via a dual-association strategy to recover heavily occluded objects.
   - **BoT-SORT**: Integrates Camera Motion Compensation (CMC) and Kalman Filter state integration for camera-ego tracking.
   - **DeepSORT**: Couples appearance features extracted from a Re-Identification deep neural network with spatial Kalman filtering.
3. **Analytics & Counting**: Implements stateful tracking history queues to register vector intersections across a dynamic counting boundary line.

---

## 📊 Benchmark & Experimental Evaluation

### 1. Detector Fine-Tuning (VisDrone Dataset)
Fine-tuned on **VisDrone2019-DET** for 10 custom classes across 30 epochs.

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes** | **0.510** | **0.372** | **0.373** | **0.218** |
| Pedestrian | 0.535 | 0.400 | 0.406 | 0.179 |
| Car | 0.697 | 0.781 | 0.776 | 0.531 |
| Motor | 0.521 | 0.444 | 0.434 | 0.184 |
| Bus | 0.685 | 0.477 | 0.514 | 0.367 |

### 2. Multi-Object Tracking Benchmarks (MOT17 Train Subset)
Evaluated under uniform confidence thresholds (`CONF = 0.1`) on 3 primary FRCNN sequences.

| Detector & Tracker Configuration | MOTA (%) | IDF1 (%) | ID Switches (IDSW) | FP | FN | End-to-End FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11s + BoT-SORT** | 41.0% | 52.9% | 214 | 8,742 | 33,643 | 20.8 |
| **YOLO11s + ByteTrack** | 40.3% | 51.4% | 242 | 8,742 | 33,643 | 34.3 |
| **YOLO11n + ByteTrack** | 40.1% | 50.1% | 264 | 8,917 | 33,147 | 35.6 |
| **YOLO11s + DeepSORT** | 36.5% | 50.5% | 461 | 11,200 | 31,450 | 3.5 |

### 3. Inference Engine Acceleration (PyTorch vs ONNX)
Measured on an Intel Xeon CPU + NVIDIA Tesla T4 GPU platform.

```
Inference Throughput (FPS) - Higher is Better
===========================================
YOLO11n PyTorch:  ████████████████ 85.7 FPS
YOLO11n ONNX:     ████████████████████ 105.9 FPS

YOLO11s PyTorch:  ███████████████ 83.4 FPS
YOLO11s ONNX:     ██████████████ 75.2 FPS
```

---

## 🚀 Execution Guide

### ⚙️ Installation & Setup
Install necessary requirements directly via PyPI:
```bash
pip install ultralytics motmetrics deep-sort-realtime lap onnx onnxruntime-gpu opencv-python
```

### 📈 Model Optimization (ONNX Serialization)
Export model weights to ONNX format with simplified graph structure for optimized runtime execution:
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
# Exporting with ONNX Simplifier enabled
onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
```

### 🎥 Running Track and Count Stream
Use the dynamic line counter engine to draw trajectory history trails and capture line-crossing stats:
```python
# Execution command within python runtime
raw_output = make_demo(video="input.mp4", weights="yolo11s.pt", out="output_raw.mp4")
```
Post-process the raw output video using `FFmpeg` to output H.264 encoded stream suitable for web rendering:
```bash
ffmpeg -y -loglevel error -i output_raw.mp4 -vcodec libx264 -pix_fmt yuv420p output_processed.mp4
```

---

## 📁 Saved Results Directory
After building the pipeline, find the runtime outputs archived inside `/content/results.zip`:
- `tracker_comparison.csv` — Comprehensive MOT benchmarks.
- `speed_benchmark.csv` — Performance optimization metrics.
- `conf_sweep.csv` — Threshold sweep analytics for ByteTrack optimization.
- `best_visdrone_yolo11s.pt` — Fine-tuned custom VisDrone detector weights.

---

## 📜 License
Distributed under the **MIT License**. Feel free to use, modify, and distribute for commercial or private projects.
