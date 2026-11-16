# Multi-Object Tracking & Directional Counting Pipeline (YOLO11)
# Auto-generated codebase export

import os, time, glob, json, shutil, collections
import numpy as np, pandas as pd, cv2, torch
from ultralytics import YOLO

# Tracking configurations
DEMO_WEIGHTS = "yolo11s.pt"
CLASSES      = None
TRACKER      = "bytetrack.yaml"
LINE_FRAC    = 0.55

def palette(i):
    rng = np.random.RandomState(int(i) * 7 + 3)
    return tuple(int(x) for x in rng.randint(60, 255, 3))

# Refer to GitHub repository for full interactive notebook execution.
