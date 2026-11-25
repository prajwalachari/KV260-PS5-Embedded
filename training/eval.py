#!/usr/bin/env python3
"""FP32 accuracy of a trained model on the validation or test split.

    python3 eval.py runs/detect/leaky_ft/weights/best.pt val
    python3 eval.py runs/detect/leaky_ft/weights/best.pt test
"""
import sys
from ultralytics import YOLO

weights = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else "val"
m = YOLO(weights).val(data="data.yaml", split=split, imgsz=640, conf=0.001, iou=0.6, plots=True)
print(f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  P {m.box.mp:.3f}  R {m.box.mr:.3f}")
