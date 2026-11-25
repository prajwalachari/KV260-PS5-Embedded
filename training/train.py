#!/usr/bin/env python3
"""Training in two stages (Ultralytics 8.4, PyTorch 2.x, one GPU).

    python3 train.py base        # YOLOv8s with SiLU, 100 epochs  -> runs/detect/base
    python3 train.py leaky       # swap SiLU for LeakyReLU(26/256), fine-tune from base -> runs/detect/leaky_ft

The DPU's LeakyReLU has a fixed slope of 26/256 (= 0.1015625). Training with exactly that slope
means the PC model and the board compute the same function.
"""
import sys
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv

stage = sys.argv[1] if len(sys.argv) > 1 else "base"

if stage == "base":
    YOLO("yolov8s.pt").train(data="data.yaml", epochs=100, imgsz=640, batch=16, name="base")

elif stage == "leaky":
    Conv.default_act = nn.LeakyReLU(26 / 256, inplace=True)        # every Conv block built from here on
    model = YOLO("yolov8s.yaml")                                   # rebuild the graph with LeakyReLU
    model.load("runs/detect/base/weights/best.pt")                 # start from the trained SiLU weights
    model.train(data="data.yaml", epochs=80, imgsz=640, batch=16, lr0=0.002, name="leaky_ft")
