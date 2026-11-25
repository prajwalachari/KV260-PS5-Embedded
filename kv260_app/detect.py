#!/usr/bin/env python3
"""Run the INT8 YOLOv8s model on the KV260 DPU for one or more images.

    sudo -i
    source /etc/profile.d/pynq_venv.sh
    python3 detect.py image.jpg [more.jpg ...]       # writes <name>_det.jpg next to each image
    python3 detect.py --bench 50 image.jpg            # average timing over 50 runs
"""
import argparse
import os
import numpy as np
import cv2
from pcb_core import Detector, CLASSES, draw

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XMODEL = os.environ.get("PCB_XMODEL", os.path.join(HERE, "..", "model", "kv260", "PCB_YOLO_LEAKY_B4096.xmodel"))

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--xmodel", default=DEFAULT_XMODEL)
ap.add_argument("--bench", type=int, default=0, help="repeat each image N times and report averages")
a = ap.parse_args()

det = Detector(xmodel=a.xmodel, conf=a.conf)
print(f"input fix_point={det.in_fix}, output fix_points={det.out_fix}")

for p in a.images:
    img = cv2.imread(p)
    if img is None:
        print(f"[skip] cannot read {p}")
        continue
    det(img)                                   # warm-up
    n = max(a.bench, 1)
    ts = [det(img)[1] for _ in range(n)]
    dets, _ = det(img)
    avg = {k: np.mean([t[k] for t in ts]) for k in ts[0]}
    total = sum(avg.values())
    print(f"\n{p}: {len(dets)} detections | pre {avg['pre']:.1f} ms, DPU {avg['dpu']:.1f} ms, "
          f"post {avg['post']:.1f} ms | total {total:.1f} ms ({1000 / total:.1f} FPS)" + (f"  [avg of {n}]" if n > 1 else ""))
    for (x1, y1, x2, y2), s, c in dets:
        print(f"  {CLASSES[c]:16s} {s:.2f}  [{x1},{y1},{x2},{y2}]")
    out = p.rsplit(".", 1)[0] + "_det.jpg"
    cv2.imwrite(out, draw(img, dets))
    print(f"  saved {out}")
