#!/usr/bin/env python3
"""Checks and fixes we needed on the PKU-Market-PCB YOLO export before training.

    python3 dataset_prep.py ../pcb_dataset

1. Deletes Ultralytics label caches (a stale cache once made a run train on a single class).
2. Renames label files *_256.txt to *_600.txt when the image is *_600.jpg
   (25% of labels had the wrong name, so those images were treated as defect-free).
3. Prints images without labels and the number of boxes per class for each split.
"""
import os
import sys
from collections import Counter
from pathlib import Path

NAMES = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]
root = Path(sys.argv[1] if len(sys.argv) > 1 else "../pcb_dataset")

for cache in root.rglob("*.cache"):
    cache.unlink()
    print("deleted", cache)

for split in ("train", "val", "test"):
    img_dir, lab_dir = root / split / "images", root / split / "labels"
    if not img_dir.exists():
        continue
    renamed = 0
    for lab in lab_dir.glob("*_256.txt"):
        target = lab.with_name(lab.name.replace("_256.txt", "_600.txt"))
        if (img_dir / target.with_suffix(".jpg").name).exists() and not target.exists():
            lab.rename(target)
            renamed += 1
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    missing = [p.name for p in images if not (lab_dir / (p.stem + ".txt")).exists()]
    counts = Counter()
    for lab in lab_dir.glob("*.txt"):
        for line in lab.read_text().split("\n"):
            if line.strip():
                counts[int(line.split()[0])] += 1
    print(f"\n{split}: {len(images)} images, {renamed} labels renamed, {len(missing)} without labels")
    for i, n in enumerate(NAMES):
        print(f"  {n:16s} {counts[i]}")
