"""Append one row per inspected board to a CSV file."""
import csv
import os
import time

FIELDS = ["time", "image", "verdict", "severity", "n_defects", "defects", "dpu_ms", "cycle_s", "advice"]


class DefectLogger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, FIELDS).writeheader()

    def log(self, image, verdict, severity, defects, dpu_ms="", cycle_s="", advice=""):
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": os.path.basename(image),
            "verdict": verdict,
            "severity": severity,
            "n_defects": len(defects),
            "defects": "; ".join(f"{d['cls']} {d['conf']:.2f} {d['box']}" for d in defects),
            "dpu_ms": dpu_ms,
            "cycle_s": cycle_s,
            "advice": advice.replace("\n", " "),
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        return row
