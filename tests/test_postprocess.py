"""Board-free tests for post-processing, severity, logging and advice.

    python3 -m pytest tests
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "kv260_app"), os.path.join(ROOT, "advisory_llm"), os.path.join(ROOT, "model", "export")]

import pcb_core  # noqa: E402
from severity import board_severity  # noqa: E402
from defect_logger import DefectLogger  # noqa: E402
import advisor  # noqa: E402


def test_lut_matches_formula():
    lut = pcb_core.make_lut(6).view(np.int8)
    for p in (0, 1, 128, 200, 255):
        assert lut[p] == min(127, round(p / 255 * 64))


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 140, 140]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    cls = np.array([3, 3, 3])
    assert sorted(pcb_core.nms(boxes, scores, cls).tolist()) == [0, 2]


def test_nms_is_class_aware():
    boxes = np.array([[10, 10, 50, 50], [10, 10, 50, 50]], np.float32)
    keep = pcb_core.nms(boxes, np.array([0.9, 0.8], np.float32), np.array([0, 1]))
    assert len(keep) == 2


def test_decode_single_cell():
    o = np.full((20, 20, 70), -128, np.int8)        # fix point 1: every class logit = -64
    o[5, 7, 64 + 3] = 4                              # short, logit 2.0 -> confidence 0.88
    o[5, 7, :64] = 0                                 # flat DFL bins -> 7.5 cells per side
    boxes, scores, cls = pcb_core.decode_fast([o], [1], 0.25)
    assert len(scores) == 1 and cls[0] == 3 and abs(scores[0] - 0.881) < 0.01
    cx = (7 + 0.5) * 32
    assert abs((boxes[0, 0] + boxes[0, 2]) / 2 - cx) < 1e-3


def test_severity_and_advice():
    d = [{"cls": "spur", "conf": 0.6, "box": ""}, {"cls": "short", "conf": 0.8, "box": ""}]
    assert board_severity(d) == "critical"
    assert board_severity([]) == "none"
    assert advisor.rule_advice(d).startswith("short")
    assert "Pass" in advisor.rule_advice([])


def test_logger_writes_row(tmp_path):
    path = tmp_path / "log.csv"
    DefectLogger(str(path)).log("a.jpg", "FAIL", "major", [{"cls": "missing_hole", "conf": 0.82, "box": "[1,2,3,4]"}], "33", "15")
    rows = list(csv.DictReader(open(path)))
    assert rows[0]["verdict"] == "FAIL" and rows[0]["n_defects"] == "1"


def test_model_output_shapes():
    try:
        import torch
    except ImportError:
        return
    from vai_yolov8_leaky import PCB_YOLO
    with torch.no_grad():
        outs = PCB_YOLO().eval()(torch.zeros(1, 3, 640, 640))
    assert [tuple(o.shape) for o in outs] == [(1, 70, 80, 80), (1, 70, 40, 40), (1, 70, 20, 20)]
