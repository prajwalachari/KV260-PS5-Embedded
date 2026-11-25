#!/usr/bin/env python3
"""
PCB defect detection on KV260 DPU (PYNQ, Vitis AI 2.5, B4096).
YOLOv8-style head: 3 outputs (strides 8/16/32), 70 ch = 64 DFL box + 6 classes.

Run as root inside pynq-venv:
    python3 pcb_infer_reference.py image.jpg [more images...]
Writes <name>_det.jpg next to each input and prints detections + timings.
"""
import os, sys, time
import numpy as np
import cv2
from pynq_dpu import DpuOverlay

XMODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "kv260", "PCB_YOLO_LEAKY_B4096.xmodel")
CLASSES = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]
IMG = 640
REG_MAX = 16
CONF_THR = 0.25
IOU_THR = 0.45

# ---------- DPU setup ----------
ov = DpuOverlay("dpu.bit")
ov.load_model(XMODEL)
runner = ov.runner
in_t = runner.get_input_tensors()[0]
out_ts = runner.get_output_tensors()

in_shape = tuple(in_t.dims)
in_fix = in_t.get_attr("fix_point")
in_int = "int8" in str(in_t.dtype)          # handle both int8 and float runner buffers
out_info = [(tuple(t.dims), t.get_attr("fix_point")) for t in out_ts]
out_int = "int8" in str(out_ts[0].dtype)

in_buf = [np.empty(in_shape, dtype=np.int8 if in_int else np.float32, order="C")]
out_buf = [np.empty(s, dtype=np.int8 if out_int else np.float32, order="C") for s, _ in out_info]

# ---------- helpers ----------
def letterbox(img, size=IMG, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode(outs):
    boxes, scores, cls_ids = [], [], []
    proj = np.arange(REG_MAX, dtype=np.float32)
    for o in outs:                              # o: (H, W, 70) float
        H, W, _ = o.shape
        stride = IMG / H
        reg = o[..., :4 * REG_MAX].reshape(H, W, 4, REG_MAX)
        reg = np.exp(reg - reg.max(-1, keepdims=True))
        reg /= reg.sum(-1, keepdims=True)
        ltrb = (reg * proj).sum(-1)             # (H, W, 4) in grid units
        cls = sigmoid(o[..., 4 * REG_MAX:])      # (H, W, 6)

        conf = cls.max(-1)
        keep = conf > CONF_THR
        if not keep.any():
            continue
        ys, xs = np.nonzero(keep)
        cx, cy = xs + 0.5, ys + 0.5
        l, t, r, b = [ltrb[ys, xs, i] for i in range(4)]
        boxes.append(np.stack([(cx - l), (cy - t), (cx + r), (cy + b)], 1) * stride)
        scores.append(conf[ys, xs])
        cls_ids.append(cls[ys, xs].argmax(-1))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls_ids)

def nms(boxes, scores, cls_ids):
    if len(boxes) == 0:
        return []
    off = boxes + cls_ids[:, None] * 4096.0      # class-aware NMS via offset
    x1, y1, x2, y2 = off.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < IOU_THR]
    return keep

def run(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[skip] cannot read {path}")
        return
    t0 = time.time()
    lb, r, padx, pady = letterbox(img)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0   # training used RGB, 0-1
    if in_int:
        in_buf[0][0] = np.clip(np.round(x * (2 ** in_fix)), -128, 127).astype(np.int8)
    else:
        in_buf[0][0] = x
    t1 = time.time()
    job = runner.execute_async(in_buf, out_buf)
    runner.wait(job)
    t2 = time.time()
    outs = []
    for buf, (_, fix) in zip(out_buf, out_info):
        o = buf[0].astype(np.float32)
        if out_int:
            o *= 2.0 ** (-fix)
        outs.append(o)
    boxes, scores, cls_ids = decode(outs)
    keep = nms(boxes, scores, cls_ids)
    t3 = time.time()

    h, w = img.shape[:2]
    print(f"\n{path}: {len(keep)} detections | pre {1e3*(t1-t0):.1f} ms, "
          f"DPU {1e3*(t2-t1):.1f} ms, post {1e3*(t3-t2):.1f} ms")
    for i in keep:
        bx = boxes[i].copy()
        bx[[0, 2]] = np.clip((bx[[0, 2]] - padx) / r, 0, w)
        bx[[1, 3]] = np.clip((bx[[1, 3]] - pady) / r, 0, h)
        x1, y1, x2, y2 = bx.astype(int)
        name = CLASSES[cls_ids[i]]
        print(f"  {name:16s} {scores[i]:.2f}  [{x1},{y1},{x2},{y2}]")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{name} {scores[i]:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out_path = path.rsplit(".", 1)[0] + "_det.jpg"
    cv2.imwrite(out_path, img)
    print(f"  saved {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 pcb_infer_reference.py img1.jpg [img2.jpg ...]")
        sys.exit(1)
    print(f"input fix_point={in_fix} ({'int8' if in_int else 'float'}), outputs={out_info}")
    for p in sys.argv[1:]:
        run(p)
