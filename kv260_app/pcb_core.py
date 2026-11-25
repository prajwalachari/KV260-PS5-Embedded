"""
pcb_core.py - fast PCB defect detector for the KV260 DPU (PYNQ, Vitis AI 2.5, B4096).

Same model and same results as pcb_infer_reference.py, but much less CPU work:
  * pre-process: letterbox once, then one cv2.LUT call turns pixels straight into
    the DPU's INT8 values (no float image, no divide by 255).
  * post-process: the confidence threshold is checked on the raw INT8 class
    outputs first (sigmoid is monotonic, so  sigmoid(x) > 0.25  <=>  x > logit(0.25)).
    Only the few cells that pass get dequantized, softmaxed (DFL) and decoded,
    instead of all 8,400 cells x 70 channels.
"""
import math
import os
import time
import numpy as np
import cv2

XMODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "kv260", "PCB_YOLO_LEAKY_B4096.xmodel")
CLASSES = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]
COLORS = [(255, 64, 64), (0, 200, 255), (0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]  # BGR
IMG = 640
REG_MAX = 16
NC = len(CLASSES)


# ---------------------------------------------------------------- pre-process
def make_lut(fix):
    """pixel p (0..255) -> round(p/255 * 2^fix), clipped to int8. Stored as uint8 for cv2.LUT."""
    v = np.clip(np.round(np.arange(256) / 255.0 * (2 ** fix)), -128, 127).astype(np.int8)
    return v.view(np.uint8)


class Letterbox:
    """Re-uses one 640x640 grey canvas; only the resize + copy happen per frame."""

    def __init__(self, size=IMG, color=114):
        self.size, self.color = size, color
        self.canvas = np.full((size, size, 3), color, np.uint8)
        self.key = None

    def __call__(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        if self.key != (h, w):              # new frame size: reset the border once
            self.canvas[:] = self.color
            self.key = (h, w)
        self.canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return self.canvas, r, left, top


def preprocess(img_bgr, lb, lut, dst_int8):
    """BGR frame -> letterboxed RGB INT8 written straight into the DPU input buffer."""
    canvas, r, padx, pady = lb(img_bgr)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    dst_int8[...] = cv2.LUT(rgb, lut).view(np.int8)
    return r, padx, pady


# --------------------------------------------------------------- post-process
def decode_fast(outs_q, fixes, conf_thr=0.25):
    """outs_q: list of int8 arrays (H, W, 70) straight from the DPU; fixes: their fix points."""
    proj = np.arange(REG_MAX, dtype=np.float32)
    logit = math.log(conf_thr / (1 - conf_thr))
    boxes, scores, cls_ids = [], [], []
    for o, fix in zip(outs_q, fixes):
        H, W, _ = o.shape
        stride = IMG / H
        q_thr = math.floor(logit * 2 ** fix)            # x > logit  <=>  q > floor(logit * 2^fix)
        cls_q = o[..., 4 * REG_MAX:]
        best_q = cls_q.max(-1)                          # int8 max, very cheap
        ys, xs = np.nonzero(best_q > q_thr)
        if ys.size == 0:
            continue
        scale = 2.0 ** (-fix)
        sel = o[ys, xs].astype(np.float32) * scale      # only the surviving cells, (N, 70)
        cls_logit = sel[:, 4 * REG_MAX:]
        cid = cls_logit.argmax(1)
        conf = 1.0 / (1.0 + np.exp(-cls_logit[np.arange(len(cid)), cid]))
        reg = sel[:, :4 * REG_MAX].reshape(-1, 4, REG_MAX)
        reg = np.exp(reg - reg.max(-1, keepdims=True))
        ltrb = (reg * proj).sum(-1) / reg.sum(-1)       # (N, 4) in grid units
        cx, cy = xs + 0.5, ys + 0.5
        boxes.append(np.stack([cx - ltrb[:, 0], cy - ltrb[:, 1], cx + ltrb[:, 2], cy + ltrb[:, 3]], 1) * stride)
        scores.append(conf)
        cls_ids.append(cid)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls_ids)


def nms(boxes, scores, cls_ids, iou_thr=0.45):
    """Class-aware NMS (offset trick). Usually only a handful of boxes reach here."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    off = boxes + cls_ids[:, None] * 4096.0
    x1, y1, x2, y2 = off.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) < iou_thr]
    return np.array(keep, int)


def to_image_coords(boxes, r, padx, pady, w, h):
    b = boxes.copy()
    b[:, [0, 2]] = np.clip((b[:, [0, 2]] - padx) / r, 0, w)
    b[:, [1, 3]] = np.clip((b[:, [1, 3]] - pady) / r, 0, h)
    return b.astype(int)


def draw(img, dets):
    for (x1, y1, x2, y2), s, c in dets:
        col = COLORS[c % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{CLASSES[c]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yt = max(y1, th + 4)
        cv2.rectangle(img, (x1, yt - th - 4), (x1 + tw + 4, yt), col, -1)
        cv2.putText(img, label, (x1 + 2, yt - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ detector
class Detector:
    def __init__(self, xmodel=XMODEL, bitfile="dpu.bit", conf=0.25, iou=0.45):
        from pynq_dpu import DpuOverlay          # imported here so the maths can be tested off-board
        self.ov = DpuOverlay(bitfile)
        self.ov.load_model(xmodel)
        self.runner = self.ov.runner
        it = self.runner.get_input_tensors()[0]
        ots = self.runner.get_output_tensors()
        self.in_fix = it.get_attr("fix_point")
        self.out_fix = [t.get_attr("fix_point") for t in ots]
        self.in_buf = [np.empty(tuple(it.dims), np.int8, order="C")]
        self.out_buf = [np.empty(tuple(t.dims), np.int8, order="C") for t in ots]
        self.lut = make_lut(self.in_fix)
        self.lb = Letterbox()
        self.conf, self.iou = conf, iou

    def __call__(self, img):
        """Returns (detections, timings_ms). detections = [((x1,y1,x2,y2), score, class_id), ...]"""
        t0 = time.perf_counter()
        r, px, py = preprocess(img, self.lb, self.lut, self.in_buf[0][0])
        t1 = time.perf_counter()
        self.runner.wait(self.runner.execute_async(self.in_buf, self.out_buf))
        t2 = time.perf_counter()
        boxes, scores, cids = decode_fast([b[0] for b in self.out_buf], self.out_fix, self.conf)
        keep = nms(boxes, scores, cids, self.iou)
        h, w = img.shape[:2]
        b = to_image_coords(boxes[keep], r, px, py, w, h) if len(keep) else []
        dets = [(tuple(bb), float(scores[k]), int(cids[k])) for bb, k in zip(b, keep)]
        t3 = time.perf_counter()
        return dets, {"pre": 1e3 * (t1 - t0), "dpu": 1e3 * (t2 - t1), "post": 1e3 * (t3 - t2)}
