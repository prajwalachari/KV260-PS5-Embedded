"""
Converts the fine-tuned Ultralytics checkpoint for Vitis AI and checks the conversion.

Run on the training PC (needs ultralytics), from the project folder:
    python3 make_leaky_state_dict.py

1. Builds PCB_YOLO (vai_yolov8_leaky.py) and loads the LeakyReLU checkpoint into it.
2. Feeds the same input to both models and compares the raw head outputs per scale.
   Any weight-mapping or activation mismatch stops the script before quantization.
3. Saves the weights as a plain .npz, which the older PyTorch inside the Vitis AI 2.5 Docker can read.
"""
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from vai_yolov8_leaky import PCB_YOLO, load_ultralytics_weights

SRC = "runs/detect/leaky_ft/weights/best.pt"
DST = "deployment_candidate/pcb_leaky_vai_weights.npz"

# ---- standalone model ----
model = PCB_YOLO().eval()
ret = load_ultralytics_weights(model, SRC)
if isinstance(ret, nn.Module):
    model = ret.eval()

acts = {type(m).__name__ for m in model.modules()
        if any(k in type(m).__name__.lower() for k in ("relu", "swish", "silu", "sigmoid"))}
print("standalone activations:", acts)

# ---- reference: Ultralytics raw head outputs (cv2 box + cv3 cls per scale) ----
um = YOLO(SRC).model.float().eval()
det = um.model[-1]
captured = {}
det.register_forward_hook(lambda mod, inp, out: captured.__setitem__("feats", inp[0]))

torch.manual_seed(0)
x = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    um(x)
    feats = captured["feats"]
    ref = [torch.cat((det.cv2[i](feats[i]), det.cv3[i](feats[i])), 1) for i in range(len(feats))]
    out = model(x)

outs = list(out) if isinstance(out, (list, tuple)) else [out]
print("ultralytics head shapes:", [tuple(r.shape) for r in ref])
print("standalone output shapes:", [tuple(o.shape) for o in outs])

ok = True
for r in ref:
    match = [o for o in outs if o.numel() == r.numel()]
    if not match:
        print("  no standalone output matching shape", tuple(r.shape)); ok = False; continue
    o = match[0].reshape(r.shape) if match[0].shape == r.shape else match[0].permute(0, 3, 1, 2).reshape(r.shape)
    diff = (o - r).abs().max().item()
    print("  scale %s  max abs diff %.6f" % (tuple(r.shape[2:]), diff))
    ok &= diff < 1e-3

if not ok:
    raise SystemExit("MISMATCH: PCB_YOLO does not match the Ultralytics model. Fix the rebuild before quantizing.")

np.savez(DST, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
print("OK - outputs match. Saved", DST)
