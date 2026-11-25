"""
INT8 post-training quantization of PCB_YOLO (LeakyReLU 26/256) with Vitis AI 2.5.

Run inside the Vitis AI 2.5 Docker:
    docker run -it --rm -v ~/kv260_workspace:/workspace -w /workspace/pcb_project xilinx/vitis-ai:2.5.0 bash
    conda activate vitis-ai-pytorch
    python quantize_leaky_vai25.py calib     # 500 training images -> fix points
    python quantize_leaky_vai25.py export    # writes quantized_leaky_vai25/PCB_YOLO_int.xmodel

Calibration uses training images only, so the validation split stays clean for measuring accuracy.
Pre-processing (letterbox, RGB, /255) is identical to the code that runs on the board.
"""
import os
import sys
import random
import cv2
import torch
import numpy as np

from pytorch_nndct.apis import torch_quantizer
from nndct_shared.utils import NndctOption

from vai_yolov8_leaky import PCB_YOLO

ROOT = "/workspace/pcb_project"
WEIGHTS = os.path.join(ROOT, "deployment_candidate/pcb_leaky_vai_weights.npz")
CALIB_DIR = os.path.join(ROOT, "pcb_dataset/train/images")
OUT_DIR = os.path.join(ROOT, "quantized_leaky_vai25")
IMAGE_SIZE = 640
N_CALIB = 500


def get_image_paths():
    paths = sorted(
        os.path.join(CALIB_DIR, n) for n in os.listdir(CALIB_DIR)
        if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not paths:
        raise RuntimeError("No images in " + CALIB_DIR)
    random.Random(0).shuffle(paths)
    return paths[:N_CALIB]


def letterbox(img, size=IMAGE_SIZE, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Cannot read image: " + path)
    img = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float()


def make_model():
    model = PCB_YOLO().eval()
    w = np.load(WEIGHTS)
    model.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files}, strict=True)
    acts = {type(m).__name__ for m in model.modules() if "relu" in type(m).__name__.lower()}
    print("Activations:", acts)
    return model.eval()


def calibrate():
    paths = get_image_paths()
    sample = load_image(paths[0])
    print("Calibration images:", len(paths))
    model = make_model()
    NndctOption.nndct_param_corr.value = 0
    quantizer = torch_quantizer(quant_mode="calib", module=model,
                                input_args=(sample,), output_dir=OUT_DIR)
    qm = quantizer.quant_model
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            qm(load_image(p))
            if i % 100 == 0 or i == len(paths):
                print("Calibration: {}/{}".format(i, len(paths)))
    quantizer.export_quant_config()
    print("CALIBRATION CONFIG EXPORTED")


def export():
    sample = load_image(get_image_paths()[0])
    model = make_model()
    NndctOption.nndct_param_corr.value = 0
    quantizer = torch_quantizer(quant_mode="test", module=model,
                                input_args=(sample,), output_dir=OUT_DIR)
    with torch.no_grad():
        quantizer.quant_model(sample)
    quantizer.export_xmodel(output_dir=OUT_DIR, deploy_check=False)
    xm = os.path.join(OUT_DIR, "PCB_YOLO_int.xmodel")
    if not os.path.isfile(xm):
        raise RuntimeError("Export did not create " + xm)
    print("XMODEL CREATED:", xm, os.path.getsize(xm), "bytes")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("calib", "export"):
        raise SystemExit("Usage: python quantize_leaky_vai25.py calib|export")
    calibrate() if sys.argv[1] == "calib" else export()
