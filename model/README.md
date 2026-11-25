# Model

YOLOv8s, 6 classes, 640 x 640 input, SiLU replaced by LeakyReLU(26/256) so the whole network runs on the DPU.

| File | What it is |
| --- | --- |
| `export/vai_yolov8_leaky.py` | Plain PyTorch rebuild of YOLOv8s that Vitis AI 2.5 can trace |
| `export/make_leaky_state_dict.py` | Loads the trained checkpoint into the rebuild, checks the outputs match, saves .npz |
| `export/quantize_leaky_vai25.py` | INT8 post-training quantization, 500 calibration images |
| `export/compile.sh` | Compiles for DPUCZDX8G_ISA1_B4096 |
| `kv260/PCB_YOLO_LEAKY_B4096.xmodel` | Deployed model, about 12 MB |

Order: `training/train.py leaky` -> `export/make_leaky_state_dict.py` (checks the rebuild matches Ultralytics, max difference 0.000000) -> `quantize_leaky_vai25.py calib` -> `quantize_leaky_vai25.py export` -> `compile.sh`.

| | FP32 (PC) | INT8 (KV260 DPU) |
| --- | --- | --- |
| mAP50 | 0.992 | 0.991 |
| mAP50-95 | 0.748 | 0.734 |
| Size | about 45 MB | about 12 MB |

The rebuild matches the Ultralytics 8.4 graph to within 1e-8 (float64) on random weights and inputs.

Accuracy is on the validation split, 1,066 images, 2,164 defects. The DPU runs all 519 operations as one subgraph in about 33 ms.
