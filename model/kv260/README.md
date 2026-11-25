# Deployed model

`PCB_YOLO_LEAKY_B4096.xmodel` (about 12 MB) goes in this folder. It is the output of `model/export/compile.sh`:

- Target DPU: DPUCZDX8G_ISA1_B4096, fingerprint 0x101000016010407 (the DPU in Kria-PYNQ's `dpu.bit`)
- Input: 1 x 640 x 640 x 3, INT8, fix point 6
- Outputs: 80 x 80, 40 x 40 and 20 x 20, 70 channels each (64 DFL box bins + 6 class logits), fix point 1
- 519 operations, one DPU subgraph

`kv260_app/detect.py` and `kv260_app/station.py` load it from here. Another path can be given with `--xmodel` or the `PCB_XMODEL` environment variable.
