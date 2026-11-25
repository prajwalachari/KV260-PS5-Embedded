# KV260 application

| File | Purpose |
| --- | --- |
| `station.py` | Inspection station: live camera view in the browser, SPACE to inspect, result with severity and advice |
| `run.sh` | Starts the station as root inside the PYNQ environment |
| `detect.py` | Detection on image files, with timing (`--bench N`) |
| `pcb_core.py` | Detector: letterbox, INT8 lookup table, DPU call, DFL decode, NMS |
| `severity.py` | Defect class to severity (critical, major, minor) |
| `defect_logger.py` | One CSV row per inspected board |
| `camera_setup.sh` | Loads the Pi Camera v2 FPGA design and sets the capture pipeline formats |
| `ps_regs.py` | Restores the PS AXI settings that change when the DPU design is loaded |
| `focus_view.py` | Live view with a sharpness score, used to focus the lens |
| `pcb_infer_reference.py` | First, unoptimised version of the detector, kept as a reference |

Inspection flow: camera design loaded -> SPACE -> frame saved -> DPU design loaded -> `detect.py` -> severity, advice, CSV row -> camera design loaded again.
