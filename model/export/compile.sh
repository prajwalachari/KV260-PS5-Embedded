#!/bin/bash
# Compile the quantized model for the KV260 DPU (run inside the Vitis AI 2.5 Docker).
# The fingerprint selects the DPU configuration that Kria-PYNQ loads: DPUCZDX8G_ISA1_B4096.
set -e
echo '{"fingerprint":"0x101000016010407"}' > arch_b4096.json
vai_c_xir -x quantized_leaky_vai25/PCB_YOLO_int.xmodel -a arch_b4096.json \
          -o compiled_b4096 -n PCB_YOLO_LEAKY_B4096
# Expect: "Total device subgraph number 5, DPU subgraph number 1"
ls -l compiled_b4096/PCB_YOLO_LEAKY_B4096.xmodel
