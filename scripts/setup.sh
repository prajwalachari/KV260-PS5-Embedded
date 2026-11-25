#!/bin/bash
# One-time setup on a KV260 running Kria Ubuntu 22.04 with Kria-PYNQ already installed.
#   git clone <this repo> && cd pcb-defect-kv260 && sudo bash scripts/setup.sh
set -e
if [ "$EUID" -ne 0 ]; then echo "run with sudo"; exit 1; fi
REPO=$(cd "$(dirname "$0")/.." && pwd)

echo "== packages"
apt-get install -y v4l-utils dnsmasq-base

echo "== Pi Camera v2 FPGA design (AMD prebuilt kv260-smartcam-raspi)"
if [ ! -d /lib/firmware/xilinx/kv260-smartcam-raspi ]; then
    echo "Copy the kv260-smartcam-raspi folder (bit.bin, dtbo, xclbin, shell.json) to /lib/firmware/xilinx/ first."
    echo "Source: Xilinx kria-apps-docs, raspi_example / smartcam-raspi_firmware_works"
    exit 1
fi

echo "== model"
[ -f "$REPO/model/kv260/PCB_YOLO_LEAKY_B4096.xmodel" ] || { echo "model/kv260/PCB_YOLO_LEAKY_B4096.xmodel missing"; exit 1; }

echo "== install to /opt and enable services"
ln -sfn "$REPO" /opt/pcb-defect-kv260
install -m 755 "$REPO/scripts/pcb-ip.sh" /usr/local/bin/pcb-ip.sh
cp "$REPO/scripts/pcb-ip.service" "$REPO/scripts/pcb-inspect.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now pcb-ip.service pcb-inspect.service
echo "done. Connect a laptop to the Ethernet port and open http://192.168.50.1:8080"
