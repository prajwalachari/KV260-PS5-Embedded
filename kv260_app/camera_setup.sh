#!/bin/bash
# Load the Pi Camera v2 FPGA design and configure the capture pipeline.
# Run as root. station.py calls this every time it switches back to the camera.
set -e
APP=kv260-smartcam-raspi
W=${W:-1280}; H=${H:-720}          # camera frame size delivered to /dev/video0

echo "[1/5] loading FPGA firmware $APP"
xmutil unloadapp >/dev/null 2>&1 || true
xmutil loadapp $APP
python3 "$(dirname "$0")/ps_regs.py"     # undo the PS settings PYNQ changes when it loads the DPU
sleep 2

echo "[2/5] configuring pipeline: imx219 1920x1080 RAW10 -> ISP -> scaler ${W}x${H} BGR"
M="media-ctl -d /dev/media0 -V"
SENSOR=$(media-ctl -d /dev/media0 -p | grep -o '"imx219 [0-9]*-0010"' | head -1 | tr -d '"')
$M "\"$SENSOR\":0 [fmt:SRGGB10_1X10/1920x1080 field:none]"
$M '"80002000.csiss":0 [fmt:SRGGB10_1X10/1920x1080 field:none]'
$M '"80002000.csiss":1 [fmt:SRGGB10_1X10/1920x1080 field:none]'
$M '"b0000000.isp_accel":0 [fmt:SRGGB10_1X10/1920x1080 field:none]'
$M '"b0000000.isp_accel":1 [fmt:RBG888_1X24/1920x1080 field:none]'
$M '"b0040000.scaler":0 [fmt:RBG888_1X24/1920x1080 field:none]'
$M "\"b0040000.scaler\":1 [fmt:RBG888_1X24/${W}x${H} field:none]"
v4l2-ctl -d /dev/video0 --set-fmt-video=width=$W,height=$H,pixelformat=BGR3

echo "[3/5] fixing HP0 port width (128 -> 64 bit), otherwise frames come out striped"
python3 - <<'EOF'
import mmap, os, struct
f = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
m = mmap.mmap(f, 0x1000, offset=0xFD380000)          # AFIFM2 = S_AXI_HP0_FPD
for o in (0x0, 0x14):                                # RDCTRL, WRCTRL; bits[1:0] 1 = 64-bit
    v = struct.unpack('<I', m[o:o+4])[0]
    m[o:o+4] = struct.pack('<I', (v & ~3) | 1)
EOF

echo "[4/5] camera: test pattern off"
SUB=$(media-ctl -d /dev/media0 -e "$SENSOR")
v4l2-ctl -d $SUB --set-ctrl=test_pattern=0

echo "[5/5] pointing the DPU runtime at this firmware's DPU"
[ -f /etc/vart.conf.pynq ] || cp /etc/vart.conf /etc/vart.conf.pynq 2>/dev/null || true
echo "firmware: /lib/firmware/xilinx/$APP/$APP.xclbin" > /etc/vart.conf
echo "done. camera at /dev/video0 (${W}x${H} BGR)"
