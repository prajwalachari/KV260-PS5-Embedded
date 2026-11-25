#!/bin/bash
# Start the inspection station. Open http://<board-ip>:8080 and press SPACE.
if [ "$EUID" -ne 0 ]; then exec sudo "$0" "$@"; fi
cd "$(dirname "$0")"
pkill -f station.py; pkill v4l2-ctl; sleep 1
source /etc/profile.d/pynq_venv.sh
python3 /usr/local/share/pynq-venv/pynq-dts/insert_dtbo.py 2>/dev/null
exec python3 station.py
