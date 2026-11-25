#!/usr/bin/env python3
# One-key PCB inspection station: Pi Camera v2 + YOLOv8s INT8 on the KV260 DPU.
#   sudo ./run.sh      then open http://<board-ip>:8080
#   live preview -> SPACE = inspect -> result -> SPACE = next board
# The FPGA holds one design at a time, so each inspection switches camera design -> DPU -> camera design.
import os, sys, time, threading, subprocess
import numpy as np, cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP, "..", "advisory_llm"))
from severity import SEVERITY, board_severity
from defect_logger import DefectLogger
import advisor

W, H, PORT = 1280, 720, 8080
OUT = os.environ.get("PCB_OUT", "/home/ubuntu/inspections"); os.makedirs(OUT, exist_ok=True)
LOG = DefectLogger(os.path.join(OUT, "inspections.csv"))
FS = W * H * 3
st = {"mode": "starting", "jpg": None, "img": None, "msg": "starting camera...", "busy": False}
lock = threading.Lock()
cam_proc = [None]

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def placeholder(text):
    im = np.full((540, 960, 3), (40, 28, 24), np.uint8)
    cv2.putText(im, text, (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", im)[1].tobytes()

def start_camera():
    for attempt in range(1, 4):                          # the camera sometimes stalls after a firmware swap: retry
        with lock: st.update(mode="starting", img=None, msg=f"loading camera firmware... (try {attempt})", jpg=placeholder("loading camera..."))
        sh("pkill v4l2-ctl; xmutil unloadapp; rmdir /sys/kernel/config/device-tree/overlays/pynq 2>/dev/null; python3 /usr/local/share/pynq-venv/pynq-dts/insert_dtbo.py 2>/dev/null")   # back to the boot-time state
        r = sh(f"bash {APP}/camera_setup.sh")
        if r.returncode != 0:
            print("camera_setup.sh failed:", (r.stderr or r.stdout)[-300:], flush=True); continue
        p = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video0", f"--set-fmt-video=width={W},height={H},pixelformat=BGR3",
                              "--stream-mmap", "--stream-to=-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        cam_proc[0] = p
        threading.Thread(target=read_camera, args=(p,), daemon=True).start()
        with lock: st.update(mode="preview", msg="position the PCB, then press SPACE")
        t = time.time()
        while time.time() - t < 6:                       # frames must arrive within 6 s
            with lock: ok = st["img"] is not None
            if ok: return
            time.sleep(0.2)
        print(f"camera gave no frames (try {attempt}), reloading", flush=True)
        p.kill(); p.wait(); cam_proc[0] = None
        sh("pkill v4l2-ctl; xmutil unloadapp")
    with lock: st.update(mode="result", msg="camera did not start after 3 tries - power-cycle the board, then: sudo ./run.sh")

def read_camera(p):
    buf = bytearray(FS); n = 0
    while True:
        v, got = memoryview(buf), 0
        while got < FS:
            k = p.stdout.readinto(v[got:])
            if not k: return
            got += k
        n += 1
        img = np.frombuffer(bytes(buf), np.uint8).reshape(H, W, 3)
        if n <= 10 or n % 2: continue                        # skip stale first frames; preview at ~15 fps
        pv = cv2.resize(img, (960, 540))
        cv2.putText(pv, "LIVE  -  SPACE = inspect", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        j = cv2.imencode(".jpg", pv, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        with lock:
            if st["mode"] == "preview": st.update(img=img, jpg=j)

def stop_camera():
    p = cam_proc[0]
    if p: p.kill(); p.wait()
    cam_proc[0] = None
    sh("pkill v4l2-ctl; xmutil unloadapp; cp /etc/vart.conf.pynq /etc/vart.conf 2>/dev/null; python3 /usr/local/share/pynq-venv/pynq-dts/insert_dtbo.py 2>/dev/null")

def inspect():
    with lock: img = st["img"]; st.update(mode="busy", msg="inspecting on the DPU...", jpg=placeholder("inspecting on the DPU..."))
    name = time.strftime("%Y%m%d_%H%M%S")
    src, dst = f"{OUT}/{name}.png", f"{OUT}/{name}_result.jpg"
    cv2.imwrite(src, img)
    stop_camera()
    t = time.time()
    r = sh(f"python3 {APP}/detect.py {src}")
    det_path = src.rsplit(".", 1)[0] + "_det.jpg"
    if not os.path.exists(det_path):
        with lock: st.update(mode="result", msg="detection failed: " + (r.stderr or r.stdout)[-300:])
        return
    defects = []
    for l in r.stdout.splitlines():
        if l.startswith("  ") and "[" in l:
            cls, conf, box = l.split()[:3]
            defects.append({"cls": cls, "conf": float(conf), "box": box})
    n = len(defects); sev = board_severity(defects)
    verdict = "PASS" if n == 0 else "FAIL"
    timing = [l for l in r.stdout.splitlines() if "DPU" in l and "|" in l]
    dpu_ms = timing[-1].split("DPU")[1].split("ms")[0].strip() if timing else ""
    cycle = f"{time.time() - t:.1f}"
    advice = advisor.rule_advice(defects)
    print(f"{name}: {verdict} {sev}  {[(d['cls'], d['conf']) for d in defects]}  DPU {dpu_ms} ms  (cycle {cycle} s)", flush=True)
    im = cv2.imread(det_path)
    banner = "PASS" if n == 0 else f"FAIL - {n} defect(s), {sev}"
    cv2.rectangle(im, (0, 0), (im.shape[1], 60), (40, 150, 40) if n == 0 else (40, 40, 200), -1)
    cv2.putText(im, banner + "   (SPACE = next board)", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(dst, im)
    LOG.log(dst, verdict, sev, defects, dpu_ms, cycle, advice)
    with lock:
        st.update(mode="result", jpg=cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(),
                  msg=f"{banner}  |  {advice}  |  SPACE = next board")
    if n and advisor.llm_available():          # Llama advice runs in the background, the operator is not kept waiting
        threading.Thread(target=llm_update, args=(defects, banner), daemon=True).start()


def llm_update(defects, banner):
    text = advisor.llm_advice(defects)
    if text:
        print("LLM advice:", text, flush=True)
        with lock:
            if st["mode"] == "result":
                st["msg"] = f"{banner}  |  {text}  |  SPACE = next board"

def on_space():
    with lock:
        if st["busy"] or st["mode"] in ("starting", "busy"): return
        st["busy"] = True; mode = st["mode"]
    try:
        if mode == "preview" and st["img"] is not None: inspect()
        elif mode == "result": start_camera()
    finally:
        with lock: st["busy"] = False

PAGE = b"""<html><body style='margin:0;background:#111;color:#fff;font:20px sans-serif;text-align:center'>
<img src='/stream' style='width:100vw;height:90vh;object-fit:contain'><div id=m></div>
<script>
setInterval(()=>fetch('/msg').then(r=>r.text()).then(t=>{document.getElementById('m').textContent=t;}),300);
document.addEventListener('keydown',e=>{ if(e.key===' '){ e.preventDefault(); fetch('/space',{method:'POST'}); } });
</script></body></html>"""

class Hd(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_POST(self):
        if self.path == "/space": threading.Thread(target=on_space, daemon=True).start()
        self.send_response(200); self.end_headers()
    def do_GET(self):
        if self.path == "/msg":
            with lock: m = st["msg"]
            self.send_response(200); self.end_headers(); self.wfile.write(m.encode()); return
        if self.path != "/stream":
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE); return
        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
        last, tl = None, 0
        try:
            while True:
                with lock: j = st["jpg"]
                if j is None or (j is last and time.time() - tl < 0.3): time.sleep(0.03); continue
                last, tl = j, time.time()                    # resend still images so the browser shows them
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
        except Exception:
            pass

threading.Thread(target=start_camera, daemon=True).start()
print(f"open http://<board-ip>:{PORT}   SPACE = inspect / next board   results -> {OUT}   (Ctrl+C to stop)", flush=True)
try:
    ThreadingHTTPServer(("0.0.0.0", PORT), Hd).serve_forever()
except KeyboardInterrupt:
    stop_camera(); print("stopped")
