#!/usr/bin/env python3
# Camera focus tool: live preview in the browser with a sharpness score. SPACE saves a photo.
#   sudo bash camera_setup.sh && sudo python3 focus_view.py   ->  open http://<board-ip>:8080
# Photos go to /home/ubuntu/photos. The number in the corner is sharpness: turn the lens to make it as high as possible.
import os, time, threading, subprocess
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

W, H, PORT, OUT = 1280, 720, 8080, "/home/ubuntu/photos"
os.makedirs(OUT, exist_ok=True)
FS = W * H * 3
cam = {"img": None, "jpg": None, "n": 0}; lock = threading.Lock()

def camera():
    p = subprocess.Popen(["v4l2-ctl", "-d", "/dev/video0", f"--set-fmt-video=width={W},height={H},pixelformat=BGR3",
                          "--stream-mmap", "--stream-to=-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    buf = bytearray(FS)
    while True:
        v, got = memoryview(buf), 0
        while got < FS:
            k = p.stdout.readinto(v[got:])
            if not k: print("camera stream ended - run camera_setup.sh first", flush=True); os._exit(1)
            got += k
        img = np.frombuffer(bytes(buf), np.uint8).reshape(H, W, 3)
        with lock: cam["img"] = img; cam["n"] += 1

def preview():
    last = 0
    while True:
        with lock: img, n = cam["img"], cam["n"]
        if img is None or n == last: time.sleep(0.01); continue
        last = n
        g = cv2.cvtColor(img[H // 4:3 * H // 4, W // 4:3 * W // 4], cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(g, cv2.CV_64F).var()           # focus measure of the centre
        v = cv2.resize(img, (960, 540))
        cv2.rectangle(v, (240, 135), (720, 405), (255, 255, 255), 1)   # centre box used for sharpness
        cv2.putText(v, f"sharpness {sharp:6.0f}   SPACE = photo", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        ok, j = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with lock: cam["jpg"] = j.tobytes()

PAGE = b"""<html><body style='margin:0;background:#111;color:#fff;font:20px sans-serif;text-align:center'>
<img src='/stream' style='width:100vw;height:92vh;object-fit:contain'><div id=m>press SPACE to take a photo</div>
<script>
let busy=false;
function snap(){ if(busy) return; busy=true; document.body.style.background='#fff';
  fetch('/snap',{method:'POST'}).then(r=>r.text()).then(t=>{ document.getElementById('m').textContent=t; })
  .finally(()=>{ setTimeout(()=>{document.body.style.background='#111'; busy=false;},150); }); }
document.addEventListener('keydown',e=>{ if(e.key===' '){ e.preventDefault(); snap(); } });
</script></body></html>"""

class Hd(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_POST(self): self.do_GET()
    def do_GET(self):
        if self.path == "/snap" and self.command == "POST":
            with lock: img = cam["img"]
            if img is None: self.send_error(503); return
            name = f"{OUT}/IMG_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}.jpg"
            cv2.imwrite(name, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n = len([f for f in os.listdir(OUT) if f.endswith(".jpg")])
            print("saved", name, flush=True)
            self.send_response(200); self.end_headers(); self.wfile.write(f"saved {os.path.basename(name)}  ({n} photos)".encode()); return
        if self.path != "/stream":
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE); return
        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
        last = None
        try:
            while True:
                with lock: j = cam["jpg"]
                if j is None or j is last: time.sleep(0.01); continue
                last = j
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
        except Exception:
            pass

threading.Thread(target=camera, daemon=True).start()
threading.Thread(target=preview, daemon=True).start()
print(f"preview: http://<board-ip>:{PORT}   SPACE = photo -> {OUT}   (Ctrl+C to stop)", flush=True)
try:
    ThreadingHTTPServer(("0.0.0.0", PORT), Hd).serve_forever()
except KeyboardInterrupt:
    print("stopped")
