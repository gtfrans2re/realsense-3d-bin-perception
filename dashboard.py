"""
Live web dashboard for the RealSense D421 3D bin-perception pipeline.
Serves the annotated IR view + live stats over the local network.
View from any device: http://<pi-ip>:5000
"""
import os
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, "models", "MobileNetSSD_deploy.prototxt")
MODEL = os.path.join(HERE, "models", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
CONF_THRESH = 0.35
W, H = 480, 270

net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

# shared state between the camera thread and the web server
state = {"frame": None, "stats": {"objects": 0, "bin_a": 0, "bin_b": 0, "fps": 0.0}}
lock = threading.Lock()


def median_depth(depth_frame, cx, cy, k=4):
    vals = []
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < W and 0 <= y < H:
                d = depth_frame.get_distance(x, y)
                if d > 0:
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0


def camera_loop():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, 15)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 15)
    pipeline.start(config)
    prev = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            ir = frames.get_infrared_frame(1)
            depth = frames.get_depth_frame()
            if not ir or not depth:
                continue

            img = cv2.cvtColor(np.asanyarray(ir.get_data()), cv2.COLOR_GRAY2BGR)

            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843,
                                         (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            n, a, b = 0, 0, 0
            for d in range(detections.shape[2]):
                conf = float(detections[0, 0, d, 2])
                if conf < CONF_THRESH:
                    continue
                idx = int(detections[0, 0, d, 1])
                box = detections[0, 0, d, 3:7] * np.array([W, H, W, H])
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = max(0, min(W-1, (x1+x2)//2)), max(0, min(H-1, (y1+y2)//2))
                dist = median_depth(depth, cx, cy)
                bin_id = "BIN A" if cx < W // 2 else "BIN B"
                col = (0, 200, 0) if bin_id == "BIN A" else (0, 140, 255)
                label = CLASSES[idx] if idx < len(CLASSES) else str(idx)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                cv2.circle(img, (cx, cy), 3, col, -1)
                cv2.putText(img, f"{label} {dist:.2f}m {bin_id}", (x1, max(14, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                n += 1
                a += bin_id == "BIN A"
                b += bin_id == "BIN B"

            cv2.line(img, (W//2, 0), (W//2, H), (120, 120, 120), 1)
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            # upscale 2x so it's easier to see in the browser
            img = cv2.resize(img, (W*2, H*2), interpolation=cv2.INTER_NEAREST)
            ok, jpg = cv2.imencode(".jpg", img)
            if ok:
                with lock:
                    state["frame"] = jpg.tobytes()
                    state["stats"] = {"objects": n, "bin_a": a, "bin_b": b, "fps": round(fps, 1)}
    finally:
        pipeline.stop()


app = Flask(__name__)

PAGE = """
<!doctype html><html><head><title>RealSense 3D Bin Perception</title>
<meta http-equiv="refresh" content="2">
<style>
 body{background:#0f1419;color:#e6e6e6;font-family:system-ui,sans-serif;text-align:center;margin:0;padding:20px}
 h1{color:#2a9df4;font-size:20px} img{border:2px solid #2a9df4;border-radius:6px;max-width:95%}
 .stats{display:flex;justify-content:center;gap:24px;margin:14px 0;font-size:16px}
 .stat{background:#1a2530;padding:10px 18px;border-radius:6px}
 .a{color:#4cd964}.b{color:#ff9f40}
</style></head><body>
<h1>RealSense D421 &mdash; Live 3D Bin Perception (Raspberry Pi 4)</h1>
<div class="stats">
 <div class="stat">Objects: <b>{{s.objects}}</b></div>
 <div class="stat a">Bin A: <b>{{s.bin_a}}</b></div>
 <div class="stat b">Bin B: <b>{{s.bin_b}}</b></div>
 <div class="stat">FPS: <b>{{s.fps}}</b></div>
</div>
<img src="/video"><p style="color:#888;font-size:13px">IR + depth fusion &bull; MobileNet-SSD &bull; USB 2.0</p>
</body></html>
"""

@app.route("/")
def index():
    with lock:
        s = state["stats"]
    return render_template_string(PAGE, s=s)

def gen():
    while True:
        with lock:
            frame = state["frame"]
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.06)

@app.route("/video")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    time.sleep(2)
    app.run(host="0.0.0.0", port=5000, threaded=True)