"""
Live web dashboard + runtime perception monitor for the RealSense D421 pipeline.
Streams the annotated IR view and plots live RUNTIME metrics (detection
confidence, measured distance, FPS, per-bin counts) in the browser via Chart.js.

NOTE: these are operational/runtime metrics (what the system is doing live),
NOT evaluation metrics. Accuracy metrics (mAP, precision/recall) require labeled
ground truth and are the planned offline-evaluation next step (cf. JdeRobot
PerceptionMetrics).

View from any device on the network: http://<pi-ip>:5000
"""
import os
import threading
import time
from collections import deque, Counter
import numpy as np
import cv2
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, "models", "MobileNetSSD_deploy.prototxt")
MODEL = os.path.join(HERE, "models", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
CONF_THRESH = 0.35
W, H = 480, 270
HIST = 60  # rolling window length for charts

net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

state = {
    "frame": None,
    "stats": {"objects": 0, "bin_a": 0, "bin_b": 0, "fps": 0.0},
    "hist": {
        "t": deque(maxlen=HIST),
        "conf": deque(maxlen=HIST),     # mean detection confidence
        "dist": deque(maxlen=HIST),     # mean measured distance (m)
        "fps": deque(maxlen=HIST),
    },
    "class_counts": Counter(),
}
lock = threading.Lock()
t0 = time.time()


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
            confs, dists = [], []
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
                confs.append(conf)
                if dist > 0:
                    dists.append(dist)
                with lock:
                    state["class_counts"][label] += 1

            cv2.line(img, (W//2, 0), (W//2, H), (120, 120, 120), 1)
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            img = cv2.resize(img, (W*2, H*2), interpolation=cv2.INTER_NEAREST)
            ok, jpg = cv2.imencode(".jpg", img)
            if ok:
                with lock:
                    state["frame"] = jpg.tobytes()
                    state["stats"] = {"objects": n, "bin_a": a, "bin_b": b, "fps": round(fps, 1)}
                    h = state["hist"]
                    h["t"].append(round(now - t0, 1))
                    h["conf"].append(round(float(np.mean(confs)), 3) if confs else 0.0)
                    h["dist"].append(round(float(np.mean(dists)), 3) if dists else 0.0)
                    h["fps"].append(round(fps, 1))
    finally:
        pipeline.stop()


app = Flask(__name__)

PAGE = """
<!doctype html><html><head><title>RealSense 3D Bin Perception</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 body{background:#0f1419;color:#e6e6e6;font-family:system-ui,sans-serif;text-align:center;margin:0;padding:18px}
 h1{color:#2a9df4;font-size:20px;margin:6px} .sub{color:#888;font-size:13px}
 .stats{display:flex;justify-content:center;gap:18px;margin:12px 0;font-size:15px;flex-wrap:wrap}
 .stat{background:#1a2530;padding:9px 16px;border-radius:6px}
 .a{color:#4cd964}.b{color:#ff9f40}
 img{border:2px solid #2a9df4;border-radius:6px;max-width:94%}
 .charts{display:flex;flex-wrap:wrap;justify-content:center;gap:16px;margin-top:16px}
 .card{background:#1a2530;border-radius:8px;padding:10px;width:360px}
 .card h3{margin:4px 0 8px;font-size:13px;color:#9bb}
 .note{color:#667;font-size:12px;margin-top:10px}
</style></head><body>
<h1>RealSense D421 &mdash; Live 3D Bin Perception (Raspberry Pi 4)</h1>
<div class="sub">IR + depth fusion &bull; MobileNet-SSD &bull; USB 2.0 &bull; headless Pi served to browser</div>
<div class="stats">
 <div class="stat">Objects: <b id="s-obj">0</b></div>
 <div class="stat a">Bin A: <b id="s-a">0</b></div>
 <div class="stat b">Bin B: <b id="s-b">0</b></div>
 <div class="stat">FPS: <b id="s-fps">0</b></div>
</div>
<img src="/video">
<div class="charts">
 <div class="card"><h3>Mean detection confidence</h3><canvas id="c-conf" height="150"></canvas></div>
 <div class="card"><h3>Mean distance (m)</h3><canvas id="c-dist" height="150"></canvas></div>
 <div class="card"><h3>FPS</h3><canvas id="c-fps" height="150"></canvas></div>
 <div class="card"><h3>Detections per class</h3><canvas id="c-cls" height="150"></canvas></div>
</div>
<div class="note">Runtime (operational) metrics, not accuracy evaluation.
Offline mAP / precision-recall vs. ground truth is the planned next step.</div>

<script>
const mk=(id,type,label,color)=>new Chart(document.getElementById(id),{
  type:type,data:{labels:[],datasets:[{label:label,data:[],borderColor:color,
  backgroundColor:color,tension:0.25,pointRadius:0}]},
  options:{animation:false,plugins:{legend:{display:false}},
  scales:{x:{ticks:{color:'#789',maxTicksLimit:6}},y:{ticks:{color:'#789'},beginAtZero:true}}}});
const chConf=mk('c-conf','line','conf','#4cd964');
const chDist=mk('c-dist','line','dist','#2a9df4');
const chFps =mk('c-fps','line','fps','#ff9f40');
const chCls =mk('c-cls','bar','class','#9b6cf2');

async function tick(){
 try{
  const r=await fetch('/metrics'); const m=await r.json();
  document.getElementById('s-obj').textContent=m.stats.objects;
  document.getElementById('s-a').textContent=m.stats.bin_a;
  document.getElementById('s-b').textContent=m.stats.bin_b;
  document.getElementById('s-fps').textContent=m.stats.fps;
  const upd=(ch,lab,dat)=>{ch.data.labels=lab;ch.data.datasets[0].data=dat;ch.update();};
  upd(chConf,m.hist.t,m.hist.conf);
  upd(chDist,m.hist.t,m.hist.dist);
  upd(chFps ,m.hist.t,m.hist.fps);
  upd(chCls ,m.classes.labels,m.classes.counts);
 }catch(e){}
}
setInterval(tick,700); tick();
</script>
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/metrics")
def metrics():
    with lock:
        h = state["hist"]
        cc = state["class_counts"].most_common(8)
        return jsonify({
            "stats": state["stats"],
            "hist": {"t": list(h["t"]), "conf": list(h["conf"]),
                     "dist": list(h["dist"]), "fps": list(h["fps"])},
            "classes": {"labels": [c for c, _ in cc], "counts": [n for _, n in cc]},
        })

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