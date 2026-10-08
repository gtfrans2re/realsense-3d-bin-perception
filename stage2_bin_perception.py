"""
Stage 2: 3D bin perception
RealSense D421 (color + depth) -> MobileNet-SSD detection -> depth lookup per
object -> real-world distance -> Bin A/B classification by horizontal position.

Runs headless: prints a live summary and saves annotated frames to ./output/.
Architected to swap in YOLOv11n (used on Cowbot) where the detector is loaded.
"""
import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, "models", "MobileNetSSD_deploy.prototxt")
MODEL = os.path.join(HERE, "models", "MobileNetSSD_deploy.caffemodel")
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# MobileNet-SSD (VOC) class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
CONF_THRESH = 0.35

# ---------- load detector ----------
print("Loading MobileNet-SSD detector...")
net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

# ---------- start RealSense (USB2-friendly resolution) ----------
# D421 is depth-only (stereo IR + depth, NO color sensor).
# Run detection on the infrared image; depth is natively aligned to it.
W, H = 480, 270
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, 15)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 15)
profile = pipeline.start(config)
print("Streaming IR + depth. Processing 150 frames (~10s). Ctrl+C to stop early.\n")

def median_depth(depth_frame, cx, cy, k=4):
    """Median distance (m) over a small patch around (cx,cy); robust to holes."""
    vals = []
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < W and 0 <= y < H:
                d = depth_frame.get_distance(x, y)
                if d > 0:
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0

try:
    for i in range(400):
        frames = pipeline.wait_for_frames()
        ir = frames.get_infrared_frame(1)
        depth = frames.get_depth_frame()
        if not ir or not depth:
            continue

        ir_img = np.asanyarray(ir.get_data())          # grayscale (H,W)
        img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)  # 3-channel for detector + drawing

        # detection
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843,
                                     (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        found = []
        for d in range(detections.shape[2]):
            conf = float(detections[0, 0, d, 2])
            if conf < CONF_THRESH:
                continue
            idx = int(detections[0, 0, d, 1])
            box = detections[0, 0, d, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = max(0, min(W - 1, cx)); cy = max(0, min(H - 1, cy))

            dist = median_depth(depth, cx, cy)
            bin_id = "BIN A" if cx < W // 2 else "BIN B"  # left vs right
            label = CLASSES[idx] if idx < len(CLASSES) else str(idx)

            # draw
            col = (0, 200, 0) if bin_id == "BIN A" else (0, 140, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.circle(img, (cx, cy), 3, col, -1)
            txt = f"{label} {dist:.2f}m -> {bin_id}"
            cv2.putText(img, txt, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            found.append(txt)

        # divider line between bins
        cv2.line(img, (W // 2, 0), (W // 2, H), (120, 120, 120), 1)

        if i % 10 == 0:
            print(f"frame {i:3d}: " + ("; ".join(found) if found else "no objects"))

        cv2.imwrite(os.path.join(OUT_DIR, "latest.jpg"), img)
        if found and i % 30 == 0:
            cv2.imwrite(os.path.join(OUT_DIR, f"detect_{i:03d}.jpg"), img)

finally:
    pipeline.stop()
    print(f"\nDone. Annotated frames saved in {OUT_DIR}/")    
