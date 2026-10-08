# realsense-3d-bin-perception

Edge RGB-D perception on a Raspberry Pi 4 with an Intel RealSense D421 stereo
depth camera. Detects objects, fuses each detection with aligned depth to recover
its real-world distance, and classifies it into **Bin A / Bin B** by position —
the perception core of a 3D bin-picking / sorting system.

Built overnight as a working proof-of-concept.

## Contents
- [What it does](#what-it-does)
- [Status](#status)
- [Results](#results)
- [Hardware](#hardware)
- [Dependencies](#dependencies)
- [Run](#run)
- [Roadmap](#roadmap)

## What it does
1. Streams **infrared + depth** from the RealSense D421.
2. Runs **MobileNet-SSD** object detection on the IR frame.
3. Looks up **depth at each detection** (median over a small patch) to get real-world distance (Z, in metres).
4. Classifies each object into **Bin A (left) / Bin B (right)** by horizontal position.
5. Saves annotated frames and prints a live per-frame summary.

> **Note on the sensor:** the D421 is a **depth-only stereo module** — it has no
> RGB camera. Detection is therefore run directly on the **infrared stream**,
> which is natively aligned with depth from the same stereo module. Running a
> color-trained detector on IR reduces detection confidence, which is visible in
> the sample results below — an honest limitation of this hardware choice, not a
> bug in the pipeline.

## Status
- [x] Ubuntu 22.04 (ARM64) on Raspberry Pi 4
- [x] RealSense D421 streaming via pyrealsense2 (IR + depth)
- [x] Live depth → real-world distance
- [x] Object detection (MobileNet-SSD) on the IR stream
- [x] Depth fusion → 3D localisation → Bin A/B classification
- [ ] Swap in **YOLOv11n** (used on Cowbot) — blocked tonight by the CUDA-heavy default install on the Pi; CPU-only build is the next step
- [ ] USB 3.0 cable for full-resolution streaming (currently running on USB 2.0)
- [ ] Live perception dashboard
- [ ] Gazebo arm integration for collision-free pick

## Results
See `output/` for annotated frames.
- **Good frames:** clean detection, correct distance, correct bin assignment.
- **Weaker frames:** class confusion or `0.00 m` readings — these happen when the
  object is inside the camera's minimum stereo range (~<0.3 m) or when IR lacks
  the texture cues the detector was trained on. Shown deliberately for honesty.

## Hardware
- Raspberry Pi 4 (4 GB), Ubuntu Server 22.04 LTS (ARM64)
- Intel RealSense D421 stereo depth camera (USB 2.0 in this build)

## Dependencies
Python deps are pinned in [`requirements.txt`](requirements.txt).

**System (apt):** `python3-pip`, `build-essential`, `python3-dev`, `libgl1-mesa-glx`
**Python (pip):** `pyrealsense2`, `numpy`, `opencv-python` (4.10 — `readNetFromCaffe` was removed in OpenCV 5.x)

```bash
pip install -r requirements.txt
```

## Dependencies
**System (apt):** `python3-pip`, `build-essential`, `python3-dev`, `libgl1-mesa-glx`
**Python (pip):** `pyrealsense2`, `numpy`, `opencv-python` (4.10 — note: `readNetFromCaffe` was removed in OpenCV 5.x)

## Run
```bash
# Stage 1: live depth at frame centre
python3 stage1_depth.py

# Stage 2: detection + depth → bin classification
python3 stage2_bin_perception.py
```
Model files live in `models/` (MobileNet-SSD prototxt + caffemodel).

## Roadmap
Replace MobileNet-SSD with **YOLOv11n**, add a live dashboard, move to USB 3.0 for
full resolution, and integrate a **Gazebo robot arm** for collision-free picking
driven by the 3D positions this pipeline produces.