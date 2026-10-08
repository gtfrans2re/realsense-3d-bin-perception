# realsense-3d-bin-perception

Edge RGB-D perception prototype on a Raspberry Pi 4 using an Intel RealSense D421
stereo depth camera. Streams live depth and reports the real-world distance (Z, in
metres) to objects in view — the perception core for a 3D bin-classification system.

Built overnight as a working proof-of-concept.

## Status
- [x] Ubuntu 22.04 (ARM64) on Raspberry Pi 4
- [x] RealSense D421 streaming via pyrealsense2
- [x] Live depth → real-world distance at frame centre
- [ ] YOLO/OpenCV object detection on the colour stream (next)
- [ ] 3D object localisation + bin (A/B) classification by position (next)
- [ ] Live perception dashboard (next)
- [ ] Gazebo arm integration for collision-free pick (future)

## Hardware
- Raspberry Pi 4 (4 GB), Ubuntu Server 22.04 LTS (ARM64)
- Intel RealSense D421 stereo depth camera (USB)

## Dependencies
System (apt): `python3-pip`, `build-essential`, `python3-dev`, `libgl1-mesa-glx`
Python (pip): `pyrealsense2`, `numpy`, `opencv-python`

## Run
```bash
python3 stage1_depth.py
```

## Notes
Currently running the camera over USB 2.0 (480 Mbps), so depth streams at reduced
resolution; USB 3.0 enables full-resolution aligned depth + colour.
