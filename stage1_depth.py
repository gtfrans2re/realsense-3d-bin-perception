import pyrealsense2 as rs
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15)

profile = pipeline.start(config)
print("Depth streaming on USB 2. Move your hand in front of the camera. Ctrl+C to stop.")
try:
    for i in range(120):
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth:
            continue
        w, h = depth.get_width(), depth.get_height()
        dist = depth.get_distance(w // 2, h // 2)
        print(f"frame {i}: center distance = {dist:.3f} m")
        time.sleep(0.1)
finally:
    pipeline.stop()
