import pyrealsense2 as rs
import numpy as np
import cv2
import os

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

dataset_dir = "rgbd_dataset"
create_directory(dataset_dir)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('\r'):   # Enter 키를 누르면
            img_index = len(os.listdir(dataset_dir)) // 2 +1 
            cv2.imwrite(f"{dataset_dir}/rgb_{img_index}.png", color_image)
            cv2.imwrite(f"{dataset_dir}/depth_{img_index}.png", depth_colormap)
            print(f"Saved RGB and Depth frame {img_index}")

    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
