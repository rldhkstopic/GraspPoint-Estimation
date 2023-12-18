import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

# parameters
WITDH = 1280
HEIGHT = 720

model = YOLO('weights\weights-train1\last.pt')
classes = yaml_load(check_yaml('datasets\pringles\data.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(classes), 3))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        align_frame = align.process(frames)
        align_depth_frame = align_frame.get_depth_frame()
        align_color_frame = align_frame.get_color_frame()
        
        if not align_depth_frame or not align_color_frame:
            continue
        
        depth_image = np.asanyarray(align_depth_frame.get_data())
        color_image = np.asanyarray(align_color_frame.get_data())
        depth_image = cv2.resize(depth_image, (WITDH, HEIGHT))
        color_image = cv2.resize(color_image, (WITDH, HEIGHT))
        
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03) # not applyColorMap
        results = model(color_image, stream=True)
        
        confs = []
        bboxes = []
        class_ids = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf
                if conf > 0.5:
                    xyxy = box.xyxy.tolist()[0]
                    
                    bboxes.append(xyxy)
                    confs.append(float(conf))
                    class_ids.append(box.cls.tolist())

        result_boxes = cv2.dnn.NMSBoxes(bboxes, confs, 0.25, 0.45, 0.5)

        # Visualization of the results of a detection.
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(bboxes)):
            if i in result_boxes:
                x1, y1, x2, y2 = bboxes[i]
                label = f'{classes[class_ids[i][0]]} {confs[i]:.2f}' 
                color = colors([class_ids[i][0]])
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(color_image, label, (x1, y1 - 5), font, 1, color, 2)  

        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_colormap)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
