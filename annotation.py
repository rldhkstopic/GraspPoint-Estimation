import cv2
import numpy as np
import os

dataset_dir = "rgbd_dataset"
annotation_dir = "_annotations"
os.makedirs(annotation_dir, exist_ok=True)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point: {(x, y)}")
        points.append((x, y))
        if len(points) == 4:
            grasps.append(list(points))
            points.clear()


def mouse_callback(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

def save_grasps(file_name, grasps):
    with open(file_name, 'w') as f:
        for grasp in grasps:
            for point in grasp:
                f.write(f"{point[0]} {point[1]}\n")

for file in sorted(os.listdir(dataset_dir)):
    if file.startswith("rgb"):
        img = cv2.imread(f"{dataset_dir}/{file}")
        points = []
        grasps = []

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mouse_callback)

        while True:
            cv2.imshow('Image', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(points) == 4:
                annotation_file = f"{annotation_dir}/{file.replace('rgb.png', 'cpos.txt')}"
                save_grasps(annotation_file, points)
                print(f"Saved annotation for {file}")
                break
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
