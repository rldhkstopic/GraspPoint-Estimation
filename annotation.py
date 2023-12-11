import cv2
import numpy as np
import os

dataset_dir = "rgbd_dataset"
annotation_dir = "rgbd_dataset_annotations"
os.makedirs(annotation_dir, exist_ok=True)

points = []
grasps = []

def on_mouse(event, x, y, flags, param):
    global points, grasps, img

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        if len(points) == 4:
            grasps.append(points.copy())
            points.clear()

def save_grasps(file_name, grasps):
    with open(file_name, 'w') as f:
        for grasp in grasps:
            for point in grasp:
                f.write(f"{point[0]} {point[1]}\n")

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', on_mouse)

for file in sorted(os.listdir(dataset_dir)):
    if file.startswith("rgb") and file.endswith(".png"):
        img_path = os.path.join(dataset_dir, file)
        img = cv2.imread(img_path)
        
        points = []
        grasps = []

        while True:
            cv2.imshow('Image', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                annotation_file_name = file.replace(".png", "_cpos.txt")
                annotation_file_path = os.path.join(annotation_dir, annotation_file_name)
                save_grasps(annotation_file_path, grasps)
                print(f"Saved annotation for {file}")
                break
            elif key == ord('q'):
                break

cv2.destroyAllWindows()
