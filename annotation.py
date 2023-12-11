import cv2
import numpy as np
import os

dataset_dir = "rgbd_dataset"
annotation_dir = "rgbd_dataset_annotations"
os.makedirs(annotation_dir, exist_ok=True)

drawing = False
ix, iy = -1, -1
grasps = []

def on_mouse(event, x, y, flags, param):
    global ix, iy, drawing, grasps

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        grasps.append([(ix, iy), (x, iy), (x, y), (ix, y)])
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

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
        img_copy = img.copy()
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
            elif key == ord('c') and grasps:
                grasps.pop()
                img = img_copy.copy()  
                for g in grasps:  
                    cv2.rectangle(img, g[0], g[2], (0, 255, 0), 2)
            elif key == ord('q'):
                break

cv2.destroyAllWindows()
