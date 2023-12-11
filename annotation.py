import cv2
import numpy as np
import os

dataset_dir = "rgbd_dataset"
annotation_dir = "_annotations"
os.makedirs(annotation_dir, exist_ok=True)

drawing = False 
ix, iy = -1, -1 

def on_mouse(event, x, y, flags, param):
    global ix, iy, drawing

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


if os.path.exists(dataset_dir) is False:
    print(f"Dataset directory {dataset_dir} does not exist")
    os.makedirs(dataset_dir)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', on_mouse)

for file in sorted(os.listdir(dataset_dir)):
    if file.startswith("rgb"):
        img_path = os.path.join(dataset_dir, file)
        img = cv2.imread(img_path)
        grasps = []

        while True:
            cv2.imshow('Image', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                annotation_file = os.path.join(annotation_dir, file.replace('rgb.png', 'cpos.txt'))
                save_grasps(annotation_file, grasps)
                print(f"Saved annotation for {file}")
                break

            elif key == ord('q'):
                break

cv2.destroyAllWindows()