import cv2
import numpy as np
import os
from skimage.transform import rotate
from math import atan2, degrees

def bboxes_to_grasps(box):
    x = box[0][0] + (box[2][0] - box[0][0]) / 2
    y = box[0][1] + (box[2][1] - box[0][1]) / 2
    tan = (box[2][0] - box[3][0]) / (box[2][1] - box[3][1])
    w = np.sqrt((box[1][0] - box[0][0])**2 + (box[1][1] - box[0][1])**2)
    h = np.sqrt((box[3][0] - box[0][0])**2 + (box[3][1] - box[0][1])**2)
    theta = 360 - (degrees(atan2(tan, 1)) + 90)
    return round(x, 3), round(y, 3), round(theta, 3), round(h, 3), round(w, 3)

def augment_image(image, depth, bbox):
    angle = np.random.randint(-180, 180)
    tx = np.random.randint(-50, 50)
    ty = np.random.randint(-50, 50)
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    rotated_image = rotate(image, angle, resize=True)
    rotated_depth = rotate(depth, angle, resize=True)
    translated_image = cv2.warpAffine(rotated_image, M, (rotated_image.shape[1], rotated_image.shape[0]))
    translated_depth = cv2.warpAffine(rotated_depth, M, (rotated_depth.shape[1], rotated_depth.shape[0]))
    grasps = bboxes_to_grasps(bbox)
    return translated_image, translated_depth, grasps


def read_bbox_from_file(file_path):
            bboxes = []
            with open(file_path, 'r') as file:
                for line in file:
                    bbox_coords = line.strip().split(',')
                    bbox = [((int(bbox_coords[0]), int(bbox_coords[1])),
                             (int(bbox_coords[2]), int(bbox_coords[3])),
                             (int(bbox_coords[4]), int(bbox_coords[5])),
                             (int(bbox_coords[6]), int(bbox_coords[7])))]
                    bboxes.append(bbox)
            return bboxes
        
augmented_data = []
dataset_dir = "rgbd_dataset"
for file in sorted(os.listdir(dataset_dir)):
    if file.endswith("rgb.png"):
        img = cv2.imread(os.path.join(dataset_dir, file))
        depth = cv2.imread(os.path.join(dataset_dir, file.replace("rgb.png", "depth.png")), cv2.IMREAD_UNCHANGED)
        
        annotation_file = file.replace("rgb.png", "cpos.txt")
        bbox = read_bbox_from_file(annotation_file)
        

        for _ in range(500):
            augmented_img, augmented_depth, augmented_bbox = augment_image(img, depth, bbox)
            augmented_data.append((augmented_img, augmented_depth, augmented_bbox))
