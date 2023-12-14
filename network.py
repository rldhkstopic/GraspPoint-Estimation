import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from skimage.transform import rotate
from math import atan2, degrees
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GraspDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, depth, bbox = self.data[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        depth_tensor = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)
        
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        return img_tensor, depth_tensor, bbox_tensor

class GraspNet(nn.Module):
    def __init__(self):
        super(GraspNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 5)  # 출력: x, y, theta, h, w

    def forward(self, x, depth):
        x = torch.cat([x, depth], dim=1) # (B, 4, 224, 224)
        x = self.resnet(x)
        x = self.fc(x)
        return x
    
def bboxes_to_grasps(box):
    x = box[0][0] + (box[2][0] - box[0][0]) / 2
    y = box[0][1] + (box[2][1] - box[0][1]) / 2
    tan = (box[2][0] - box[3][0]) / (box[2][1] - box[3][1] + 1e-6)  # add a small epsilon to prevent division by zero
    w = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    h = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    theta = math.degrees(math.atan(tan))
    if theta < 0:
        theta += 90
    else:
        theta -= 90
    theta = 360 + theta if theta < 0 else theta
    return round(x, 3), round(y, 3), round(theta, 3), round(w, 3), round(h, 3)


def read_bbox_from_file(file_path):
    bboxes = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            bbox = []
            for j in range(4):
                coords = lines[i+j].strip().split()
                bbox.append((int(coords[0]), int(coords[1])))
            bboxes.append(bbox)
    return bboxes
1
def load_data(augmented_data):
    train_data, test_data = train_test_split(augmented_data, test_size=0.2, random_state=42)
    train_dataset = GraspDataset(train_data)
    test_dataset = GraspDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

from tqdm import tqdm

def train_model(train_loader, test_loader):
    model = GraspNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (img, depth, bbox) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = model(img, depth)
            loss = criterion(output, bbox)
            loss.backward()
            optimizer.step()
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for img, depth, bbox in tqdm(test_loader):
                output = model(img, depth)
                test_loss += criterion(output, bbox).item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
    torch.save(model.state_dict(), "grasp_model.pth")

dataset_dir = "rgbd_dataset"
annotations_dir = "rgbd_dataset_annotations"

train_loader, test_loader = load_data(augmented_data)
train_model(train_loader, test_loader)
