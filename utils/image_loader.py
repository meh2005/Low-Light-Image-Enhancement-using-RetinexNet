import os
import cv2
import torch
from torch.utils.data import Dataset

class LowLightDataset(Dataset):

    def __init__(self, low_dir, normal_dir):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.images = [f for f in os.listdir(low_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        low_path = os.path.join(self.low_dir, img_name)
        normal_path = os.path.join(self.normal_dir, img_name)

        low_img = cv2.imread(low_path)
        normal_img = cv2.imread(normal_path)

        if low_img is None:
            raise Exception(f"Low image not found: {low_path}")
        if normal_img is None:
            raise Exception(f"Normal image not found: {normal_path}")

        # Fix: convert BGR (OpenCV default) to RGB
        low_img    = cv2.cvtColor(low_img,    cv2.COLOR_BGR2RGB)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)

        low_img    = cv2.resize(low_img,    (256, 256))
        normal_img = cv2.resize(normal_img, (256, 256))

        # Normalize to [0, 1] using float32
        low_img    = low_img.astype('float32')    / 255.0
        normal_img = normal_img.astype('float32') / 255.0

        low_img    = torch.tensor(low_img).permute(2, 0, 1)
        normal_img = torch.tensor(normal_img).permute(2, 0, 1)

        return low_img, normal_img