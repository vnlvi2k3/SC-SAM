import os
import h5py
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import patients_to_slices
import albumentations as A
from copy import deepcopy
import random
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

class ColonDataset(Dataset):
    def __init__(self, args, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.image_normalization = transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
        self.args = args

        if "test" in self.split:
            if "CVC-300" in self.split:
                self.split_dir = os.path.join(self.data_dir + "/TestDataset/CVC-300")
            elif "CVC-ClinicDB" in self.split:
                self.split_dir = os.path.join(self.data_dir + "/TestDataset/CVC-ClinicDB")
            elif "CVC-ColonDB" in self.split:
                self.split_dir = os.path.join(self.data_dir + "/TestDataset/CVC-ColonDB")
            elif "ETIS-LaribPolypDB" in self.split:
                self.split_dir = os.path.join(self.data_dir + "/TestDataset/ETIS-LaribPolypDB")
            elif "Kvasir" in self.split:
                self.split_dir = os.path.join(self.data_dir + "/TestDataset/Kvasir")
            else:
                self.split_dir = None
            self.images_dir = os.path.join(self.split_dir, 'images')   
        else:
            self.split_dir = os.path.join(self.data_dir, split)
            self.images_dir = os.path.join(self.split_dir, 'image')

        
        self.labels_dir = os.path.join(self.split_dir, 'masks')
        
        # Verify directories exist
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory does not exist: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise RuntimeError(f"Labels directory does not exist: {self.labels_dir}")

        self.images = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = [os.path.join(self.labels_dir, f) for f in os.listdir(self.labels_dir) if f.endswith('.png')]

        print(f"Loaded {self.split} split with {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image_path = self.images[idx]
        label_path = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        image = image / 255.0
        label = label / 255.0

        if "val" in self.split or "test" in self.split:
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image']
                label = data['mask']
        else:
            if self.transform:
                if idx < self.args.labeled_num:
                    data = self.transform["train_weak"](image=image, mask=label)
                    image = data['image']
                    label = data['mask']
                else:
                    data = self.transform["train_strong"](image=image, mask=label)
                    image = data['image']
                    label = data['mask']

        label[label < 0.5] = 0
        label[label > 0.5] = 1
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        label = torch.tensor(label, dtype=torch.long)
        image = self.image_normalization(image)

        sample = {"image": image, "label": label}
        return sample
