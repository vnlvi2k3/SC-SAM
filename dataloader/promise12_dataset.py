import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class Promise12Dataset(Dataset):
    def __init__(self, args, base_dir, split='train', transform=None, normalize=True):

        assert split in ['train', 'val', 'test', 'calib'], f"Split '{split}' not recognized"
        self.split_dir = os.path.join(base_dir, split)
        self.images_dir = os.path.join(self.split_dir, 'images')
        self.labels_dir = os.path.join(self.split_dir, 'labels')
        self.args = args
        
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory does not exist: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise RuntimeError(f"Labels directory does not exist: {self.labels_dir}")
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")
            
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        print(f"Loaded {split} split with {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get filenames
        img_filename = self.image_files[idx]
        label_filename = img_filename 
        
        # Load image and label
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, label_filename)
        
        image = Image.open(img_path).convert('L')  
        label = Image.open(label_path).convert('L')  
        
        image = np.array(image, dtype=np.float32) / 255.0
        label = np.array(label, dtype=np.float32) / 255.0
        
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
        image, label = torch.tensor(image), torch.tensor(label)

        image = image.repeat(3, 1, 1)
        sample = {
            'image': image, 
            'label': label.squeeze(0),  
            'filename': img_filename
        }
        
        return sample

