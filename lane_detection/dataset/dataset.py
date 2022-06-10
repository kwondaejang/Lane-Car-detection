import numpy as np
import torch
import os
from PIL import Image

# class to define tusimple dataset
class tusimpleDatasetReg(torch.utils.data.Dataset):
    def __init__(self, root_path_to_data, dataclass, transforms=None):
        self.root_path_to_data = root_path_to_data
        self.dataclass_ = dataclass
        self.img_paths = []
        self.transforms = transforms

        # define path to file
        self.train_file = os.path.join(root_path_to_data, 'train_split.txt')
        self.val_file = os.path.join(root_path_to_data, 'valid_split.txt')
        self.test_file = os.path.join(root_path_to_data, 'test.txt')

        if self.dataclass_ == 'train':
            file_open = self.train_file
        elif self.dataclass_ == 'valid':
            file_open = self.val_file
        elif self.dataclass_ == 'test':
            file_open = self.test_file

        # read file based on dataclass
        with open(file_open, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split()
                self.img_paths.append(line)


    def __len__(self):
        return len(self.img_paths)

    # get item method
    def __getitem__(self, index):
        # get path to image and mask
        image_path = self.img_paths[index][0]
        mask_path = self.img_paths[index][1]

        # image preprocess
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        # image transforms
        if(self.transforms is not None):
            transformed = self.transforms(image =image, mask=mask )
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
        
        # retrun the image transforms
        return transformed_image, transformed_mask
