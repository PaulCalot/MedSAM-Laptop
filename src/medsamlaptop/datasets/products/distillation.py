import numpy as np
import random
import cv2
import torch
import os
from .interface import DatasetInterface

class EncoderDistillationDataset(DatasetInterface): 
    def __init__(self
                 , data_root
                 , image_size=256):
        self.data_root = data_root
        self.gt_path = data_root / 'encoder_gts'
        self.img_path = data_root / 'imgs'
        self.gt_path_files = sorted(self.gt_path.rglob("*.npy"))
        self.gt_path_files = [
            file for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        self.image_size = image_size
        self.target_length = image_size

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        assert img_name == os.path.basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_3c = np.load(os.path.join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        # TODO: make it much better, this is only very quick fix (this is dumb)
        img_resize = self.resize_longest_side(img_3c, long_side_length=self.target_length)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize, self.image_size) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # Output should be something like (256, 64, 64)
        # with batch size would be (B, 256, 64, 64)
        return {
            "image": torch.tensor(img_padded).float(),
            "encoder_gts": torch.squeeze(torch.tensor(gt).float()), # TODO: is the squeeze ok ?
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    def resize_longest_side(self, image, long_side_length):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image, image_size):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = image_size - h
        padw = image_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded