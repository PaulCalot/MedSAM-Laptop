import numpy as np
import random
import cv2
import torch
import os

from .interface import DatasetInterface

class Stage2DistillationDataset(DatasetInterface): 
    def __init__(self
                 , data_root
                 , image_size=256 # This image size is the target length at the end of this preprocessing
                 , gt_size=256
                 , bbox_shift=5
                 , data_aug=True):
        self.data_root = data_root
        self.gt_path = data_root / 'gts'
        self.teacher_gt_path = data_root / 'teacher_gts'
        self.img_path = data_root / 'imgs'
        self.gt_path_files = sorted(self.gt_path.rglob("*.npy"))
        self.gt_path_files = [
            file for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        self.gt_target_size = gt_size # Size added to allow for different img input size and output gt size 

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        assert img_name == os.path.basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_3c = np.load(os.path.join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        # TODO: make it much better, this is only very quick fix (this is dumb)
        img_resize = self.resize_longest_side(img_3c, long_side_length=self.target_length)
        img_resize_for_gt = self.resize_longest_side(img_3c, long_side_length=self.gt_target_size)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize, self.image_size) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'

        # Original Ground truth
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        gt = cv2.resize(
            gt,
            (img_resize_for_gt.shape[1], img_resize_for_gt.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = self.pad_image(gt, self.gt_target_size) # (256, 256)
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)

        # Teacher ground truth
        teacher_gt = self.get_teacher_gt(img_name)
        teacher_gt = cv2.resize(
                        teacher_gt,
                        (img_resize_for_gt.shape[1], img_resize_for_gt.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ) # .astype(np.uint8)
        teacher_gt = self.pad_image(teacher_gt, self.gt_target_size) # (256, 256)
    
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                teacher_gt = np.ascontiguousarray(np.flip(teacher_gt, axis=-1))
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                teacher_gt = np.ascontiguousarray(np.flip(teacher_gt, axis=-2))

        # NOTE: we still use original ground truth for boxe, is this correct ?
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": torch.tensor(img_padded).float(), # 
            "teacher_gt2D": torch.tensor(teacher_gt[None, :,:]).float(), # Float not long
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    def get_teacher_gt(self, img_name):
        path = os.path.join(self.teacher_gt_path, img_name)
        teacher_gt = np.squeeze(np.load(path, 'r', allow_pickle=True))
        return teacher_gt

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

    # def __init__(self
    #              , data_root
    #              , image_size=256 # This image size is the target length at the end of this preprocessing
    #              , gt_size=256
    #              , bbox_shift=5
    #              , data_aug=True):
    #     self.data_root = data_root
    #     self.gt_path = data_root / 'gts'
        
    #     self.teacher_gt_path = data_root / 'teacher_gts'
    #     self.img_path = data_root / 'imgs'
    #     self.gt_path_files = sorted(self.gt_path.rglob("*.npy"))
    #     self.gt_path_files = [
    #         file for file in self.gt_path_files
    #         if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
    #     ]
    #     self.image_size = image_size
    #     self.target_length = image_size
    #     self.bbox_shift = bbox_shift
    #     self.data_aug = data_aug
    #     self.gt_target_size = gt_size

    # def __len__(self):
    #     return len(self.gt_path_files)

    # def __getitem__(self, index):
    #     img_name = os.path.basename(self.gt_path_files[index])
    #     assert img_name == os.path.basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
    #     img_3c = np.load(os.path.join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
    #     # TODO: make it much better, this is only very quick fix (this is dumb)
    #     img_resize = self.resize_longest_side(img_3c, long_side_length=self.target_length)
    #     img_resize_for_gt = self.resize_longest_side(img_3c, long_side_length=self.gt_target_size)
    #     # Resizing
    #     img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
    #     img_padded = self.pad_image(img_resize, self.image_size) # (256, 256, 3)
    #     # convert the shape to (3, H, W)
    #     img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
    #     assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'

    #     # NOTE: don't forget the Squeeze, the mask share is otherwise something like (1, 1, 256, 256)
    #     teacher_gt = np.squeeze(np.load(self.gt_path_files[index], 'r', allow_pickle=True))
    #     teacher_gt = cv2.resize(
    #                     teacher_gt,
    #                     (img_resize_for_gt.shape[1], img_resize_for_gt.shape[0]),
    #                     interpolation=cv2.INTER_NEAREST
    #                 ) # .astype(np.uint8)
    #     teacher_gt = self.pad_image(teacher_gt, self.gt_target_size) # (256, 256)
    
    #     if self.data_aug:
    #         if random.random() > 0.5:
    #             img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
    #             teacher_gt = np.ascontiguousarray(np.flip(teacher_gt, axis=-1))
    #         if random.random() > 0.5:
    #             img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
    #             teacher_gt = np.ascontiguousarray(np.flip(teacher_gt, axis=-2))
    #     # TODO: We need to keep this, but is this OK ?
    #     y_indices, x_indices = np.where(teacher_gt > 0)
    #     x_min, x_max = np.min(x_indices), np.max(x_indices)
    #     y_min, y_max = np.min(y_indices), np.max(y_indices)
    #     # add perturbation to bounding box coordinates
    #     H, W = teacher_gt.shape
    #     x_min = max(0, x_min - random.randint(0, self.bbox_shift))
    #     x_max = min(W, x_max + random.randint(0, self.bbox_shift))
    #     y_min = max(0, y_min - random.randint(0, self.bbox_shift))
    #     y_max = min(H, y_max + random.randint(0, self.bbox_shift))
    #     bboxes = np.array([x_min, y_min, x_max, y_max])
    #     return {
    #         "image": torch.tensor(img_padded).float(), # 
    #         "teacher_gt2D": torch.tensor(teacher_gt[None, :,:]).float(), # NOTE: still float
    #         "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
    #         "image_name": img_name,
    #         "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
    #         "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
    #     }

    # def resize_longest_side(self, image, long_side_length):
    #     """
    #     Expects a numpy array with shape HxWxC in uint8 format.
    #     """
    #     oldh, oldw = image.shape[0], image.shape[1]
    #     scale = long_side_length * 1.0 / max(oldh, oldw)
    #     newh, neww = oldh * scale, oldw * scale
    #     neww, newh = int(neww + 0.5), int(newh + 0.5)
    #     target_size = (neww, newh)
    #     return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # def pad_image(self, image, image_size):
    #     """
    #     Expects a numpy array with shape HxWxC in uint8 format.
    #     """
    #     # Pad
    #     h, w = image.shape[0], image.shape[1]
    #     padh = image_size - h
    #     padw = image_size - w
    #     if len(image.shape) == 3: ## Pad image
    #         image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    #     else: ## Pad gt mask
    #         image_padded = np.pad(image, ((0, padh), (0, padw)))

    #     return image_padded