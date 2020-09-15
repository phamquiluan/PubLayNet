import os
import json
import random
import logging

import cv2
import numpy as np
import torch
import torch.utils.data
from torchvision.datasets.coco import CocoDetection
from PIL import Image
import albumentations
from albumentations import (
    Blur,
    Compose,
    RGBShift,
    GaussNoise,
    ImageCompression,
    RandomBrightnessContrast,
    IAAAdditiveGaussianNoise,
    RandomGamma,
    ToGray,
    OneOf
)
from albumentations.pytorch import ToTensor


def overlay_mask(image, mask, alpha=0.5):
    c = (np.random.random((1, 3)) * 153 + 102).tolist()[0]
 
    mask = np.dstack([mask.astype(np.uint8)] * 3)
    mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    inv_mask = 255 - mask

    overlay = image.copy()
    overlay = np.minimum(overlay, inv_mask) 

    color_mask = (mask.astype(np.bool) * c).astype(np.uint8)
    overlay = np.maximum(overlay, color_mask).astype(np.uint8) 

    image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
    return image


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


logger = logging.getLogger(__name__)


table_transforms = Compose([
    OneOf([
        ImageCompression(quality_lower=5, quality_upper=100, p=1.),
        Blur(blur_limit=(3, 5), p=1.),
        GaussNoise(var_limit=(10.0, 151.0), mean=0, always_apply=False, p=1.),
    ],p=0.5),

    IAAAdditiveGaussianNoise(loc=0, scale=(10.55, 50.75), per_channel=False, always_apply=False, p=0.2),

    OneOf([
        RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=1.),
        ToGray(p=0.8),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=False, p=.3),
    ], p=0.5),

    ToTensor()
])  


class TableBank(CocoDetection):
    """
    Table Detection dataset.
    """
    def __init__(
            self,
            root_dir="",
        ):
        self.root_dir = os.path.join(root_dir, "images") 
        ann_path = os.path.join(root_dir, "label.json")

        assert os.path.exists(ann_path), os.listdir(os.path.dirname(ann_path))
        super(TableBank, self).__init__(root=self.root_dir, annFile=ann_path)

        self.albumentation_transforms = Compose([
            OneOf([
                ImageCompression(quality_lower=5, quality_upper=100, p=1.),
                Blur(blur_limit=(3, 5), p=1.),
                GaussNoise(var_limit=(10.0, 151.0), mean=0, always_apply=False, p=1.),
            ],p=0.5),

            IAAAdditiveGaussianNoise(loc=0, scale=(10.55, 50.75), per_channel=False, always_apply=False, p=0.2),

            OneOf([
                RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=1.),
                ToGray(p=0.8),
                RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=False, p=.3),
            ], p=0.5),

            ToTensor()
        ])  

        self.h_flip = RandomHorizontalFlip(0.5)

        self.new_ids = []
        
        for img_id in self.ids:
            path = self.coco.loadImgs(img_id)[0]["file_name"]
            # print(path)
            if os.path.exists(os.path.join(self.root_dir, path)):
                self.new_ids.append(img_id)
 

    def __len__(self):
        return len(self.new_ids)

    def _getitem(self, idx):
        # idx = idx % len(self.ids)

        coco = self.coco
        # img_id = self.ids[idx]
        img_id = self.new_ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        image = cv2.imread(os.path.join(self.root_dir, path))

        if image is None:
            raise FileNotFoundError

        image_height, image_width = image.shape[:2]

        boxes = []
        labels = []
        instance_masks = []
        iscrowd = []


        for ann in anns:
            x_min, y_min, w, h = ann["bbox"]
            x_max = x_min + w
            y_max = y_min + h


            x_min = min(max(0, x_min), image_width)
            y_min = min(max(0, y_min), image_height)
            x_max = min(max(0, x_max), image_width)
            y_max = min(max(0, y_max), image_height)
            if x_min >= x_max or y_min >= y_max:
                continue

            boxes.append([x_min, y_min, x_max, y_max])


            labels.append(ann["category_id"])
            instance_masks.append(self.coco.annToMask(ann))
            iscrowd.append(ann["iscrowd"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        instance_masks = torch.as_tensor(instance_masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = instance_masks
        target["image_id"] = torch.tensor([anns[0]["image_id"]])
        target["area"] = area
        target["iscrowd"] = iscrowd

        image = self.albumentation_transforms(image=image)["image"]
        image, target = self.h_flip(image, target)

        return image, target

    def __getitem__(self, idx):
        while True:
            try: 
                return self._getitem(idx)
            except FileNotFoundError as e:
                idx += 1
                continue


if __name__ == "__main__":
    # dataset = TableBank("/mnt/data/luan/data/TableBank_data/tiny_Detection/")
    dataset = TableBank("/home/z/research/data/tiny_Detection/")
    import random
    import cv2

    for i in range(40):
        idx = random.randint(0, len(dataset) - 1)
        print(idx) 
        tensor, labels = dataset[idx]

        tensor = tensor.permute(1, 2, 0)    
        tensor *= 255
        image = tensor.numpy().astype(np.uint8)


        """
	labels.keys()
	dict_keys(['boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'])

        """
        boxes = labels['boxes'].numpy()
        print(boxes)
        for xmin, ymin, xmax, ymax in boxes:
            xmin, ymin, xmax, ymax = list(map(int, [xmin, ymin, xmax, ymax]))
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 0, 255), 2
            )

        masks = (labels['masks'] * 150).numpy().astype(np.uint8)
        
        for mask in masks:
            image = overlay_mask(image, mask)


        cv2.imwrite("./debug/{}.png".format(i), image)

        # print(tensor)
        # cv2.imwrite("./debug/{}.png".format(idx), vis_image)

