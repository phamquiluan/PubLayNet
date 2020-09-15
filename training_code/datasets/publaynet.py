import os
import logging
import json
import numpy as np
import torch
import torch.utils.data
from torchvision.datasets.coco import CocoDetection
from PIL import Image


logger = logging.getLogger(__name__)


class PubLayNet(CocoDetection):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        ann_path = root_dir + ".json"
        assert os.path.exists(ann_path), os.listdir(os.path.dirname(ann_path))
        super(PubLayNet, self).__init__(root=root_dir, annFile=ann_path)
        self.transforms = transforms


    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        image_width, image_height = image.size

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

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


if __name__ == "__main__":
    dataset = PubLayNet(root_dir="/data/publay/val")
    print(dataset[0])
