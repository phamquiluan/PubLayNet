import os
import logging
import json

import cv2
import numpy as np
import torch
import torch.utils.data
from torchvision.datasets.coco import CocoDetection
from PIL import Image
from natsort import natsorted


logger = logging.getLogger(__name__)


# class Layout(CocoDetection):
class Layout():
    def __init__(self, root_dir="/data/layout/maskrcnn/", transforms=None):
        self.root_dir = root_dir
        ann_path = os.path.join(root_dir, "labels.json")

        assert os.path.exists(ann_path), os.listdir(os.path.dirname(ann_path))
        # super(PubLayNet, self).__init__(root=root_dir, annFile=ann_path)
        
        with open(ann_path) as ann_data:
            self.anns = json.load(ann_data)
            # self.anns = natsorted(self.anns, key=lambda x:x[0])
            print("image name: ", self.anns[0][0])
            
        self.transforms = transforms

    def __len__(self):
        return len(self.anns)
    
    def get_height_and_width(self, idx):
        image_name, labels_boxes = self.anns[idx]
        image_path = os.path.join(self.root_dir, "images", image_name)
        image = cv2.imread(image_path)
        return image.shape[0], image.shape[1]
        
    def __getitem__(self, idx):
        # coco = self.coco
        # image_name, labels_boxes = self.anns[idx]
        image_name, labels_boxes = self.anns[0]

        image_path = os.path.join(self.root_dir, "images", image_name)
        image = cv2.imread(image_path)
        # TODO: checking if we need to convert image to PIL ?
        # and if need, for what ?

        boxes = []
        labels = []
        instance_masks = []
        iscrowd = []

        for box in labels_boxes:
            location = box["location"]

            xmin = min(p[0] for p in location)
            ymin = min(p[1] for p in location)
            xmax = max(p[0] for p in location)
            ymax = max(p[1] for p in location)

            assert xmin < xmax and ymin < ymax, "{} {} {} {}".format(xmin, ymin, xmax, ymax)
            # if x_min >= x_max or y_min >= y_max:
            #     continue

            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(1)

            instance_mask = np.zeros((image.shape[0], image.shape[1]))
            instance_mask[ymin:ymax, xmin:xmax] = 1
            instance_masks.append(instance_mask)

            assert instance_mask.shape == image.shape[:2]
            iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # labels = [a["category_id"] for a in anns]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # instance_masks = [self.coco.annToMask(ann) for ann in anns]
        instance_masks = np.array(instance_masks)
        instance_masks = torch.as_tensor(instance_masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # iscrowd = [a["iscrowd"] for a in anns]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = instance_masks
        # target["image_id"] = torch.tensor([anns[0]["image_id"]])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


if __name__ == "__main__":
    dataset = Layout()
    print(dataset[0])
