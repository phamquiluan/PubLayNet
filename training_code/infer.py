import os
import sys
import random
import glob
import json
import math
import time
import datetime
import logging
import argparse
import warnings
import subprocess
from zipfile import ZipFile


import cv2
import numpy as np
from PIL import Image
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from group_by_aspect_ratio import (
    GroupedBatchSampler,
    create_aspect_ratio_groups
)

import utils
import transforms as T

import models

from config import arch as arch_name
arch = models.__dict__[arch_name]



def main():
    model = arch(num_classes=2)
    model.cuda()

    checkpoint = torch.load("/home/z/Downloads/maskrcnn001_56000.pth", map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    model.eval()

    from datasets.tb_detection import table_transforms
    debug_image = None
    debug_image_list = []
    cnt = 0
    # for image_path in glob.glob("./table_test/*"):
    # for image_path in glob.glob("/data/document_data/ICDAR2013/ICDAR2013_table competition_image/*.jpg"):
    for image_path in glob.glob("/data/table/400table/images/*.png"):
        cnt += 1
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        rat = 1300 / image.shape[0]

        image = cv2.resize(image, None, fx=rat, fy=rat)

        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

        # i1 = table_transforms(image=image)["image"]
        # i2 = table_transforms(image=image)["image"]
        # i3 = table_transforms(image=image)["image"]

        # show(np.concatenate((i1, i2, i3), axis=1))

        st = datetime.datetime.now()
        # put the model in evaluation mode
        with torch.no_grad():
            tensor = [image.cuda()]
            # tensor = [i1.cuda(), i2.cuda(), i3.cuda()]
            prediction = model(tensor)


        print("{}\t{}".format(image_name, datetime.datetime.now() - st))
    
        # continue

        image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

        # perform nms on output

        for pred in prediction:
            for idx, mask in enumerate(pred['masks']):
                if pred['scores'][idx].item() < 0.5:
                    continue
            
                m =  mask[0].mul(255).byte().cpu().numpy()
                box = list(map(int, pred["boxes"][idx].tolist())) 
                score = pred["scores"][idx].item()
                # image = overlay_ann(image, m, box, "", score)
        
        # show(image)
        # cv2.imwrite("./debug/{}".format(image_name), image)



if __name__ == "__main__":
    main()
