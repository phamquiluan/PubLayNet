import os
import sys
import random
import shutil
import glob
import json
import math
import time
import datetime
import logging
import argparse
import warnings
import subprocess
import warnings
from zipfile import ZipFile

warnings.simplefilter(action="ignore", category=FutureWarning)


IS_SM = False

# =================================================
# setting logger
if os.getcwd() != "/opt/ml/code":
    logging_format = "\033[02m \033[36m[%(asctime)s] [%(levelname)s]\033[0m %(message)s \033[02m <%(name)s, %(funcName)s(): %(lineno)d>\033[0m"
else:
    IS_SM = True
    logging_format = "[%(asctime)s] [%(levelname)s] %(message)s <%(name)s, %(funcName)s(): %(lineno)d>"


if hasattr(time, "tzset"):
    os.environ["TZ"] = "Asia/Ho_Chi_Minh"
    time.tzset()

datefmt = "%b-%d %H:%M"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    datefmt=datefmt,
    filename="log",
    filemode="w"
)

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter(logging_format, datefmt=datefmt)
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logger = logging.getLogger(__name__)
# =================================================


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
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms

from torch.utils.data import (
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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
from datasets.publaynet import PubLayNet
# from engine import train_one_epoch, evaluate
from engine import evaluate


def run_shell(command):
    if isinstance(command, str):
        command = command.split(" ")

    logger.info(" ".join(command))
    process = subprocess.Popen(command)
    while process.poll() is not None:
        output = process.stdout.readline()
        err = process.stderr.readline()
        if output:
            print(output.strip())
        if err:
            print(err.strip())
    process.wait()


parser = argparse.ArgumentParser()

parser.add_argument('--training-data-path', default='/mnt/data/luan/publaynet/train')
parser.add_argument('--device', default='cuda')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--iter-num', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float, dest='weight_decay')
parser.add_argument('--lr-step-size', default=2, type=int)
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--output-dir', default='/opt/ml/checkpoints/checkpoints/')
parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
parser.add_argument(
    "--test-only",
    default=True,
    dest="test_only",
    action="store_true",
)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    action="store_true",
)


# NOTE: for distributed training, distributed training parameters
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0)
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str)
parser.add_argument('--dist-backend', default='nccl')
parser.add_argument('--gpu', default=None, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True)


if IS_SM:
    channel_name = "train"
    zip_name = "val.zip"
    zip_path = "/opt/ml/input/data/{}/{}".format(channel_name, zip_name)

    # NOTE: consider to remove 2 lines below
    # if not os.path.exists(zip_path[:-4]):
    # if not os.path.exists("{}/publaynet".format(os.path.dirname(zip_path))):
    if len(os.listdir(os.path.dirname(zip_path))) == 1:  # mean you just have a zip file.
        logger.info("=> START UNZIPPING {}".format(zip_path))
        zip_ref = ZipFile(zip_path, "r")
        zip_ref.extractall(os.path.dirname(zip_path))
        zip_ref.close()
         
        logger.info("=> UNZIPPING DONE!")
        run_shell("ls {}".format(os.path.dirname(zip_path)))
         
    checkpoint_dir = "/opt/ml/checkpoints/checkpoints"
    eval_dir = "/opt/ml/checkpoints/eval"
    validate_results_dir = "/opt/ml/checkpoints/validate_results"
    log_dir = "/opt/ml/checkpoints/tensorboard_multi_machine/"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(validate_results_dir):
        os.makedirs(validate_results_dir, exist_ok=True)

    if not os.path.exists(log_dir):
        os.makedirs(eval_dir, exist_ok=True)

    tensor_writer = SummaryWriter(log_dir)
else:
    checkpoint_dir = "."
    log_dir = "tensorboard/"
    tensor_writer = SummaryWriter(log_dir)


def get_instance_segmentation_model(num_classes):
    # FIXME: custom the model

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model
 

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    # TODO: try to open this
    """
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    """
    return T.Compose(transforms)


def main(args):
    if IS_SM:  # FIXME: you are still hard code, dude :)) 
        args.test_data_path = "{}/publaynet/val".format(os.path.dirname(zip_path))
    else:
        args.test_data_path = "/mnt/data/luan/publaynet/val"

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be ajusted accordingly 
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simple call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes.
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    # load model here
    model = get_instance_segmentation_model(num_classes=6)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all availabel devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per 
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set.
            model = DistributedDataParallel(model) 
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divice and allocate batch_size to all availabel GPUs
        model = torch.nn.DataParallel(model).cuda()

    if args.distributed:
        # model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    logger.info("==================================")
    logger.info("Create dataset with root_dir={}".format(args.test_data_path))
    assert os.path.exists(args.test_data_path), "root_dir does not exists!"
    test_set = PubLayNet(root_dir=args.test_data_path, transforms=get_transform(train=False))

    if args.distributed:
        test_sampler = DistributedSampler(test_set)
    else:
        test_sampler = SequentialSampler(test_set)

    # create loader
    logger.info("Create data_loader..")
    test_loader = DataLoader(
        test_set,
        batch_size=8,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn
    )

    # step 1: check unvalidate checkpoints in validate_results_dir
    checkpoint_name_list = os.listdir(checkpoint_dir)
    validated_name_list = [os.listdir(validate_results_dir)]

    unvalidation_name_list = [x for x in checkpoint_name_list if "{}.json".format(x) not in validated_name_list]

    if len(unvalidation_name_list) == 0:
        logger.info("All checkpoints are evaluated!!! EXIT...")
        exit(0)

    # step 2: shuffle it 
    random.shuffle(unvalidation_name_list)

    # step 3: choose first elements
    chosen_checkpoint_name = unvalidation_name_list[0]
    checkpoint_path = os.path.join(checkpoint_dir, chosen_checkpoint_name)

    # step 4: validation
    logger.info("=> loading checkpoint {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    iter_num = checkpoint['iter_num']
    logger.info("==> iter_num is {}".format(iter_num))

    coco_evaluator = evaluate(model, test_loader, device=args.gpu)
    
    stats = coco_evaluator.coco_eval["bbox"].stats
    AP, AP50, AP75, AP_small, AP_medium, AP_large = stats[:6]

    tensor_writer.add_scalar('ap/main/ap', AP, iter_num)
    tensor_writer.add_scalar('ap/main/ap50', AP50, iter_num)
    tensor_writer.add_scalar('ap/main/ap75', AP75, iter_num)

    tensor_writer.add_scalar('ap/area/small', AP_small, iter_num)
    tensor_writer.add_scalar('ap/area/medium', AP_medium, iter_num)
    tensor_writer.add_scalar('ap/area/large', AP_large, iter_num)

    tensor_writer.close()

    result_path = os.path.join(
        validate_results_dir,
        "{}.json".format(chosen_checkpoint_name)
    )

    stats = [AP, AP50, AP75, AP_small, AP_medium, AP_large]
    if not os.path.exists(result_path):
        with open(result_path, "w") as result_ref:
            json.dump(stats, result_ref)



if __name__ == "__main__":
    logger.info("sys.argv: {}".format(sys.argv))

    args, unknown_args = parser.parse_known_args()
    logger.warning("Unknown arguments: {}".format(unknown_args))

    if IS_SM and args.output_dir:
        utils.mkdir(args.output_dir)

    try:
        start_time = time.time()
        main(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

    except Exception as e:
        logger.fatal(e, exc_info=True)
