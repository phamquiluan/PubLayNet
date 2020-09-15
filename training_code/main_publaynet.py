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

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import (
    MaskRCNN, maskrcnn_resnet50_fpn
)

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
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url 

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
from datasets.layout import Layout 
from datasets.teenet import TeeNet
from datasets.tb_detection import TableBank
# from engine import train_one_epoch, evaluate
from engine import evaluate
from models import (
    maskrcnn001,
    maskrcnn_densenet121_rpn,
    maskrcnn_resnext101_32x8d_rpn,
    maskrcnn_inceptionresnetv2_rpn
)


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

# parser.add_argument("--train-data-path", "-d", default="/mnt/data/luan/publaynet/train")
# parser.add_argument("--device", default="cuda")
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--start-epoch", default=0, type=int)
parser.add_argument("--iter-num", default=0, type=int)

if "opt" in os.getcwd():
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
else:
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("-j", "--workers", default=1, type=int)

parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--wd", "--weight-decay", default=0.0001, type=float, dest="weight_decay")
parser.add_argument("--print-freq", default=10, type=int)
parser.add_argument("--aspect-ratio-group-factor", default=0, type=int)
parser.add_argument("--test-only", dest="test_only", action="store_true",)
parser.add_argument("--pretrained", dest="pretrained", action="store_true",)


# for distributed training, distributed training parameters
parser.add_argument("--world-size", default=1, type=int)
parser.add_argument("--rank", default=0)
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str)
parser.add_argument("--dist-backend", default="nccl")
parser.add_argument("--gpu", default=None, help="GPU id to use.")
parser.add_argument("--multiprocessing-distributed", action="store_true", default=True)


import models
from config import arch as arch_name
# arch = maskrcnn_densenet121_rpn
# arch = maskrcnn_inceptionresnetv2_rpn
# arch = maskrcnn_resnext101_32x8d_rpn
arch = models.__dict__[arch_name]


if IS_SM:
    channel_name = "train"
    
    zip_name = "Detection.zip"
    zip_path = "/opt/ml/input/data/{}/{}".format(channel_name, zip_name)

    if len(os.listdir(os.path.dirname(zip_path))) == 1:  # mean you just have a zip file.
        logger.info("=> START UNZIPPING {}".format(zip_path))
        zip_ref = ZipFile(zip_path, "r")
        zip_ref.extractall(os.path.dirname(zip_path))
        zip_ref.close()

        logger.info("=> UNZIPPING DONE!")
        run_shell("ls {}".format(os.path.dirname(zip_path)))
    

    # TODO: checkpoint dir
    # checkpoint_dir = "/opt/ml/checkpoints/maskrcnn_resnet50_fpn_2019_04_07"
    
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%d_%m_2020_%H%M")
    checkpoint_dir = "/opt/ml/checkpoints/{}".format(arch.__name__)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
else:
    checkpoint_dir = "."


def main(args):
    if IS_SM: 
        args.train_data_path = "/opt/ml/input/data/train/Detection/"
        logger.info(os.listdir("/opt/ml/input/data/train/"))
        logger.info(os.listdir(args.train_data_path))
    else:
        # args.train_data_path = "/mnt/data/luan/data/TableBank_data/tiny_Detection/"
        args.train_data_path = "/mnt/data/luan/data/TableBank_data/Detection/"

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if ngpus_per_node == 1:
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be ajusted accordingly 
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simple call main_worker function
        # main_worker(args.gpu, ngpus_per_node, args)
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

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
    # model = maskrcnn001(num_classes=2)

    model = arch(num_classes=2)

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
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

    if args.distributed:
        # model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    lr_scheduler = MultiStepLR(optimizer, milestones=[20000, 40000], gamma=0.1)

    # ================================
    # resume RESUME CHECKPOINT
    if IS_SM:  # load latest checkpoints 
        checkpoint_list = os.listdir(checkpoint_dir)

        logger.info("=> Checking checkpoints dir.. {}".format(checkpoint_dir))
        logger.info(checkpoint_list)

        latest_path_parent = ""
        latest_path = ""
        latest_iter_num = -1

        for checkpoint_path in natsorted(glob.glob(os.path.join(checkpoint_dir, "*.pth"))):
            checkpoint_name = os.path.basename(checkpoint_path)
            logger.info("Found checkpoint {}".format(checkpoint_name))
            iter_num = int(os.path.splitext(checkpoint_name)[0].split("_")[-1])

            if iter_num > latest_iter_num:
                latest_path_parent = latest_path
                latest_path = checkpoint_path
                latest_iter_num = iter_num 

        logger.info("> latest checkpoint is {}".format(latest_path))

        if latest_path_parent:
            logger.info("=> loading checkpoint {}".format(latest_path_parent))
            checkpoint = torch.load(latest_path_parent, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            args.start_epoch = checkpoint["epoch"]
            args.iter_num = checkpoint["iter_num"]
            logger.info("==> args.iter_num is {}".format(args.iter_num))

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return
    
    logger.info("==================================")
    logger.info("Create dataset with root_dir={}".format(args.train_data_path))
    assert os.path.exists(args.train_data_path), "root_dir does not exists!"
    train_set = TableBank(root_dir=args.train_data_path)

    if args.distributed:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = RandomSampler(train_set)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            train_set,
            k=args.aspect_ratio_group_factor
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler,
            group_ids,
            args.batch_size
        )
    else:
        train_batch_sampler = BatchSampler(
            train_sampler,
            args.batch_size,
            drop_last=True
        )

    logger.info("Create data_loader.. with batch_size = {}".format(args.batch_size))
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        pin_memory=True
    )

    logger.info("Start training.. ")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model=model,
            arch=arch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            data_loader=train_loader,
            device=args.gpu,
            epoch=epoch,
            print_freq=args.print_freq,
            ngpus_per_node=4,
            model_without_ddp=model_without_ddp,
            args=args
        )
        

def train_one_epoch(
        model,
        arch,
        optimizer,
        lr_scheduler,
        data_loader,
        device,
        epoch,
        print_freq,
        ngpus_per_node,
        model_without_ddp,
        args
    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # header = "Epoch: [{}]".format(epoch)

    for images, targets in metric_logger.log_every(
            iterable=data_loader,
            print_freq=print_freq,
            # header=header,
            iter_num=args.iter_num
        ):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        """
        [{"boxes": tensor([], device="cuda:0"), "labels": tensor([], device="cuda:0", dtype=torch.int64), "masks": tensor([], device="cuda:0", dtype=torch.uint8), "iscrowd": tensor([], device="cuda:0", dtype=torch.int64)}]
        """

        try:
            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                logger.fatal("Loss is {}, stopping training".format(loss_value))
                logger.fatal(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        except Exception as e:
            logger.warning(e, exc_info=True)
            # logger.info("print target for debug")
            # print(targets)

        args.iter_num += 1

        # save checkpoint here
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if args.iter_num % 1000 == 0:
                utils.save_on_master({
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "iter_num": args.iter_num,
                        "args": args,
                    },
                    "{}/{}_{}.pth".format(checkpoint_dir, arch.__name__, args.iter_num)
                )

                os.makedirs("{}/debug_image/".format(checkpoint_dir), exist_ok=True)

                if args.iter_num < 5000:
                    continue

                model.eval()

                from barez import overlay_ann	
                debug_image = None
                debug_image_list = []
                cnt = 0
                for image_path in glob.glob("./table_test/*"):
                    cnt += 1
                    image_name = os.path.basename(image_path)
                    # print(image_name)
                    image = cv2.imread(image_path)
                    rat = 1300 / image.shape[0]
                    image = cv2.resize(image, None, fx=rat, fy=rat)

                    transform = transforms.Compose([transforms.ToTensor()])
                    image = transform(image)

                    # put the model in evaluation mode
                    with torch.no_grad():
                        tensor = [image.to(device)]
                        prediction = model(tensor)
                        
                    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

                    for pred in prediction:
                        for idx, mask in enumerate(pred['masks']):
                            if pred['scores'][idx].item() < 0.5:
                                continue
                        
                            m =  mask[0].mul(255).byte().cpu().numpy()
                            box = list(map(int, pred["boxes"][idx].tolist())) 
                            score = pred["scores"][idx].item()
                            image = overlay_ann(image, m, box, "", score)

                    if debug_image is None:
                        debug_image = image
                    else:
                        debug_image = np.concatenate((debug_image, image), axis=1)

                    if cnt == 10:
                        cnt = 0
                        debug_image_list.append(debug_image)
                        debug_image = None
                    
                avg_length = np.mean([i.shape[1] for i in debug_image_list])

                
                di = None

                
                for debug_image in debug_image_list:
                    rat = avg_length / debug_image.shape[1]
                    debug_image = cv2.resize(debug_image, None, fx=rat, fy=rat)

                    if di is None:
                        di = debug_image
                    else:
                        
                        di = np.concatenate((di, debug_image), axis=0)
            

                di = cv2.resize(di, None, fx=0.4, fy=0.4)
                cv2.imwrite("{}/debug_image/{}.jpg".format(checkpoint_dir, args.iter_num), di)

                model.train()

        # hard stop
        if args.iter_num == 50000:
            logger.info("ITER NUM == 50k, training successfully!")
            raise SystemExit


if __name__ == "__main__":
    logger.info("sys.argv: {}".format(sys.argv))

    args, unknown_args = parser.parse_known_args()
    logger.warning("Unknown arguments: {}".format(unknown_args))

    try:
        start_time = time.time()
        main(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("Training time {}".format(total_time_str))

    except Exception as e:
        logger.fatal(e, exc_info=True)
