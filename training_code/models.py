import os
import logging

import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms

from core.rpn import AnchorGenerator
from core import MaskRCNN, maskrcnn_resnet50_fpn
from core.faster_rcnn import FastRCNNPredictor 
from core.mask_rcnn import MaskRCNNPredictor
from core.backbone_utils import (
    BackboneWithFPN,
    resnet_fpn_backbone
)

from torchvision.models.utils import load_state_dict_from_url 

from group_by_aspect_ratio import (
    GroupedBatchSampler,
    create_aspect_ratio_groups
)

from pytorchcv.model_provider import get_model as ptcv_get_model

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


def maskrcnn001(num_classes=2):
    # load backbone
    backbone = resnet_fpn_backbone("resnet50", pretrained=False)

    # hard load coco
    model = MaskRCNN(
        backbone, num_classes=91, # for coco

        # transform parameters
        min_size=800, max_size=1333,
        image_mean=None, image_std=None,

        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None,

        # Mask parameters
        mask_roi_pool=None, mask_head=None, mask_predictor=None 
    )

    state_dict = load_state_dict_from_url(
        "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        progress=True
    )
    model.load_state_dict(state_dict)

    # START HARD CUSTOM

    # # change anchor
    # anchor_generator = AnchorGenerator(
    #     sizes=((16, 32, 64, 128),),
    #     aspect_ratios=((0.5, 1.0, 2.0),)
    # )

    anchor_generator = AnchorGenerator(
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model.rpn_anchor_generator = anchor_generator

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


def maskrcnn_densenet121_rpn(num_classes=2):
    backbone = torchvision.models.densenet121(pretrained=True).features
    backbone.out_channels = 1024

    anchor_generator = AnchorGenerator(
        sizes=((128, 256, 512, 768),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )


    model = MaskRCNN(
        backbone, num_classes=num_classes,

        # transform parameters
        min_size=800, max_size=1333,
        image_mean=None, image_std=None,

        # RPN parameters
        rpn_anchor_generator=anchor_generator, rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None,

        # Mask parameters
        mask_roi_pool=None, mask_head=None, mask_predictor=None 
    )

    return model


def maskrcnn_inceptionresnetv2_rpn(num_classes=2):
    backbone = ptcv_get_model("InceptionResNetV2", pretrained=True).features
    backbone.out_channels = 1536

    anchor_generator = AnchorGenerator(
        sizes=((128, 256, 512, 768),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = MaskRCNN(
        backbone, num_classes=num_classes,

        # transform parameters
        min_size=800, max_size=1333,
        image_mean=None, image_std=None,

        # RPN parameters
        rpn_anchor_generator=anchor_generator, rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None,

        # Mask parameters
        mask_roi_pool=None, mask_head=None, mask_predictor=None 
    )

    return model


def maskrcnn_resnext101_32x8d_rpn(num_classes=2, pretrained=True):
    # backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone("resnet50", pretrained=False)
    backbone = resnet_fpn_backbone("resnext101_32x8d", pretrained=pretrained, norm_layer=None)

    # hard load coco
    model = MaskRCNN(
        backbone, num_classes=91, # for coco

        # transform parameters
        min_size=800, max_size=1333,
        image_mean=None, image_std=None,

        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None,

        # Mask parameters
        mask_roi_pool=None, mask_head=None, mask_predictor=None 
    )


    anchor_generator = AnchorGenerator(
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model.rpn_anchor_generator = anchor_generator

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
