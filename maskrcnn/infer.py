import os
import sys
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms

import cv2
import numpy as np

from utils import (
    overlay_ann,
    overlay_mask,
    show
)

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


CATEGORIES2LABELS = {
    0: "bg",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model


def main(argv):
    num_classes = 6
    model = get_instance_segmentation_model(num_classes)
    model.cuda()

    checkpoint_path = "/home/z/Downloads/model_12000.pth"

    assert os.path.exists(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # NOTE: custom  image
    if len(argv) > 0 and os.path.exists(argv[0]):
        image_path = argv[0]
    else:
        image_path = '/home/z/Downloads/hor02_013.A.A0001.png'

    assert os.path.exists(image_path)

    image = cv2.imread(image_path)
    rat = 1300 / image.shape[0]
    image = cv2.resize(image, None, fx=rat, fy=rat)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = transform(image)

    with torch.no_grad():
        prediction = model([image.cuda()])

    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.7:
                continue

            m = mask[0].mul(255).byte().cpu().numpy()
            box = list(map(int, pred["boxes"][idx].tolist()))
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]

            score = pred["scores"][idx].item()

            image = overlay_mask(image, m)
            # image = overlay_ann(image, m, box, label, score)

    cv2.imwrite('/home/z/research/publaynet/example_images/{}'.format(os.path.basename(image_path)), image)
    # show(image)


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    main(argv)
