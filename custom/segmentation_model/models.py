import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN

import os
import sys


# https://github.com/soyeonm/FILM/blob/fa4aefda700a58b54d78a7e8e996f7c977cdd045/models/segmentation/alfworld_mrcnn.py#L83
# https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code/blob/main/train.py
def get_model_instance_segmentation(num_classes: int, hidden_size: int):
    model: MaskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # replace parts of model with new ones.
    anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]),
    )
    model.rpn.anchor_generator = anchor_generator

    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_size, num_classes)

    return model
    