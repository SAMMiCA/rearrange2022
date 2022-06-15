import os
from PIL import Image
import torch
import numpy as np
from torchvision.datasets.coco import CocoDetection


class SegmentationCOCODataset(CocoDetection):

    def __init__(self, root_dir: str, mode: str, transform=None, target_transform=None) -> None:
        root = os.path.join(os.path.expanduser(root_dir), mode)
        annFile = os.path.join(os.path.expanduser(root_dir), f"{mode}_anno.json")
        super().__init__(root=root, annFile=annFile, transform=transform, target_transform=target_transform)


