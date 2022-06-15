from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from torchvision.transforms import transforms as T, InterpolationMode
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                *_, width = F.get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)

        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        
        return image, target
