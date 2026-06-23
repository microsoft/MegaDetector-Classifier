"""DINOv3 inference transform matching SINCHI training preprocessing."""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

__all__ = ["DINOv3Transform"]


class DINOv3Transform:
    """Bicubic 518px ImageNet-normalized DINOv3 transform."""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, target_size=518):
        self.target_size = target_size
        self.trans = transforms.Compose([
            transforms.Resize(
                (target_size, target_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, img) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            img = TF.to_pil_image(img).convert("RGB")
        return self.trans(img)
