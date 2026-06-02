"""DINOv3 inference loader from a local .ckpt (Phase 5 v1)."""
from __future__ import annotations

from typing import List, Optional

import torch

from .base_classifier import DINOv3BaseInference, _extract_state_dict
from ...dinov3 import DINOv3Classifier

__all__ = ["DINOv3CustomWeights"]


class DINOv3CustomWeights(DINOv3BaseInference):
    """Load a DINOv3 classifier from a local Lightning checkpoint.

    v1 supports only MD-Cls ckpts saved by `src.algorithms.dinov3_linear.DINOv3Linear`.
    For SINCHI flavours, wait for Phase 8 (which extends `_extract_state_dict`).
    """

    IMAGE_SIZE = 518

    def __init__(
        self,
        weights,
        class_names: List[str],
        device: str = "cpu",
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        embedding_dim: Optional[int] = None,
    ):
        if class_names is None:
            raise ValueError("class_names is required")
        self.CLASS_NAMES = list(class_names)
        self.num_cls = len(self.CLASS_NAMES)
        self.device = device

        try:
            net = DINOv3Classifier(
                model_name=model_name,
                num_classes=self.num_cls,
                embedding_dim=embedding_dim,
                freeze_backbone=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct DINOv3Classifier(model_name={model_name!r}). "
                f"Original: {e}"
            ) from e

        super().__init__(net=net)

        if weights is None:
            raise ValueError("weights path is required for DINOv3CustomWeights")

        try:
            ckpt = torch.load(weights, map_location=torch.device(device), weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to torch.load({weights!r}); valid checkpoint? Original: {e}"
            ) from e

        state = _extract_state_dict(ckpt)

        cls_w = state.get("classifier.weight")
        if cls_w is not None:
            in_dim = cls_w.shape[1]
            if in_dim != self.net.embedding_dim:
                raise ValueError(
                    f"Checkpoint classifier expects in_features={in_dim} but "
                    f"DINOv3Classifier(model_name={model_name!r}) inferred "
                    f"embedding_dim={self.net.embedding_dim}. Pass the matching "
                    f"model_name or embedding_dim kwarg."
                )

        self.net.load_state_dict(state, strict=True)
        self.eval()
        self.to(device)
