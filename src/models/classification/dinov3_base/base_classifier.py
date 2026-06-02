"""DINOv3 inference module (Phase 5 v1).

v1 handles only the 'state_dict' + 'net.' prefix flavour (own MD-Cls ckpts
saved by P4's DINOv3Linear). SINCHI flavours raise NotImplementedError
until Phase 8.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["DINOv3BaseInference"]


def _extract_state_dict(ckpt: dict, peft_config: Optional[dict] = None) -> dict:
    """Extract & remap the backbone state dict from a Lightning checkpoint (v1).

    v1 handles ONLY:
      (1) MD-Cls own ckpts: ckpt['state_dict'] contains 'net.*' keys.

    SINCHI raw and SINCHI PEFT flavours raise NotImplementedError — they
    land in Phase 8.

    Args:
        ckpt: torch.load output (dict).
        peft_config: ignored in v1 (reserved for Phase 8).

    Returns:
        dict mapping backbone keys (e.g. 'feature.*', 'classifier.*')
        to tensors.
    """
    # (1) MD-Cls v1 (own)
    if "state_dict" in ckpt and any(k.startswith("net.") for k in ckpt["state_dict"]):
        return {
            k[len("net."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("net.")
        }

    # (2) SINCHI raw → P8
    if "model_state_dict" in ckpt and any(
        k.startswith("backbone.") for k in ckpt["model_state_dict"]
    ):
        raise NotImplementedError("SINCHI checkpoint flavours land in Phase 8")

    # (3) SINCHI PEFT → P8
    if "model_state_dict" in ckpt and any(
        "base_model.model." in k for k in ckpt["model_state_dict"]
    ):
        raise NotImplementedError("SINCHI checkpoint flavours land in Phase 8")

    raise ValueError(
        f"Unrecognised checkpoint format. Top-level keys: {list(ckpt.keys())[:10]}"
    )


class DINOv3BaseInference(nn.Module):
    """Lightweight wrapper holding a DINOv3Classifier `net` attribute.

    Mirrors the role of `PlainResNetInference` for ResNet ckpts. Subclasses
    (e.g. DINOv3CustomWeights) call `_extract_state_dict` + `load_state_dict`.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
