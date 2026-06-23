"""DINOv3 classifier with frozen backbone (Phase 4 MVP).

Lazy `transformers` import inside __init__ keeps default-env `import src` clean
(invariant I1). PEFT wrapping is deferred to Phase 7.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["DINOv3Classifier"]


class DINOv3Classifier(nn.Module):
    """Linear-probe classifier on top of a frozen DINOv3 backbone.

    Args:
        model_name: HuggingFace Hub repo id (e.g. "facebook/dinov3-vits16-pretrain-lvd1689m").
        num_classes: Number of output classes.
        embedding_dim: Backbone embedding dim. If None, inferred from
            backbone.config.hidden_size.
        freeze_backbone: If True (default), all backbone params get
            requires_grad=False. False reserved for Phase 6 — raises
            NotImplementedError("Full FT lands in Phase 6").
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        embedding_dim: Optional[int] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if not freeze_backbone:
            raise NotImplementedError("Full FT lands in Phase 6")

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "DINOv3Classifier requires the optional DINOv3 dependencies. "
                "Install with: pip install -r requirements-dinov3.txt "
                f"(original error: {e})"
            ) from e

        try:
            from huggingface_hub.utils import HfHubHTTPError
        except ImportError:
            HfHubHTTPError = OSError

        try:
            self.feature = AutoModel.from_pretrained(model_name)
        except (HfHubHTTPError, OSError) as e:
            raise RuntimeError(
                f"Failed to download {model_name}. "
                "If gated: run 'huggingface-cli login'. "
                "If offline: set HF_HOME and TRANSFORMERS_OFFLINE=1. "
                f"Original: {e}"
            ) from e

        if embedding_dim is None:
            cfg = getattr(self.feature, "config", None)
            embedding_dim = getattr(cfg, "hidden_size", None) if cfg is not None else None
            if embedding_dim is None:
                raise ValueError(
                    f"Could not infer embedding_dim from model {model_name}; "
                    "pass embedding_dim=... explicitly."
                )
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(embedding_dim, num_classes)

        for p in self.feature.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (B, num_classes)."""
        out = self.feature(pixel_values=x)
        feats = out.last_hidden_state[:, 0, :]
        return self.classifier(feats)

    @staticmethod
    def _cls_embedding_from_output(out) -> torch.Tensor:
        last_hidden_state = None
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            last_hidden_state = out.last_hidden_state
        elif isinstance(out, dict) and out.get("last_hidden_state") is not None:
            last_hidden_state = out["last_hidden_state"]
        elif isinstance(out, (tuple, list)) and out:
            candidate = out[0]
            if hasattr(candidate, "last_hidden_state") and candidate.last_hidden_state is not None:
                last_hidden_state = candidate.last_hidden_state
            elif isinstance(candidate, dict) and candidate.get("last_hidden_state") is not None:
                last_hidden_state = candidate["last_hidden_state"]
            elif torch.is_tensor(candidate):
                last_hidden_state = candidate
        elif torch.is_tensor(out):
            last_hidden_state = out
        if torch.is_tensor(last_hidden_state) and last_hidden_state.ndim == 3:
            return last_hidden_state[:, 0, :]
        raise TypeError(
            "DINOv3 backbone output must expose a 3-D last_hidden_state tensor "
            "so CLS embeddings are taken from token index 0."
        )

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embeddings from the frozen backbone."""
        def _run_backbone() -> torch.Tensor:
            out = self.feature(pixel_values=x)
            return self._cls_embedding_from_output(out)

        if self.training:
            return _run_backbone()
        with torch.no_grad():
            return _run_backbone()
