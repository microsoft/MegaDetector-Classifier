"""DINOv3 linear-probe Lightning algorithm (Phase 4 MVP).

PEFT and full-FT are explicit NotImplementedError stubs landing in Phase 6/7.
"""
from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src import models as _models
from src.utils.mixup import efficient_batch_same_class_mixup, noisy_same_class_mixup

__all__ = ["DINOv3Linear"]


class DINOv3Linear(pl.LightningModule):
    """Linear-probe head over a DINOv3 backbone.

    Reads config from `conf.dinov3` (Munch / dict): `model_name`, `num_classes`
    or top-level `conf.num_classes`, `freeze_backbone`, `peft.enable`,
    `use_mixup`. PEFT and full-FT are deferred.
    """

    def __init__(
        self,
        conf: Any,
        train_class_counts: Optional[Any] = None,
        id_to_labels: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["conf", "train_class_counts", "id_to_labels"])
        self.conf = conf
        self.train_class_counts = train_class_counts
        self.id_to_labels = id_to_labels

        dinov3_cfg = conf.get("dinov3", {}) if hasattr(conf, "get") else getattr(conf, "dinov3", {})

        def _cfg(key, default=None):
            if isinstance(dinov3_cfg, dict):
                return dinov3_cfg.get(key, default)
            return getattr(dinov3_cfg, key, default)

        peft_cfg = _cfg("peft", {}) or {}
        if isinstance(peft_cfg, dict):
            peft_enabled = peft_cfg.get("enable", False)
        else:
            peft_enabled = getattr(peft_cfg, "enable", False)
        if peft_enabled:
            self._wrap_with_peft(peft_cfg)

        freeze_backbone = _cfg("freeze_backbone", True)
        model_name = _cfg("model_name")
        if model_name is None:
            raise ValueError(
                "conf.dinov3.model_name is required "
                "(e.g. 'facebook/dinov3-vits16-pretrain-lvd1689m')"
            )

        num_classes = _cfg("num_classes", None)
        if num_classes is None:
            num_classes = conf["num_classes"] if isinstance(conf, dict) else getattr(conf, "num_classes")

        self.net = _models.DINOv3Classifier(
            model_name=model_name,
            num_classes=int(num_classes),
            freeze_backbone=bool(freeze_backbone),
        )
        self.use_mixup = bool(_cfg("use_mixup", False))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        if self.use_mixup:
            x, y_a, y_b, lam = noisy_same_class_mixup(x, y)
            logits = self(x)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        lr = (
            getattr(self.conf, "lr_classifier", 1e-3)
            if hasattr(self.conf, "lr_classifier")
            else (self.conf.get("lr_classifier", 1e-3) if isinstance(self.conf, dict) else 1e-3)
        )
        return torch.optim.AdamW(trainable, lr=lr)

    def _wrap_with_peft(self, peft_config):
        """Stub. Real implementation lands in Phase 7."""
        raise NotImplementedError("PEFT lands in Phase 7")
