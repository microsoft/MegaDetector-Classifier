"""DINOv3 inference loader for local checkpoints."""
from __future__ import annotations

from typing import Any, List, Optional

import torch

from .base_classifier import (
    DINOv3BaseInference,
    _extract_state_dict,
)
from ...dinov3 import DINOv3Classifier

__all__ = ["DINOv3CustomWeights"]

_FALLBACK_DINOV3_MODEL_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge_plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}
_FALLBACK_DINOV3_DIMS = {
    "small": 384,
    "small_plus": 384,
    "base": 768,
    "large": 1024,
    "huge_plus": 1280,
}


def _registry():
    try:
        from ...dinov3_registry import DINOV3_DIMS, DINOV3_MODEL_IDS, resolve_model_id
    except ImportError:
        DINOV3_DIMS = _FALLBACK_DINOV3_DIMS
        DINOV3_MODEL_IDS = _FALLBACK_DINOV3_MODEL_IDS

        def resolve_model_id(model_size_or_id: str) -> str:
            return DINOV3_MODEL_IDS.get(model_size_or_id, model_size_or_id)

    return DINOV3_MODEL_IDS, DINOV3_DIMS, resolve_model_id


def _resolve_model_name(
    *,
    explicit_model_name: Optional[str],
    model_size: Optional[str],
    checkpoint_model_name: Optional[str],
) -> str:
    model_ids, _, resolve_model_id = _registry()

    requested = explicit_model_name
    if requested is None and model_size is not None:
        requested = model_size
    if requested is None:
        requested = checkpoint_model_name
    if requested is None:
        raise ValueError(
            "Could not resolve DINOv3 model_name. Pass model_name/model_size "
            "or use a checkpoint with meta/config model_name."
        )

    resolved = resolve_model_id(requested) if requested in model_ids else requested
    if checkpoint_model_name is not None and (explicit_model_name is not None or model_size is not None):
        ckpt_resolved = (
            resolve_model_id(checkpoint_model_name)
            if checkpoint_model_name in model_ids
            else checkpoint_model_name
        )
        if resolved != ckpt_resolved:
            raise ValueError(
                f"Requested DINOv3 model {resolved!r} but checkpoint metadata "
                f"declares {ckpt_resolved!r}."
            )
    return resolved


def _resolve_embedding_dim(
    *,
    explicit_embedding_dim: Optional[int],
    meta: dict,
    model_size: Optional[str],
    model_name: str,
) -> Optional[int]:
    if explicit_embedding_dim is not None:
        return explicit_embedding_dim
    if meta.get("embedding_dim") is not None:
        return int(meta["embedding_dim"])

    model_ids, dims, _ = _registry()
    if model_size in dims:
        return dims[model_size]
    for alias, registry_model_name in model_ids.items():
        if registry_model_name == model_name and alias in dims:
            return dims[alias]
    return None


def _num_classes_from_state(state: dict) -> Optional[int]:
    cls_w = state.get("classifier.weight")
    shape = getattr(cls_w, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) >= 1 else None


def _normalise_class_names(
    class_names: Optional[List[str] | dict[int, str]],
    *,
    meta: dict,
    num_classes: int,
) -> list[str]:
    if class_names is None:
        class_names = meta.get("class_names")
    if class_names is None:
        return [str(i) for i in range(num_classes)]
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names)]
    class_names = list(class_names)
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length {len(class_names)} does not match "
            f"checkpoint num_classes {num_classes}."
        )
    return class_names


def _feature_hidden_size(net: DINOv3Classifier) -> int:
    cfg = getattr(getattr(net, "feature", None), "config", None)
    hidden_size = getattr(cfg, "hidden_size", None) if cfg is not None else None
    return int(hidden_size if hidden_size is not None else net.embedding_dim)


def _wrap_peft_backbone(net: DINOv3Classifier, peft_config: dict) -> None:
    if not peft_config:
        raise ValueError("SINCHI PEFT checkpoint is missing config.peft metadata.")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "SINCHI PEFT inference requires `peft`; pass a merged/raw checkpoint "
            "or install peft — see plan §6 flavour C."
        ) from exc

    peft_kwargs = dict(peft_config)
    method = peft_kwargs.pop("method", None)
    if method is not None and str(method).lower() != "lora":
        raise ValueError(f"Unsupported SINCHI PEFT method {method!r}; expected 'lora'.")
    net.feature = get_peft_model(net.feature, LoraConfig(**peft_kwargs))


class DINOv3CustomWeights(DINOv3BaseInference):
    """Load a DINOv3 classifier from a local checkpoint."""

    IMAGE_SIZE = 518

    def __init__(
        self,
        weights,
        class_names: Optional[List[str] | dict[int, str]] = None,
        device: str = "cpu",
        model_size: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        if weights is None:
            raise ValueError("weights path is required for DINOv3CustomWeights")

        self.device = device
        try:
            ckpt = torch.load(weights, map_location=torch.device(device), weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to torch.load({weights!r}); valid checkpoint? Original: {e}"
            ) from e

        state, meta = _extract_state_dict(ckpt, return_meta=True)

        resolved_model_name = _resolve_model_name(
            explicit_model_name=model_name,
            model_size=model_size,
            checkpoint_model_name=meta.get("model_name"),
        )
        num_classes = (
            int(meta["num_classes"])
            if meta.get("num_classes") is not None
            else _num_classes_from_state(state)
        )
        if num_classes is None:
            raise ValueError(
                "Could not resolve num_classes from checkpoint metadata or classifier.weight."
            )
        resolved_class_names = _normalise_class_names(
            class_names,
            meta=meta,
            num_classes=num_classes,
        )
        resolved_embedding_dim = _resolve_embedding_dim(
            explicit_embedding_dim=embedding_dim,
            meta=meta,
            model_size=model_size,
            model_name=resolved_model_name,
        )

        try:
            net = DINOv3Classifier(
                model_name=resolved_model_name,
                num_classes=num_classes,
                embedding_dim=resolved_embedding_dim,
                freeze_backbone=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct DINOv3Classifier(model_name={resolved_model_name!r}). "
                f"Original: {e}"
            ) from e

        if meta.get("flavor") == "sinchi_peft":
            _wrap_peft_backbone(net, meta.get("peft"))

        super().__init__(
            net=net,
            device=device,
            class_names=resolved_class_names,
        )
        self.num_cls = num_classes
        self.CLASS_NAMES = resolved_class_names
        self.checkpoint_meta: dict[str, Any] = meta

        cls_w = state.get("classifier.weight")
        if cls_w is None:
            raise ValueError("Checkpoint is missing classifier.weight.")
        if cls_w.shape[1] != _feature_hidden_size(self.net):
            raise ValueError(
                f"Checkpoint classifier expects in_features={cls_w.shape[1]} but "
                f"DINOv3 model {resolved_model_name!r} exposes hidden_size="
                f"{_feature_hidden_size(self.net)}."
            )
        if cls_w.shape[0] != num_classes:
            raise ValueError(
                f"Checkpoint classifier has {cls_w.shape[0]} rows but resolved "
                f"num_classes={num_classes}."
            )

        strict = meta.get("flavor") != "classifier_only"
        self.net.load_state_dict(state, strict=strict)
        self.eval()
        self.to(device)
