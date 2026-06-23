"""DINOv3 checkpoint loading and inference utilities."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ....data import datasets as pw_data

__all__ = ["DINOv3BaseInference", "_extract_state_dict", "extract_checkpoint_meta"]

_CLASSIFIER_ONLY_FORMAT = "dinov3_classifier_only"
_CLASSIFIER_ONLY_VERSIONS = {1}
_CLASSIFIER_ONLY_META_KEYS = (
    "format",
    "version",
    "model_name",
    "model_size",
    "embedding_dim",
    "num_classes",
    "image_size",
    "feature_type",
    "class_names",
    "transform",
    "transform_version",
)


def _cfg_get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _classifier_num_classes(state: dict) -> Optional[int]:
    weight = state.get("classifier.weight")
    shape = getattr(weight, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) >= 1 else None


def _validate_classifier_tensors(state: dict, *, context: str) -> tuple[int, int]:
    weight = state.get("classifier.weight")
    bias = state.get("classifier.bias")
    if weight is None:
        raise ValueError(f"{context} is missing classifier.weight.")
    if bias is None:
        raise ValueError(f"{context} is missing classifier.bias.")
    weight_shape = getattr(weight, "shape", None)
    bias_shape = getattr(bias, "shape", None)
    if weight_shape is None or len(weight_shape) != 2:
        raise ValueError(
            f"{context} classifier.weight must be rank-2 "
            f"(num_classes, hidden_size); got shape {weight_shape}."
        )
    if bias_shape is None or len(bias_shape) != 1:
        raise ValueError(
            f"{context} classifier.bias must be rank-1 (num_classes,); "
            f"got shape {bias_shape}."
        )
    rows = int(weight_shape[0])
    cols = int(weight_shape[1])
    if int(bias_shape[0]) != rows:
        raise ValueError(
            f"{context} classifier.bias length {int(bias_shape[0])} does not "
            f"match classifier.weight rows {rows}."
        )
    return rows, cols


def _int_meta(meta: dict, key: str, *, context: str) -> Optional[int]:
    if meta.get(key) is None:
        return None
    try:
        return int(meta[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} metadata field {key!r} must be an integer.") from exc


def _ordered_class_names(class_names) -> Optional[list[str]]:
    if class_names is None:
        return None
    if isinstance(class_names, dict):
        def _sort_key(item):
            key = item[0]
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))

        return [str(value) for _, value in sorted(class_names.items(), key=_sort_key)]
    return [str(value) for value in class_names]


def _classifier_only_meta(ckpt: dict) -> dict:
    meta = dict(ckpt.get("meta") or {}) if isinstance(ckpt.get("meta"), dict) else {}
    for key in _CLASSIFIER_ONLY_META_KEYS:
        if key in ckpt and key not in meta:
            meta[key] = ckpt[key]
    return meta


def _validate_classifier_only_meta(meta: dict, state: dict, *, declared: bool) -> dict:
    rows, cols = _validate_classifier_tensors(state, context="DINOv3 classifier-only checkpoint")
    meta = dict(meta)
    meta.setdefault("flavor", "classifier_only")

    if declared:
        if meta.get("format") != _CLASSIFIER_ONLY_FORMAT:
            raise ValueError(
                "DINOv3 classifier-only checkpoint format must be "
                f"{_CLASSIFIER_ONLY_FORMAT!r}; got {meta.get('format')!r}."
            )
        version = _int_meta(meta, "version", context="DINOv3 classifier-only checkpoint")
        if version not in _CLASSIFIER_ONLY_VERSIONS:
            raise ValueError(
                f"Unsupported DINOv3 classifier-only checkpoint version {version!r}; "
                f"supported versions: {sorted(_CLASSIFIER_ONLY_VERSIONS)}."
            )
        for key in ("model_name", "embedding_dim", "num_classes", "image_size"):
            if meta.get(key) is None:
                raise ValueError(
                    f"DINOv3 classifier-only checkpoint is missing required "
                    f"metadata field {key!r}."
                )

    feature_type = meta.get("feature_type")
    if feature_type is not None and feature_type != "cls":
        raise ValueError(
            f"DINOv3 classifier-only checkpoint feature_type must be 'cls'; "
            f"got {feature_type!r}."
        )
    transform = meta.get("transform")
    if transform is not None and transform != "dinov3_bicubic_518_imagenet":
        raise ValueError(
            "DINOv3 classifier-only checkpoint transform must be "
            f"'dinov3_bicubic_518_imagenet'; got {transform!r}."
        )
    transform_version = _int_meta(
        meta,
        "transform_version",
        context="DINOv3 classifier-only checkpoint",
    )
    if transform_version is not None and transform_version != 1:
        raise ValueError(
            "DINOv3 classifier-only checkpoint transform_version must be 1; "
            f"got {transform_version}."
        )

    meta_num_classes = _int_meta(
        meta,
        "num_classes",
        context="DINOv3 classifier-only checkpoint",
    )
    if meta_num_classes is not None and meta_num_classes != rows:
        raise ValueError(
            f"DINOv3 classifier-only checkpoint num_classes={meta_num_classes} "
            f"but classifier.weight has {rows} rows."
        )
    meta["num_classes"] = meta_num_classes if meta_num_classes is not None else rows

    meta_embedding_dim = _int_meta(
        meta,
        "embedding_dim",
        context="DINOv3 classifier-only checkpoint",
    )
    if meta_embedding_dim is not None and meta_embedding_dim != cols:
        raise ValueError(
            f"DINOv3 classifier-only checkpoint embedding_dim={meta_embedding_dim} "
            f"but classifier.weight has {cols} columns."
        )
    if meta_embedding_dim is not None:
        meta["embedding_dim"] = meta_embedding_dim

    class_names = _ordered_class_names(meta.get("class_names"))
    if class_names is not None:
        if len(class_names) != meta["num_classes"]:
            raise ValueError(
                f"DINOv3 classifier-only checkpoint class_names length "
                f"{len(class_names)} does not match num_classes={meta['num_classes']}."
            )
        meta["class_names"] = class_names

    return meta


def _base_meta(ckpt: dict, state: dict, flavor: str, peft: Optional[dict]) -> dict:
    config = ckpt.get("config") or {}
    meta = {
        "model_name": _cfg_get(config, "model", "name"),
        "num_classes": _cfg_get(config, "model", "num_classes")
        or _classifier_num_classes(state),
        "image_size": _cfg_get(config, "data", "image_size"),
        "peft": peft,
        "flavor": flavor,
    }
    return meta


def _remap_backbone_state(state: dict) -> dict:
    remapped = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            remapped[f"feature.{key[len('backbone.'):]}"] = value
        elif key.startswith("classifier."):
            remapped[key] = value
    return remapped


def _extract_state_dict(
    ckpt: dict,
    peft_config: Optional[dict] = None,
    *,
    return_meta: bool = False,
):
    """Extract supported DINOv3 checkpoint flavours.

    Supported structural flavours:
    A. MD-Cls own Lightning ckpt: ``ckpt["state_dict"]`` contains ``net.*``.
    B. SINCHI raw: ``model_state_dict`` contains ``backbone.*`` + classifier.
    C. SINCHI PEFT: ``model_state_dict`` contains PEFT-wrapped backbone keys.
    D. classifier-only: top-level classifier tensors plus ``meta``.
    """
    meta: dict = {}

    if "state_dict" in ckpt and any(k.startswith("net.") for k in ckpt["state_dict"]):
        state = {
            k[len("net."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("net.")
        }
        _validate_classifier_tensors(state, context="DINOv3 md_cls checkpoint")
        meta = dict(ckpt.get("meta") or {})
        meta.setdefault("flavor", "md_cls")
        if "num_classes" not in meta:
            num_classes = _classifier_num_classes(state)
            if num_classes is not None:
                meta["num_classes"] = num_classes
        return (state, meta) if return_meta else state

    classifier_meta = _classifier_only_meta(ckpt)
    classifier_format = classifier_meta.get("format")
    has_classifier_only = (
        "classifier.weight" in ckpt
        and "classifier.bias" in ckpt
        and (
            classifier_format == _CLASSIFIER_ONLY_FORMAT
            or isinstance(ckpt.get("meta"), dict)
        )
    )
    if has_classifier_only:
        backbone_keys = [
            k for k in ckpt
            if k.startswith(("backbone.", "feature.", "net.feature."))
        ]
        if not backbone_keys:
            state = {
                "classifier.weight": ckpt["classifier.weight"],
                "classifier.bias": ckpt["classifier.bias"],
            }
            meta = _validate_classifier_only_meta(
                classifier_meta,
                state,
                declared=classifier_format == _CLASSIFIER_ONLY_FORMAT,
            )
            return (state, meta) if return_meta else state

    if "model_state_dict" in ckpt:
        model_state = ckpt["model_state_dict"]
        keys = list(model_state)
        has_backbone = any(k.startswith("backbone.") for k in keys)
        has_classifier = "classifier.weight" in model_state and "classifier.bias" in model_state
        has_peft_base = any(k.startswith("backbone.base_model.model.") for k in keys)
        has_lora = any(".lora_" in k or "lora_" in k for k in keys)
        cfg_peft = _cfg_get(ckpt.get("config") or {}, "peft", default=None)

        if has_peft_base and (has_lora or cfg_peft):
            state = _remap_backbone_state(model_state)
            _validate_classifier_tensors(state, context="DINOv3 SINCHI PEFT checkpoint")
            meta = _base_meta(
                ckpt,
                state,
                flavor="sinchi_peft",
                peft=cfg_peft or peft_config,
            )
            return (state, meta) if return_meta else state

        if has_backbone and has_classifier and not has_peft_base:
            state = _remap_backbone_state(model_state)
            _validate_classifier_tensors(state, context="DINOv3 SINCHI RAW checkpoint")
            meta = _base_meta(ckpt, state, flavor="sinchi_raw", peft=None)
            return (state, meta) if return_meta else state

        if has_backbone:
            raise NotImplementedError("SINCHI checkpoint flavours land in Phase 8")

    raise ValueError(
        f"Unrecognised checkpoint format. Top-level keys: {list(ckpt.keys())[:10]}"
    )


def extract_checkpoint_meta(ckpt: dict) -> dict:
    """Return checkpoint metadata without requiring callers to load state."""
    _, meta = _extract_state_dict(ckpt, return_meta=True)
    return meta


class DINOv3BaseInference(nn.Module):
    """Inference wrapper for a DINOv3Classifier-style ``net``."""

    IMAGE_SIZE = 518
    CLASS_NAMES: dict[int, str] | list[str] = {}

    def __init__(
        self,
        net: nn.Module,
        transform=None,
        device: Optional[str] = None,
        class_names: Optional[list[str] | dict[int, str]] = None,
    ):
        super().__init__()
        self.net = net
        self.predictor = net
        self.device = torch.device(device or getattr(self, "device", "cpu"))
        self.net.to(self.device)
        if self.predictor is not self.net and hasattr(self.predictor, "to"):
            self.predictor.to(self.device)
        if class_names is not None:
            self.CLASS_NAMES = class_names
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _transform(self):
        if self.transform is None:
            from .transforms import DINOv3Transform

            self.transform = DINOv3Transform(target_size=self.IMAGE_SIZE)
        return self.transform

    def _class_name(self, idx: int) -> str:
        return self.CLASS_NAMES[idx]

    def results_generation(
        self,
        logits: torch.Tensor,
        img_ids: list[str],
        id_strip: str = None,
    ) -> list[dict]:
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1)[0]

        results = []
        for pred, img_id, conf, row_probs in zip(preds, img_ids, confs, probs):
            confidences = row_probs.tolist()
            all_confidences = [
                [self._class_name(i), confidence]
                for i, confidence in enumerate(confidences)
            ]
            class_id = pred.item()
            result = {"img_id": str(img_id).strip(id_strip)}
            result["prediction"] = self._class_name(class_id)
            result["class_id"] = class_id
            result["confidence"] = conf.item()
            result["all_confidences"] = all_confidences
            results.append(result)

        return results

    def single_image_classification(self, img, img_id=None, id_strip=None):
        if isinstance(img, (str, os.PathLike)):
            if img_id is None:
                img_id = str(img)
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            img = Image.fromarray(img).convert("RGB")

        transformed = self._transform()(img).unsqueeze(0).to(self.device)
        self.eval()
        with torch.inference_mode():
            logits = self.net(transformed)
        return self.results_generation(logits.cpu(), [img_id], id_strip=id_strip)[0]

    def _build_dataset(self, data_path=None, det_results=None):
        if data_path:
            return pw_data.ImageFolder(
                data_path,
                transform=self._transform(),
                path_head=".",
            )
        if det_results:
            return pw_data.DetectionCrops(
                det_results,
                transform=self._transform(),
                path_head=".",
            )
        raise Exception("Need data for inference.")

    def batch_image_classification(
        self,
        data_path=None,
        det_results=None,
        id_strip=None,
        batch_size=32,
        num_workers=0,
    ):
        dataset = self._build_dataset(data_path=data_path, det_results=det_results)
        if len(dataset) == 0:
            return []
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        total_logits = []
        total_paths = []
        self.eval()
        with torch.inference_mode(), tqdm(total=len(dataloader)) as pbar:
            for imgs, paths in dataloader:
                imgs = imgs.to(self.device)
                total_logits.append(self.net(imgs).cpu())
                total_paths.extend(list(paths))
                pbar.update(1)

        return self.results_generation(
            torch.cat(total_logits, dim=0),
            total_paths,
            id_strip=id_strip,
        )

    def _embeddings_and_logits(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.net, "get_embeddings") or not hasattr(self.net, "classifier"):
            raise AttributeError(
                "Embedding extraction requires a DINOv3Classifier-style net "
                "with get_embeddings() and classifier attributes."
            )
        embeddings = self.net.get_embeddings(imgs)
        logits = self.net.classifier(embeddings)
        return embeddings, logits

    @staticmethod
    def _embedding_pickle_path(save_embeddings_dir, image_path: str) -> Path:
        from ....datasets.dinov3_embedding import safe_name

        return Path(save_embeddings_dir) / f"{safe_name(str(image_path))}.pkl"

    @staticmethod
    def _embedding_record(embedding: np.ndarray, image_path: str) -> dict:
        return {
            "embedding": embedding.astype(np.float32, copy=False),
            "label": -1,
            "image_path": str(image_path),
        }

    @staticmethod
    def _register_embedding_output(
        seen_outputs: dict[str, str],
        out_path: Path,
        image_path: str,
    ) -> None:
        output_name = out_path.name
        previous = seen_outputs.get(output_name)
        if previous is not None and previous != image_path:
            raise ValueError(
                "DINOv3 embedding filename collision: "
                f"{previous!r} and {image_path!r} both map to {output_name!r}. "
                "Use unique paths before saving cached embeddings."
            )
        seen_outputs[output_name] = image_path

    def batch_image_classification_with_embeddings(
        self,
        data_path=None,
        det_results=None,
        save_embeddings_dir=None,
        return_embeddings=False,
        batch_size=32,
        num_workers=0,
        id_strip=None,
    ):
        dataset = self._build_dataset(data_path=data_path, det_results=det_results)
        if save_embeddings_dir is not None:
            os.makedirs(save_embeddings_dir, exist_ok=True)
        if len(dataset) == 0:
            return ([], []) if return_embeddings else []

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        total_logits = []
        total_paths = []
        embedding_records = []
        seen_embedding_outputs: dict[str, str] = {}
        self.eval()
        with torch.inference_mode(), tqdm(total=len(dataloader)) as pbar:
            for imgs, paths in dataloader:
                imgs = imgs.to(self.device)
                embeddings, logits = self._embeddings_and_logits(imgs)
                total_logits.append(logits.cpu())
                batch_paths = list(paths)
                total_paths.extend(batch_paths)

                embeddings_np = embeddings.detach().cpu().numpy()
                for embedding, image_path in zip(embeddings_np, batch_paths):
                    record = self._embedding_record(embedding, str(image_path))
                    if save_embeddings_dir is not None:
                        out_path = self._embedding_pickle_path(save_embeddings_dir, str(image_path))
                        self._register_embedding_output(
                            seen_embedding_outputs,
                            out_path,
                            str(image_path),
                        )
                        with out_path.open("wb") as f:
                            pickle.dump(record, f)
                    if return_embeddings:
                        embedding_records.append(record)
                pbar.update(1)

        predictions = self.results_generation(
            torch.cat(total_logits, dim=0),
            total_paths,
            id_strip=id_strip,
        )
        if return_embeddings:
            return predictions, embedding_records
        return predictions

    def extract_embeddings(
        self,
        data_path=None,
        det_results=None,
        save_embeddings_dir=None,
        batch_size=32,
        num_workers=0,
        id_strip=None,
    ) -> list[dict]:
        if not hasattr(self.net, "get_embeddings"):
            raise AttributeError(
                "Embedding extraction requires a DINOv3Classifier-style net "
                "with get_embeddings()."
            )
        dataset = self._build_dataset(data_path=data_path, det_results=det_results)
        if save_embeddings_dir is not None:
            os.makedirs(save_embeddings_dir, exist_ok=True)
        if len(dataset) == 0:
            return []

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        embedding_records = []
        seen_embedding_outputs: dict[str, str] = {}
        self.eval()
        with torch.inference_mode(), tqdm(total=len(dataloader)) as pbar:
            for imgs, paths in dataloader:
                imgs = imgs.to(self.device)
                embeddings = self.net.get_embeddings(imgs)
                embeddings_np = embeddings.detach().cpu().numpy()
                for embedding, image_path in zip(embeddings_np, list(paths)):
                    image_path = str(image_path)
                    record = self._embedding_record(embedding, image_path)
                    if save_embeddings_dir is not None:
                        out_path = self._embedding_pickle_path(save_embeddings_dir, image_path)
                        self._register_embedding_output(
                            seen_embedding_outputs,
                            out_path,
                            image_path,
                        )
                        with out_path.open("wb") as f:
                            pickle.dump(record, f)
                    embedding_records.append(record)
                pbar.update(1)
        return embedding_records
