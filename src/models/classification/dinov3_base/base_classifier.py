"""DINOv3 checkpoint loading and inference utilities."""
from __future__ import annotations

import hashlib
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
        meta = dict(ckpt.get("meta") or {})
        meta.setdefault("flavor", "md_cls")
        if "num_classes" not in meta:
            num_classes = _classifier_num_classes(state)
            if num_classes is not None:
                meta["num_classes"] = num_classes
        return (state, meta) if return_meta else state

    has_classifier_only = (
        "classifier.weight" in ckpt
        and "classifier.bias" in ckpt
        and isinstance(ckpt.get("meta"), dict)
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
            meta = dict(ckpt["meta"])
            meta.setdefault("flavor", "classifier_only")
            meta.setdefault("num_classes", _classifier_num_classes(state))
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
            meta = _base_meta(
                ckpt,
                state,
                flavor="sinchi_peft",
                peft=cfg_peft or peft_config,
            )
            return (state, meta) if return_meta else state

        if has_backbone and has_classifier and not has_peft_base:
            state = _remap_backbone_state(model_state)
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
        self.device = device or getattr(self, "device", "cpu")
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
        safe_name = str(image_path).replace("\\", "_").replace("/", "_").replace(":", "_")
        digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
        return Path(save_embeddings_dir) / f"{safe_name}.{digest}.pkl"

    @staticmethod
    def _embedding_record(embedding: np.ndarray, image_path: str) -> dict:
        return {
            "embedding": embedding.astype(np.float32, copy=False),
            "label": -1,
            "image_path": str(image_path),
        }

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
        _, embeddings = self.batch_image_classification_with_embeddings(
            data_path=data_path,
            det_results=det_results,
            save_embeddings_dir=save_embeddings_dir,
            return_embeddings=True,
            batch_size=batch_size,
            num_workers=num_workers,
            id_strip=id_strip,
        )
        return embeddings
