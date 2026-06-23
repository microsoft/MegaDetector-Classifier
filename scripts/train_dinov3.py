"""Stage-2 CLI for training a DINOv3 linear head on cached embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 linear probe from cached embeddings")
    parser.add_argument("--config", default="configs/dinov3_linear.yaml")
    return parser.parse_args()


def _load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return data


def _class_names_from_label_mapping(label_to_idx: Dict[Any, int], num_classes: int) -> List[str]:
    idx_to_label = {int(idx): label for label, idx in label_to_idx.items()}
    return [str(idx_to_label.get(index, index)) for index in range(num_classes)]


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    from src.algorithms.dinov3_linear import DINOv3LinearProbe
    from src.datasets.dinov3_embedding import DINOv3EmbeddingDataset, build_label_mapping
    from src.models.dinov3_registry import DINOV3_DIMS, resolve_model_id

    model_size = config.get("model_size", "base")
    if model_size not in DINOV3_DIMS:
        raise ValueError(f"Unsupported model_size {model_size!r}; expected one of {sorted(DINOV3_DIMS)}")

    embeddings_dir = Path(config["embeddings_dir"])
    train_csv = Path(config["train_csv"])
    val_csv = Path(config["val_csv"])
    photo_col = config.get("photo_col", "Photo_ID")
    label_col = config.get("label_col", "Label")

    train_label_to_idx = build_label_mapping(train_csv, label_col=label_col)
    val_label_to_idx = build_label_mapping(val_csv, label_col=label_col)
    labels: List[Any] = []
    for mapping in (train_label_to_idx, val_label_to_idx):
        for label in mapping:
            if label not in labels:
                labels.append(label)
    labels = sorted(labels, key=lambda value: str(value))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_dataset = DINOv3EmbeddingDataset(
        embeddings_dir=embeddings_dir,
        csv_path=train_csv,
        split="train",
        label_to_idx=label_to_idx,
        photo_col=photo_col,
        label_col=label_col,
    )
    val_dataset = DINOv3EmbeddingDataset(
        embeddings_dir=embeddings_dir,
        csv_path=val_csv,
        split="val",
        label_to_idx=label_to_idx,
        photo_col=photo_col,
        label_col=label_col,
    )

    inferred_num_classes = len(label_to_idx)
    configured_num_classes = config.get("num_classes")
    num_classes = int(configured_num_classes) if configured_num_classes is not None else inferred_num_classes
    if num_classes < inferred_num_classes:
        raise ValueError(
            f"num_classes={num_classes} is smaller than inferred label mapping size {inferred_num_classes}"
        )

    batch_size = int(config.get("batch_size", 256))
    num_workers = int(config.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = DINOv3LinearProbe(
        embedding_dim=int(DINOV3_DIMS[model_size]),
        num_classes=num_classes,
        lr=float(config.get("lr", 1e-3)),
        label_smoothing=float(config.get("label_smoothing", 0.1)),
        class_names=_class_names_from_label_mapping(label_to_idx, num_classes),
        model_name=resolve_model_id(model_size),
        model_size=model_size,
        image_size=int(config.get("image_size", 518)),
    )

    seed = config.get("seed")
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    trainer = pl.Trainer(
        max_epochs=int(config.get("epochs", 30)),
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", "auto"),
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)

    output_dir = Path(config.get("output_dir", "./outputs/dinov3_linear"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_model.pt"
    torch.save(model.export_classifier_only(), output_path)
    print(f"Saved classifier-only DINOv3 checkpoint to {output_path}")


if __name__ == "__main__":
    main()
