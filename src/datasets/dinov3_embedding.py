"""Datasets for cached DINOv3 CLS embeddings."""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "safe_name",
    "build_label_mapping",
    "DINOv3EmbeddingDataset",
    "DINOv3EmbeddingDataModule",
]


def safe_name(image_path: str) -> str:
    """Return the SINCHI-compatible filename stem for an image path."""
    return str(image_path).replace("/", "_").replace("\\", "_")


def build_label_mapping(csv_path: str | Path, label_col: str = "Label") -> Dict[Any, int]:
    """Build a stable label-to-index mapping from a CSV label column."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise KeyError(f"CSV {csv_path} is missing required label column {label_col!r}")

    labels = sorted(df[label_col].dropna().unique().tolist(), key=lambda value: str(value))
    return {label: idx for idx, label in enumerate(labels)}


class DINOv3EmbeddingDataset(Dataset):
    """Load cached DINOv3 embeddings saved as one pickle per image.

    Pickle files follow the SINCHI schema and live at
    ``<embeddings_dir>/<split>/<safe_name(Photo_ID)>.pkl``. Each pickle may be
    either a dict with an ``embedding`` key or a raw numpy array / tensor.
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        csv_path: str | Path,
        split: str = "train",
        label_to_idx: Optional[Mapping[Any, int]] = None,
        photo_col: str = "Photo_ID",
        label_col: str = "Label",
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.csv_path = Path(csv_path)
        self.split = split
        self.photo_col = photo_col
        self.label_col = label_col
        self.split_dir = self.embeddings_dir / split

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Embedding split directory not found: {self.split_dir}")

        self.df = pd.read_csv(self.csv_path)
        for column in (photo_col, label_col):
            if column not in self.df.columns:
                raise KeyError(f"CSV {self.csv_path} is missing required column {column!r}")

        self.label_to_idx = dict(label_to_idx) if label_to_idx is not None else build_label_mapping(
            self.csv_path,
            label_col=label_col,
        )
        self.idx_to_label = {int(idx): label for label, idx in self.label_to_idx.items()}
        self.num_classes = max(self.idx_to_label.keys(), default=-1) + 1

        self.image_paths: List[str] = []
        self.embedding_paths: List[Path] = []
        self.labels: List[int] = []

        missing_count = 0
        for _, row in self.df.iterrows():
            image_path = str(row[photo_col])
            label_key = row[label_col]
            if label_key not in self.label_to_idx and str(label_key) in self.label_to_idx:
                label_key = str(label_key)
            if label_key not in self.label_to_idx:
                raise KeyError(f"Label {label_key!r} from {self.csv_path} is not in label_to_idx")

            embedding_path = self.split_dir / f"{safe_name(image_path)}.pkl"
            if not embedding_path.exists():
                missing_count += 1
                if missing_count <= 5:
                    warnings.warn(
                        f"Embedding not found for {image_path}; expected {embedding_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue

            self.image_paths.append(image_path)
            self.embedding_paths.append(embedding_path)
            self.labels.append(int(self.label_to_idx[label_key]))

        if missing_count > 5:
            warnings.warn(
                f"Skipped {missing_count} rows with missing cached embeddings in {self.split_dir}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self.embedding_paths:
            raise ValueError(f"No cached embeddings found for split {split!r} in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.embedding_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        embedding_path = self.embedding_paths[idx]
        try:
            with embedding_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Error loading embedding from {embedding_path}: {exc}") from exc

        if isinstance(payload, dict):
            if "embedding" not in payload:
                raise KeyError(f"Embedding pickle {embedding_path} is missing key 'embedding'")
            embedding = payload["embedding"]
        elif isinstance(payload, (np.ndarray, torch.Tensor)):
            embedding = payload
        else:
            raise TypeError(f"Unsupported embedding payload type in {embedding_path}: {type(payload)!r}")

        if isinstance(embedding, torch.Tensor):
            tensor = embedding.detach().cpu().to(dtype=torch.float32)
        else:
            tensor = torch.tensor(embedding, dtype=torch.float32)
        return tensor, int(self.labels[idx])

    def get_class_names(self) -> List[str]:
        """Return class names ordered by label index."""
        return [str(self.idx_to_label.get(idx, idx)) for idx in range(self.num_classes)]

    def get_class_distribution(self) -> Dict[str, int]:
        """Return counts per class name for loaded samples."""
        label_names = [str(self.idx_to_label[label]) for label in self.labels]
        return pd.Series(label_names).value_counts().to_dict()


class DINOv3EmbeddingDataModule(pl.LightningDataModule):
    """LightningDataModule wrapper for train/val cached embedding datasets."""

    def __init__(
        self,
        embeddings_dir: str | Path,
        train_csv: str | Path,
        val_csv: str | Path,
        batch_size: int = 256,
        num_workers: int = 4,
        photo_col: str = "Photo_ID",
        label_col: str = "Label",
        label_to_idx: Optional[Mapping[Any, int]] = None,
    ) -> None:
        super().__init__()
        self.embeddings_dir = embeddings_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.photo_col = photo_col
        self.label_col = label_col
        self.label_to_idx = dict(label_to_idx) if label_to_idx is not None else None
        self.train_dataset: Optional[DINOv3EmbeddingDataset] = None
        self.val_dataset: Optional[DINOv3EmbeddingDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.label_to_idx is None:
            self.label_to_idx = build_label_mapping(self.train_csv, label_col=self.label_col)
        self.train_dataset = DINOv3EmbeddingDataset(
            embeddings_dir=self.embeddings_dir,
            csv_path=self.train_csv,
            split="train",
            label_to_idx=self.label_to_idx,
            photo_col=self.photo_col,
            label_col=self.label_col,
        )
        self.val_dataset = DINOv3EmbeddingDataset(
            embeddings_dir=self.embeddings_dir,
            csv_path=self.val_csv,
            split="val",
            label_to_idx=self.label_to_idx,
            photo_col=self.photo_col,
            label_col=self.label_col,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )
