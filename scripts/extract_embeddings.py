"""Stage-1 CLI for extracting DINOv3 CLS embeddings from image CSVs."""
from __future__ import annotations

import argparse
import pickle
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

IMAGE_SIZE = 518
DEFAULT_PHOTO_COL = "Photo_ID"
DEFAULT_LABEL_COL = "Label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cached DINOv3 embeddings from image CSV rows")
    parser.add_argument("--model-size", choices=["small_plus", "base", "huge_plus"], required=True)
    parser.add_argument("--csv", required=True, help="CSV containing image paths in Photo_ID")
    parser.add_argument("--root", required=True, help="Root directory for paths from the CSV")
    parser.add_argument("--out", required=True, help="Output directory for cached embeddings")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _selected_splits(split: str) -> List[str]:
    return ["train", "val"] if split == "both" else [split]


def _split_column(df: pd.DataFrame) -> Optional[str]:
    for column in ("split", "Split", "dataset", "Dataset", "subset", "Subset"):
        if column in df.columns:
            return column
    return None


def _label_mapping(df: pd.DataFrame, label_col: str) -> Dict[object, int]:
    if label_col not in df.columns:
        return {}
    labels = sorted(df[label_col].dropna().unique().tolist(), key=lambda value: str(value))
    return {label: idx for idx, label in enumerate(labels)}


def _label_index(row: pd.Series, label_col: str, label_to_idx: Dict[object, int]) -> int:
    if label_col not in row or pd.isna(row[label_col]):
        return -1
    label = row[label_col]
    if label in label_to_idx:
        return int(label_to_idx[label])
    if str(label) in label_to_idx:
        return int(label_to_idx[str(label)])
    return -1


def _verify_existing_embedding(embedding_path: Path, image_path: str) -> None:
    with embedding_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "image_path" not in payload:
        raise ValueError(
            f"Cannot resume-skip {embedding_path}: existing cache lacks image_path metadata"
        )
    cached_image_path = str(payload["image_path"])
    if cached_image_path != image_path:
        raise ValueError(
            "Embedding cache collision: "
            f"{embedding_path} already stores image_path={cached_image_path!r}, "
            f"current row has image_path={image_path!r}"
        )


def _rows_for_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    column = _split_column(df)
    if column is None:
        return df.reset_index(drop=True)
    return df[df[column].astype(str).str.lower() == split].reset_index(drop=True)


def _write_timing(output_dir: Path, model_name: str, timings: Dict[str, float], total_time: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_file = output_dir / "extraction_timing.txt"
    with timing_file.open("w") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("DINOv3 EMBEDDING EXTRACTION TIME SUMMARY\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Output directory: {output_dir}\n")
        for split, elapsed in timings.items():
            handle.write(f"{split} extraction time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)\n")
        handle.write(f"Total extraction time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)\n")
        handle.write("=" * 80 + "\n")


def _resolve_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset

    from src.datasets.dinov3_embedding import safe_name
    from src.models import DINOv3Classifier
    from src.models.classification.dinov3_base import DINOv3Transform
    from src.models.dinov3_registry import DINOV3_DIMS, ensure_model_available, resolve_model_id

    class ImageCSVDataset(Dataset):
        def __init__(
            self,
            frame: pd.DataFrame,
            root: Path,
            transform,
            label_to_idx: Dict[object, int],
            photo_col: str = DEFAULT_PHOTO_COL,
            label_col: str = DEFAULT_LABEL_COL,
        ) -> None:
            if photo_col not in frame.columns:
                raise KeyError(f"CSV is missing required photo column {photo_col!r}")
            self.frame = frame.reset_index(drop=True)
            self.root = root
            self.transform = transform
            self.label_to_idx = label_to_idx
            self.photo_col = photo_col
            self.label_col = label_col

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            image_path = str(row[self.photo_col])
            full_path = self.root / image_path
            with full_path.open("rb") as handle:
                image = Image.open(handle).convert("RGB")
                tensor = self.transform(image)
            label = _label_index(row, self.label_col, self.label_to_idx)
            return tensor, int(label), image_path

    csv_path = Path(args.csv)
    root_dir = Path(args.root)
    output_dir = Path(args.out)
    df = pd.read_csv(csv_path)
    label_to_idx = _label_mapping(df, DEFAULT_LABEL_COL)

    model_name = resolve_model_id(args.model_size)
    ensure_model_available(model_name)
    device = _resolve_device(torch, args.device)

    model = DINOv3Classifier(
        model_name=model_name,
        num_classes=1,
        embedding_dim=DINOV3_DIMS[args.model_size],
        freeze_backbone=True,
    ).to(device)
    model.eval()
    transform = DINOv3Transform(IMAGE_SIZE)

    timings: Dict[str, float] = {}
    extraction_start = time.time()

    for split in _selected_splits(args.split):
        split_df = _rows_for_split(df, split)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        dataset = ImageCSVDataset(split_df, root_dir, transform, label_to_idx)
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        stats = {"total": 0, "saved": 0, "skipped": 0}
        seen_targets: Dict[Path, str] = {}
        manifest_rows: List[Dict[str, object]] = []
        split_start = time.time()
        with torch.inference_mode():
            for images, labels, image_paths in loader:
                images = images.to(device, non_blocking=True)
                embeddings = model.get_embeddings(images)
                for idx, image_path in enumerate(image_paths):
                    stats["total"] += 1
                    image_path = str(image_path)
                    embedding_path = split_dir / f"{safe_name(image_path)}.pkl"
                    previous_image_path = seen_targets.get(embedding_path)
                    if previous_image_path is not None and previous_image_path != image_path:
                        raise ValueError(
                            "Embedding cache collision within split "
                            f"{split!r}: {previous_image_path!r} and {image_path!r} "
                            f"both map to {embedding_path}"
                        )
                    seen_targets[embedding_path] = image_path
                    label = int(labels[idx].item() if hasattr(labels[idx], "item") else labels[idx])
                    if embedding_path.exists():
                        _verify_existing_embedding(embedding_path, image_path)
                        manifest_rows.append(
                            {"image_path": image_path, "pkl_path": str(embedding_path), "label": label}
                        )
                        stats["skipped"] += 1
                        continue
                    embedding = embeddings[idx].detach().cpu().to(dtype=torch.float32).numpy()
                    payload = {
                        "embedding": embedding,
                        "label": label,
                        "image_path": image_path,
                    }
                    with embedding_path.open("wb") as handle:
                        pickle.dump(payload, handle)
                    manifest_rows.append(
                        {"image_path": image_path, "pkl_path": str(embedding_path), "label": label}
                    )
                    stats["saved"] += 1

        pd.DataFrame(manifest_rows, columns=["image_path", "pkl_path", "label"]).to_csv(
            split_dir / "manifest.csv",
            index=False,
        )
        timings[split] = time.time() - split_start
        print(
            f"{split}: total={stats['total']} saved={stats['saved']} skipped={stats['skipped']} "
            f"time={str(timedelta(seconds=int(timings[split])))}"
        )

    total_time = time.time() - extraction_start
    _write_timing(output_dir, model_name, timings, total_time)
    print(f"Embedding extraction complete: {output_dir}")


if __name__ == "__main__":
    main()
