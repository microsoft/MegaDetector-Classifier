import pickle

import numpy as np
import pandas as pd
import torch

from src.datasets.dinov3_embedding import DINOv3EmbeddingDataset, safe_name


def test_embedding_dataset_roundtrip(tmp_path):
    csv_path = tmp_path / "annotations.csv"
    rows = [
        {"Photo_ID": "a/b/c.JPG", "Label": "cat"},
        {"Photo_ID": "d/e/f.JPG", "Label": "dog"},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    split_dir = tmp_path / "embeddings" / "train"
    split_dir.mkdir(parents=True)
    for idx, row in enumerate(rows):
        payload = {
            "embedding": np.asarray([idx, idx + 1, idx + 2], dtype=np.float32),
            "label": idx,
            "image_path": row["Photo_ID"],
        }
        with (split_dir / f"{safe_name(row['Photo_ID'])}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)

    assert safe_name("a/b/c.JPG") == "a_b_c.JPG"

    dataset = DINOv3EmbeddingDataset(
        embeddings_dir=tmp_path / "embeddings",
        csv_path=csv_path,
        split="train",
    )

    embedding, label = dataset[0]
    assert len(dataset) == 2
    assert isinstance(embedding, torch.Tensor)
    assert embedding.dtype == torch.float32
    assert embedding.shape == (3,)
    assert label == dataset.label_to_idx["cat"]
