import pytest

from src.models.classification.dinov3_base.amazon import (
    DINOV3_AMAZON_MANIFEST,
    DINOv3AmazonRainforest,
)


def test_dinov3_amazon_placeholder_raises_without_local_weights():
    with pytest.raises(NotImplementedError, match="Zenodo"):
        DINOv3AmazonRainforest(model_size="base")


def test_dinov3_amazon_class_names_and_manifest():
    assert len(DINOv3AmazonRainforest.CLASS_NAMES) == 36
    assert DINOv3AmazonRainforest.CLASS_NAMES[0] == "Dasyprocta"
    assert DINOv3AmazonRainforest.CLASS_NAMES[35] == "Unknown"

    assert set(DINOV3_AMAZON_MANIFEST) == {"small_plus", "base", "huge_plus"}
    assert DINOV3_AMAZON_MANIFEST["small_plus"]["embedding_dim"] == 384
    assert DINOV3_AMAZON_MANIFEST["base"]["embedding_dim"] == 768
    assert DINOV3_AMAZON_MANIFEST["huge_plus"]["embedding_dim"] == 1280
    assert all(item["head_url"] is None for item in DINOV3_AMAZON_MANIFEST.values())
    assert all(item["sha256"] is None for item in DINOV3_AMAZON_MANIFEST.values())
    assert all(
        item["flavor"] == "classifier_only"
        for item in DINOV3_AMAZON_MANIFEST.values()
    )
