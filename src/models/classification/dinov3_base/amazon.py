"""Amazon rainforest DINOv3 classifier placeholder manifest."""
from __future__ import annotations

from .base_classifier import DINOv3BaseInference
from .custom_weights import DINOv3CustomWeights

__all__ = ["DINOv3AmazonRainforest", "DINOV3_AMAZON_MANIFEST"]


# TODO(Zenodo): publish Amazon DINOv3 classifier-only heads and fill head_url + sha256 per size.
DINOV3_AMAZON_MANIFEST = {
    "small_plus": {
        "model_name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "embedding_dim": 384,
        "head_url": None,
        "sha256": None,
        "flavor": "classifier_only",
    },
    "base": {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "embedding_dim": 768,
        "head_url": None,
        "sha256": None,
        "flavor": "classifier_only",
    },
    "huge_plus": {
        "model_name": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "embedding_dim": 1280,
        "head_url": None,
        "sha256": None,
        "flavor": "classifier_only",
    },
}


class DINOv3AmazonRainforest(DINOv3BaseInference):
    """Amazon rainforest DINOv3 classifier."""

    IMAGE_SIZE = 518
    CLASS_NAMES = {
        0: "Dasyprocta",
        1: "Bos",
        2: "Pecari",
        3: "Mazama",
        4: "Cuniculus",
        5: "Leptotila",
        6: "Human",
        7: "Aramides",
        8: "Tinamus",
        9: "Eira",
        10: "Crax",
        11: "Procyon",
        12: "Capra",
        13: "Dasypus",
        14: "Sciurus",
        15: "Crypturellus",
        16: "Tamandua",
        17: "Proechimys",
        18: "Leopardus",
        19: "Equus",
        20: "Columbina",
        21: "Nyctidromus",
        22: "Ortalis",
        23: "Emballonura",
        24: "Odontophorus",
        25: "Geotrygon",
        26: "Metachirus",
        27: "Catharus",
        28: "Cerdocyon",
        29: "Momotus",
        30: "Tapirus",
        31: "Canis",
        32: "Furnarius",
        33: "Didelphis",
        34: "Sylvilagus",
        35: "Unknown",
    }

    def __init__(self, model_size="base", weights=None, device="cpu"):
        if model_size not in DINOV3_AMAZON_MANIFEST:
            raise ValueError(
                f"Unknown Amazon DINOv3 model_size {model_size!r}; "
                f"expected one of {sorted(DINOV3_AMAZON_MANIFEST)}."
            )
        manifest = DINOV3_AMAZON_MANIFEST[model_size]
        if weights is None:
            if manifest["head_url"] is None:
                raise NotImplementedError(
                    f"Amazon DINOv3 weights for size '{model_size}' are not yet "
                    "published to Zenodo. Pass weights=<local .pt path> to load "
                    "a local checkpoint. (Auto-download URL is a placeholder "
                    "pending Zenodo upload.)"
                )
            weights = self._download_and_verify_head(model_size)

        from .transforms import DINOv3Transform

        loaded = DINOv3CustomWeights(
            weights=weights,
            class_names=list(self.CLASS_NAMES.values()),
            device=device,
            model_size=model_size,
            model_name=manifest["model_name"],
            embedding_dim=manifest["embedding_dim"],
        )
        super().__init__(
            net=loaded.net,
            transform=DINOv3Transform(self.IMAGE_SIZE),
            device=device,
            class_names=list(self.CLASS_NAMES.values()),
        )
        self.num_cls = loaded.num_cls
        self.checkpoint_meta = loaded.checkpoint_meta
        self.eval()
        self.to(device)

    @staticmethod
    def _download_and_verify_head(model_size: str):
        raise NotImplementedError(
            "Amazon DINOv3 auto-download is pending Zenodo head_url + sha256 "
            f"publication for size {model_size!r}."
        )
