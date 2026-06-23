"""Amazon rainforest DINOv3 classifier placeholder manifest."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname, urlopen

from .base_classifier import DINOv3BaseInference
from .custom_weights import DINOv3CustomWeights

__all__ = ["DINOv3AmazonRainforest", "DINOV3_AMAZON_MANIFEST"]


_AMAZON_CLASS_NAMES = [
    "Dasyprocta",
    "Bos",
    "Pecari",
    "Mazama",
    "Cuniculus",
    "Leptotila",
    "Human",
    "Aramides",
    "Tinamus",
    "Eira",
    "Crax",
    "Procyon",
    "Capra",
    "Dasypus",
    "Sciurus",
    "Crypturellus",
    "Tamandua",
    "Proechimys",
    "Leopardus",
    "Equus",
    "Columbina",
    "Nyctidromus",
    "Ortalis",
    "Emballonura",
    "Odontophorus",
    "Geotrygon",
    "Metachirus",
    "Catharus",
    "Cerdocyon",
    "Momotus",
    "Tapirus",
    "Canis",
    "Furnarius",
    "Didelphis",
    "Sylvilagus",
    "Unknown",
]

# TODO(Zenodo): publish Amazon DINOv3 classifier-only heads and fill head_url + sha256 per size.
DINOV3_AMAZON_MANIFEST = {
    "small_plus": {
        "model_name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "embedding_dim": 384,
        "num_classes": len(_AMAZON_CLASS_NAMES),
        "class_names": _AMAZON_CLASS_NAMES,
        "head_url": None,
        "sha256": None,
        "flavor": "classifier_only",
        "revision": None,
    },
    "base": {
        "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "embedding_dim": 768,
        "num_classes": len(_AMAZON_CLASS_NAMES),
        "class_names": _AMAZON_CLASS_NAMES,
        "head_url": None,
        "sha256": None,
        "flavor": "classifier_only",
        "revision": None,
    },
    "huge_plus": {
        "model_name": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "embedding_dim": 1280,
        "num_classes": len(_AMAZON_CLASS_NAMES),
        "class_names": _AMAZON_CLASS_NAMES,
        "head_url": None,
        "sha256": None,
        "flavor": "classifier_only",
        "revision": None,
    },
}

_REQUIRED_MANIFEST_FIELDS = {
    "model_name",
    "embedding_dim",
    "num_classes",
    "class_names",
    "head_url",
    "sha256",
    "flavor",
    "revision",
}


def _validate_manifest(model_size: str, manifest: dict, *, require_published: bool) -> None:
    missing = sorted(_REQUIRED_MANIFEST_FIELDS.difference(manifest))
    if missing:
        raise ValueError(
            f"Amazon DINOv3 manifest entry {model_size!r} is missing field(s): "
            f"{', '.join(missing)}."
        )
    if manifest["flavor"] != "classifier_only":
        raise ValueError(
            f"Amazon DINOv3 manifest entry {model_size!r} has unsupported "
            f"flavor {manifest['flavor']!r}; expected 'classifier_only'."
        )
    class_names = list(manifest["class_names"])
    if len(class_names) != int(manifest["num_classes"]):
        raise ValueError(
            f"Amazon DINOv3 manifest entry {model_size!r} class_names length "
            f"{len(class_names)} does not match num_classes={manifest['num_classes']}."
        )
    if require_published and (manifest["head_url"] is None or manifest["sha256"] is None):
        raise NotImplementedError(
            f"Amazon DINOv3 weights for size '{model_size}' are not yet "
            "published to Zenodo. Pass weights=<local .pt path> to load "
            "a local checkpoint. (Auto-download URL is a placeholder "
            "pending Zenodo upload.)"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    actual = _sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch for Amazon DINOv3 head {path}: "
            f"expected {expected_sha256}, got {actual}."
        )


def _cache_dir() -> Path:
    return Path(
        os.environ.get(
            "DINOV3_AMAZON_CACHE_DIR",
            Path.home() / ".cache" / "megadetector-classifier" / "dinov3",
        )
    ).expanduser()


def _resolve_head_source(source: str, model_size: str, expected_sha256: str | None) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(parsed.path).name or f"amazon_dinov3_{model_size}_head.pt"
        destination = cache_dir / filename
        if not destination.exists():
            with urlopen(source) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        _verify_sha256(destination, expected_sha256)
        return destination
    if parsed.scheme == "file":
        path = Path(url2pathname(parsed.path)).expanduser()
    else:
        path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Amazon DINOv3 head not found: {path}")
    _verify_sha256(path, expected_sha256)
    return path


class DINOv3AmazonRainforest(DINOv3BaseInference):
    """Amazon rainforest DINOv3 classifier."""

    IMAGE_SIZE = 518
    CLASS_NAMES = dict(enumerate(_AMAZON_CLASS_NAMES))

    def __init__(self, model_size="base", weights=None, device="cpu"):
        if model_size not in DINOV3_AMAZON_MANIFEST:
            raise ValueError(
                f"Unknown Amazon DINOv3 model_size {model_size!r}; "
                f"expected one of {sorted(DINOV3_AMAZON_MANIFEST)}."
            )
        manifest = DINOV3_AMAZON_MANIFEST[model_size]
        _validate_manifest(model_size, manifest, require_published=weights is None)
        if weights is None:
            weights = self._download_and_verify_head(model_size)

        from .transforms import DINOv3Transform

        class_names = list(manifest["class_names"])
        loaded = DINOv3CustomWeights(
            weights=weights,
            class_names=class_names,
            device=device,
            model_size=model_size,
            model_name=manifest["model_name"],
            embedding_dim=manifest["embedding_dim"],
            expected_num_classes=manifest["num_classes"],
            expected_image_size=self.IMAGE_SIZE,
            expected_feature_type="cls",
            expected_transform="dinov3_bicubic_518_imagenet",
            expected_transform_version=1,
            expected_class_names=class_names,
        )
        super().__init__(
            net=loaded.net,
            transform=DINOv3Transform(self.IMAGE_SIZE),
            device=device,
            class_names=class_names,
        )
        self.num_cls = loaded.num_cls
        self.checkpoint_meta = loaded.checkpoint_meta
        self.eval()
        self.to(device)

    @staticmethod
    def _download_and_verify_head(model_size: str):
        manifest = DINOV3_AMAZON_MANIFEST[model_size]
        _validate_manifest(model_size, manifest, require_published=True)
        return _resolve_head_source(
            manifest["head_url"],
            model_size,
            manifest["sha256"],
        )
