from __future__ import annotations

import os
from pathlib import Path

DINOV3_MODEL_IDS: dict[str, str] = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge_plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}

DINOV3_DIMS: dict[str, int] = {
    "small": 384,
    "small_plus": 384,
    "base": 768,
    "large": 1024,
    "huge_plus": 1280,
}

__all__ = [
    "DINOV3_DIMS",
    "DINOV3_MODEL_IDS",
    "ensure_model_available",
    "resolve_model_id",
]


def resolve_model_id(size_or_id: str) -> str:
    if size_or_id in DINOV3_MODEL_IDS:
        return DINOV3_MODEL_IDS[size_or_id]
    if "/" in size_or_id:
        return size_or_id

    valid_sizes = ", ".join(DINOV3_MODEL_IDS)
    raise ValueError(
        f"Unknown DINOv3 model size {size_or_id!r}. "
        f"Valid sizes: {valid_sizes}; or pass a Hugging Face repo id."
    )


def _snapshot_has_required_files(snapshot_dir: Path) -> bool:
    has_preprocessor = (snapshot_dir / "preprocessor_config.json").is_file()
    has_weights = (
        (snapshot_dir / "model.safetensors").is_file()
        or (snapshot_dir / "model.safetensors.index.json").is_file()
        or any(snapshot_dir.glob("model-*.safetensors"))
    )
    return has_preprocessor and has_weights


def ensure_model_available(
    model_id: str,
    token=True,
    local_files_only: bool = False,
) -> str:
    resolved_model_id = resolve_model_id(model_id)
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "ensure_model_available requires huggingface_hub. "
            "Install with: pip install -r requirements-dinov3.txt"
        ) from exc

    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
        hub_errors = (
            GatedRepoError,
            RepositoryNotFoundError,
            LocalEntryNotFoundError,
            OSError,
        )
    except (AttributeError, ImportError):
        GatedRepoError = RepositoryNotFoundError = LocalEntryNotFoundError = None
        hub_errors = (OSError,)

    try:
        cached_config = huggingface_hub.try_to_load_from_cache(resolved_model_id, "config.json")
        if local_files_only and isinstance(cached_config, (str, os.PathLike)):
            snapshot_dir = Path(cached_config).parent
            if _snapshot_has_required_files(snapshot_dir):
                return str(snapshot_dir)
            raise RuntimeError(
                f"{resolved_model_id} is incomplete in the local Hugging Face cache "
                f"at {snapshot_dir}. Found config.json but missing "
                "preprocessor_config.json or model safetensors. Disable "
                "local_files_only/offline mode or pre-download the complete model."
            )

        snapshot_path = huggingface_hub.snapshot_download(
            resolved_model_id,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ],
            token=token,
            local_files_only=local_files_only,
        )
        return str(Path(snapshot_path))
    except hub_errors as exc:
        if GatedRepoError is not None and isinstance(exc, GatedRepoError):
            raise RuntimeError(
                f"{resolved_model_id} is gated. Visit https://huggingface.co/{resolved_model_id}, "
                "accept the DINOv3 license, then run `hf auth login` or set HF_TOKEN."
            ) from exc
        if RepositoryNotFoundError is not None and isinstance(
            exc, RepositoryNotFoundError
        ):
            raise RuntimeError(f"Unknown DINOv3 repo id: {resolved_model_id}") from exc
        if LocalEntryNotFoundError is not None and isinstance(
            exc, LocalEntryNotFoundError
        ):
            raise RuntimeError(
                f"{resolved_model_id} is not available in the local Hugging Face cache. "
                "Disable local_files_only/offline mode or pre-download the model after "
                "accepting the DINOv3 license."
            ) from exc
        raise RuntimeError(f"Failed to access DINOv3 model {resolved_model_id}: {exc}") from exc
