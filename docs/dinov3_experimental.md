# DINOv3 fine-tuning notes (experimental)

> Status: EXPERIMENTAL. Supplementary notes for the `main.py`-routable
> `DINOv3Linear` linear-probe path. See [`docs/dinov3.md`](dinov3.md) for the
> canonical DINOv3 install, gated-access, inference, and two-stage workflow guide.

## Install

```bash
pip install -r requirements-dinov3.txt
```

This installs `transformers`, `peft`, and `huggingface_hub` (pinned ranges).

## HuggingFace authentication

DINOv3 weights live on the HuggingFace Hub and may be gated. Before first
use, authenticate:

```bash
huggingface-cli login
```

If your environment is offline, set:

```bash
export HF_HOME=/path/to/local/hf-cache
export TRANSFORMERS_OFFLINE=1
```

## GPU memory

ViT-S (recommended for first attempts) fits on ~6 GB of VRAM at
`batch_size=32, image_size=518`. Larger ViT backbones (e.g. ViT-B, ViT-L,
ViT-H+) need proportionally more VRAM, so reduce `batch_size` accordingly.
These size names describe the `main.py`-routable `DINOv3Linear` path, which
accepts any DINOv3 Hugging Face repo id; the two-stage scripts
(`scripts/extract_embeddings.py`) accept only `small_plus`, `base`, and
`huge_plus` — see [`docs/dinov3.md`](dinov3.md).

## Image size

DINOv3 backbones expect 518×518 input. Use `Custom_Crop(conf, image_size=518)`
or set `image_size: 518` in your YAML config.

## Scope

Implemented and supported today:

- `DINOv3Classifier` (frozen backbone) and `DINOv3Linear` (linear-probe
  training, routable via `main.py`).
- DINOv3 inference surface: `DINOv3AmazonRainforest`, `DINOv3BaseInference`, and
  `batch_image_classification_with_embeddings`.
- SINCHI checkpoint loaders — flavours A–D in `_extract_state_dict`
  (`src/models/classification/dinov3_base/base_classifier.py`).
- `DINOv3CustomWeights` to load a local DINOv3 checkpoint (`.pt`/`.ckpt`). Load
  checkpoints only from trusted sources; loading uses
  `torch.load(weights_only=False)`.
- Two-stage workflow: `scripts/extract_embeddings.py` + `scripts/train_dinov3.py`
  driven by `configs/dinov3_linear.yaml`.
- Lazy optional-deps surface: `from src.models import classification` works
  without DINOv3 deps installed; `classification.DINOv3CustomWeights` raises an
  actionable `ImportError` if deps are absent.

Still deferred (raise `NotImplementedError`):

- Full-backbone fine-tuning — `DINOv3Classifier(freeze_backbone=False)`.
- PEFT/LoRA training — `DINOv3Linear._wrap_with_peft`.

See [`docs/dinov3.md`](dinov3.md) for the supported workflow.

## License

DINOv3 weights are subject to Meta's licence — see the model card on
HuggingFace Hub. Do not redistribute weights from this repo.
