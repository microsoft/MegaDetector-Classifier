# DINOv3 fine-tuning (experimental — Phase 5 MVP)

> Status: EXPERIMENTAL. Frozen-backbone linear-probe only.
> Full fine-tune lands in Phase 6; PEFT/LoRA in Phase 7; SINCHI ckpts in Phase 8.
> Full docs land in Phase 9.

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
`batch_size=32, image_size=518`. Larger backbones (ViT-B, ViT-L) need
proportionally more memory. Full memory table lands in `docs/dinov3.md` (P9).

## Image size

DINOv3 backbones expect 518×518 input. Use `Custom_Crop(image_size=518)`
or set `image_size: 518` in your YAML config.

## Scope (P5 MVP)

This phase ships:

- `DINOv3Classifier` (frozen backbone) and `DINOv3Linear` (linear-probe training).
- `DINOv3CustomWeights` to reload a `.ckpt` trained by this repo. Load
  `.ckpt` files only from trusted sources; v1 uses `torch.load(weights_only=False)`.
- Lazy optional-deps surface: `from src.models import classification` works without
  DINOv3 deps installed; `classification.DINOv3CustomWeights` raises an
  actionable `ImportError` if deps are absent.

Not included (later phases): full fine-tune (P6), PEFT/LoRA (P7),
SINCHI checkpoint flavours (P8), sample configs & full docs (P9).

## License

DINOv3 weights are subject to Meta's licence — see the model card on
HuggingFace Hub. Do not redistribute weights from this repo.
