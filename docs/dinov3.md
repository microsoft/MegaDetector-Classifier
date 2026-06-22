# DINOv3 classification

## 1. Overview

DINOv3 classification is an experimental, optional feature for extracting DINOv3 embeddings, training a linear classifier head on cached embeddings, and running PyTorch-Wildlife-style inference. The base ResNet training and inference pipeline does not require the DINOv3 dependencies and continues to work without them installed.

DINOv3 backbone weights are hosted on Hugging Face behind Meta's DINOv3 License gate. The Amazon DINOv3 classifier heads are separate checkpoints and are not published yet; see the Zenodo placeholder in the inference section.

## 2. Installation

Install the optional DINOv3 dependency set only when you need DINOv3 extraction, training, or inference:

```bash
pip install -r requirements-dinov3.txt
```

The DINOv3 path requires:

- `transformers>=4.56.0`
- `huggingface_hub`
- `peft` only when loading PEFT/LoRA checkpoints

These dependencies are optional. You do not need them for the base MegaDetector-Classifier pipeline.

## 3. Accessing gated weights

The supported DINOv3 backbone repositories are manually gated under the Meta DINOv3 License. Complete these steps before first use:

1. Create a Hugging Face account.
2. Open each model page and submit the license gate request:
   - <https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m>
   - <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
   - <https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m>
3. Wait for manual approval.
4. Create a read token at <https://huggingface.co/settings/tokens>.
5. Authenticate one of these ways:

```bash
hf auth login
# or
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

## 4. Model sizes

All supported backbones use patch size 16 and image size 518.

| Model size | Hugging Face repo ID | Embedding dim | Patch size | Image size |
|---|---|---:|---:|---:|
| `small_plus` | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | 16 | 518 |
| `base` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 16 | 518 |
| `huge_plus` | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 | 16 | 518 |

## 5. Inference

Use `DINOv3AmazonRainforest` for Amazon Rainforest DINOv3 inference:

```python
from src.models.classification.dinov3_base import DINOv3AmazonRainforest

clf = DINOv3AmazonRainforest(
    model_size="base",
    weights="<local_checkpoint.pt>",
)

single = clf.single_image_classification(img)
batch = clf.batch_image_classification(data_path="<image_dir>")
batch_with_embeddings = clf.batch_image_classification_with_embeddings(
    data_path="<image_dir>",
    save_embeddings_dir="<embeddings_dir>",
)
```

Amazon DINOv3 weights are forthcoming on Zenodo (placeholder). Until then, pass a local checkpoint via `weights=<path>`. Once published, the manifest in `src/models/classification/dinov3_base/amazon.py` will hold the Zenodo URLs + SHA256 and `DINOv3AmazonRainforest(model_size=...)` will auto-download.

| Model size | Zenodo URL | SHA256 |
|---|---|---|
| `small_plus` | `<ZENODO_URL_TBD>` | `<SHA256_TBD>` |
| `base` | `<ZENODO_URL_TBD>` | `<SHA256_TBD>` |
| `huge_plus` | `<ZENODO_URL_TBD>` | `<SHA256_TBD>` |

## 6. Two-stage workflow

The two-stage path extracts frozen DINOv3 CLS embeddings first, then trains a lightweight classifier head from cached embeddings.

Stage 1: extract embeddings.

```bash
python scripts/extract_embeddings.py --model-size base --csv <annotations.csv> --root <image_root> --out <embeddings_dir> --split both
```

Stage 2: train the DINOv3 linear head.

```bash
python scripts/train_dinov3.py --config configs/dinov3_linear.yaml
```

Use `--model-size small_plus`, `--model-size base`, or `--model-size huge_plus` in stage 1 to choose the backbone size. Stage 2 uses the config file to locate cached embeddings and write the classifier checkpoint.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GatedRepoError` | The Hugging Face account has not been approved for the model, or the process is not authenticated. | Accept the license gate on each model page, wait for approval, then run `hf auth login`, `huggingface-cli login`, or set `HF_TOKEN`. |
| Offline or air-gapped run cannot resolve model files | The model is not available from the local Hugging Face cache. | Pre-populate the cache, set `HF_HUB_OFFLINE=1`, and use local-file loading where supported with `local_files_only=True`. |
| Cache is in the wrong location | Hugging Face defaults to `~/.cache/huggingface/hub`. | Set `HF_HOME=<cache_root>` to move the cache root. `HF_HUB_CACHE` can override the hub cache directory specifically. |
| Import fails for DINOv3 modules | Optional DINOv3 dependencies are not installed. | Run `pip install -r requirements-dinov3.txt` in the environment where you use DINOv3. |
