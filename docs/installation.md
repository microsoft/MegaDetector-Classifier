---
description: "How to install and set up MegaDetector-Classifier for camera-trap species classification fine-tuning."
tags:
  - MegaDetector-Classifier installation
  - camera-trap classification setup
  - sparrow-engine
  - fine-tuning setup
  - wildlife monitoring
---

# Installation

## Requirements

- Python 3.11+ (required by sparrow-engine)
- PyTorch 2.0+
- CUDA (optional, but recommended for faster training)

## Install with pip

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
pip install -r requirements.txt
```

This installs the following dependencies:

| Package | Purpose |
|---|---|
| `sparrow-engine` | Runs MegaDetector to crop animals before training (CPU build) |
| `supervision` | Detection containers and crop writing |
| `timm` | Backbones for the bundled pretrained classifier loaders |
| `lightning` | Training loop and checkpointing (PyTorch Lightning) |
| `scikit_learn` | Data splitting utilities |
| `munch` | YAML config as dot-accessible object |
| `typer` | CLI argument handling |

## Install with conda

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
conda env create -f environment.yaml
conda activate PT_Finetuning
```

## Verify

```python
import sparrow_engine
from src.models import classification
print(sparrow_engine.__version__, sparrow_engine.active_device())
```

Run this from the repository root.

## GPU Setup

Training on GPU is recommended for faster iteration. Verify CUDA availability:

```python
import torch
print(torch.cuda.is_available())  # should print True on a CUDA-enabled machine
```

CPU training is supported but will be slower for larger datasets.

The `sparrow-engine` package in `requirements.txt` is the CPU build, so MegaDetector runs on the CPU. For GPU detection, replace it with the GPU build. Install only one of the two:

```bash
pip uninstall -y sparrow-engine
pip install sparrow-engine-gpu
```

## Next Steps

- Download [demo data](https://zenodo.org/records/15376499/files/demo_data_clf.zip) for an immediate end-to-end test
- Read the [Usage Guide](usage.md) for a full walkthrough of data preparation, configuration, and training
