---
description: "How to use MegaDetector-Classifier — data preparation, splitting strategies, configuration, training, and using the trained weights."
tags:
  - MegaDetector-Classifier usage
  - camera-trap classification training
  - data splitting
  - config.yaml
  - species classification fine-tuning
  - sparrow-engine
---

# Usage Guide

This page walks through the full MegaDetector-Classifier workflow: preparing your data, configuring training, running the classifier, and using the output weights.

---

## 1. Data Structure

Images must be stored in a single **flat directory** (no subdirectories). An annotations CSV in the same directory maps each image to its class.

```plaintext
MegaDetector-Classifier/
├── data/
│   └── imgs/                         # dataset_root: all images stored here (flat)
│       ├── leopard_001.jpg
│       ├── ...
│       └── annotation_example.csv    # split_path: annotations file
└── configs/config.yaml
```

Keep the annotations CSV inside `dataset_root`: `main.py` writes `train_annotations.csv`, `val_annotations.csv` and `test_annotations.csv` next to `split_path` and reads them back from `dataset_root`, so the two directories must be the same.

### Annotation File Format

The CSV must contain exactly these three columns:

| Column | Type | Description | Example |
|---|---|---|---|
| `path` | string | Image file name inside `dataset_root` (crops are matched by file name) | `leopard_001.jpg` |
| `classification` | integer | Unique integer ID for each class | `0` |
| `label` | string | Human-readable class name | `leopard` |

---

## 2. Data Splitting

Set `split_data: True` in `config.yaml` to have MegaDetector-Classifier split your annotations into train, validation, and test sets automatically. The splitting strategy is controlled by `split_type`.

### Splitting Strategies

| Strategy | When to use | Extra column required |
|---|---|---|
| `random` | General-purpose balanced split | None |
| `location` | Keeps all images from one camera location in the same split | `Location` |
| `sequence` | Groups burst images within 30-second windows before splitting | `Photo_time` (YYYY-MM-DD HH:MM:SS) |

> **Important — camera-trap burst images:** Camera traps frequently capture bursts of images of the same animal within seconds. With random splitting, nearly identical frames can end up in both training and validation sets, causing artificially inflated validation accuracy (overfitting). Use `location` or `sequence` splitting to prevent this.

If you already have pre-split CSV files for train/val/test, set `split_data: False` and point `annotation_dir` to the directory containing those files.

---

## 3. Configuration

All training parameters live in `configs/config.yaml`. Edit this file before running `python main.py`.

### Training Parameters

| Parameter | Description | Default |
|---|---|---|
| `conf_id` | Unique identifier for this training run | `Crop_Res18_plain_071824` |
| `algorithm` | Training algorithm | `Plain` |
| `log_dir` | Directory for training logs | `Crop` |
| `num_epochs` | Total training epochs | `30` |
| `log_interval` | How often to log training info (in steps) | `10` |
| `parallel` | Set to `1` to enable multi-GPU training | `0` |

### Data Parameters

| Parameter | Description |
|---|---|
| `dataset_root` | Root directory where images are stored |
| `dataset_name` | Dataset type (`Custom_Crop` for fine-tuning) |
| `annotation_dir` | Directory containing annotation CSV files |
| `split_path` | Path to single CSV for auto-splitting |
| `test_size` | Proportion of data for test set (e.g. `0.2`) |
| `val_size` | Proportion of data for validation set (e.g. `0.2`) |
| `split_data` | `True` to auto-split, `False` if splits already exist |
| `split_type` | `random`, `location`, or `sequence` |
| `batch_size` | Images per batch (default: `32`) |
| `num_workers` | Dataloader worker processes (default: `4`) |

### Detection Parameters

Before training, `main.py` runs MegaDetector through sparrow-engine on every image listed in the train and validation CSVs and writes the crops to `<dataset_root>/cropped_resized/`.

| Parameter | Description | Default |
|---|---|---|
| `detector_model` | sparrow-engine model id used for detection. Downloaded on first use if missing. `MDV5a` works only if already installed locally, since the model zoo lists it as link-only. | `MDV6-yolov10-e` |
| `detection_threshold` | Minimum detection confidence for a box to be cropped | `0.2` |

Every detection above the threshold is cropped, including `person` and `vehicle` boxes, and each crop takes the label of its source image row. The crop file name starts with the MegaDetector class id (`0` animal, `1` person, `2` vehicle). The default `MDV6-yolov10-e` model is licensed AGPL-3.0 by its upstream; check that this fits your use, or set another detector. The `path` column of each CSV must hold image file names (for example `leopard_001.jpg`), because crops are matched to rows by file name.

### Model Parameters

| Parameter | Description |
|---|---|
| `num_classes` | Number of species classes in your dataset |
| `model_name` | Architecture (`PlainResNetClassifier`) |
| `num_layers` | ResNet depth — `18` or `50` |
| `weights_init` | Initial weights — `ImageNet` for transfer learning |

### Optimization Parameters

| Parameter | Description |
|---|---|
| `lr_feature` | Learning rate for the feature extractor |
| `momentum_feature` | Momentum for the feature extractor optimizer |
| `weight_decay_feature` | Weight decay for the feature extractor |
| `lr_classifier` | Learning rate for the classifier head |
| `momentum_classifier` | Momentum for the classifier head optimizer |
| `weight_decay_classifier` | Weight decay for the classifier head |
| `step_size` | LR scheduler step size (epochs) |
| `gamma` | LR scheduler decay factor |

> **Architecture note:** The current version supports only `PlainResNetClassifier` with ResNet-18 or ResNet-50 backbones. The classifier head and feature extractor are trained with separate optimizers, which matches the layout expected by the bundled `CustomWeights` loader.

---

## 4. Running Training

After configuring `configs/config.yaml`:

```bash
python main.py
```

Monitor the console output for per-epoch loss and accuracy. Logs are written to the directory specified in `log_dir`.

### Quick test with demo data

```bash
# Download demo data
wget https://zenodo.org/records/15376499/files/demo_data_clf.zip
unzip demo_data_clf.zip -d data/

# The demo config already points to ./data/imgs — just run:
python main.py
```

---

## 5. Output

Trained weights are saved to the `weights/` directory at the end of training. Trained weights load with the bundled `src.models.classification` loaders (for example `CustomWeights`). To run a trained classifier inside sparrow-engine, export it to ONNX and add a `manifest.toml`, as described in the sparrow-engine user manual (custom classifiers section).

---

## 6. Integration with MegaDetector

A common workflow pairs MegaDetector detection upstream with MegaDetector-Classifier downstream:

1. `main.py` runs [MegaDetector](https://github.com/microsoft/MegaDetector) through [sparrow-engine](https://github.com/microsoft/SPARROW-Engine) (`src/utils/batch_detection_cropping.py`) on the annotated images
2. Animal detections are cropped and resized into `<dataset_root>/cropped_resized/`
3. The classifier is trained on the crops
4. At deployment, run MegaDetector with sparrow-engine and classify each animal crop with the trained weights
