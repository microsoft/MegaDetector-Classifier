<a name="source-pipeline"></a>
## Sub-area: source-pipeline

## **Key Findings**

1. **[NEW] Source workflow is two-stage: extract embeddings first, then train on cached embeddings.** `EXPERIMENT_FEATURES.md` documents `python scripts/extract_embeddings.py --config configs/config.yaml --output_dir ./embeddings`, then `experiment.use_cached_embeddings: true`, then `python scripts/train.py --config configs/config.yaml` (`EXPERIMENT_FEATURES.md:13-28`). `SWEEP_README.md` makes cached embeddings a prerequisite for mixup sweeps (`SWEEP_README.md:17-27`).

2. **[NEW] `scripts/extract_embeddings.py` CLI is minimal and config-driven.** Args are `--config` default `configs/config.yaml`, `--output_dir` default `./embeddings`, and `--split {train,val,both}` default `both` (`scripts/extract_embeddings.py:127-149`). Relative config and output paths are resolved under the source repo root (`scripts/extract_embeddings.py:151-170`).

3. **[NEW] Embedding extraction input format is image-root plus annotation CSVs from config.** The script reads `config["data"]["train_csv"]`, `val_csv`, and `root_dir` for train/val datasets (`scripts/extract_embeddings.py:193-198`, `scripts/extract_embeddings.py:227-232`). The dataset expects CSV columns `Label` and `Photo_ID` by default (`data/dataset.py:37-57`), and README describes `Photo_ID` as a path relative to `data.root_dir` and `Label` as the class name (`README.md:135-149`).

4. **[NEW] Embedding extraction uses deterministic validation preprocessing at 518 by default.** It reads `image_size = int(config["data"]["image_size"])` and calls `get_default_transforms(..., is_training=False)` (`scripts/extract_embeddings.py:181-183`). Validation transforms are bicubic resize to `(image_size, image_size)`, tensor conversion, and ImageNet normalization (`data/dataset.py:201-210`); the default helper and config both use 518 (`data/dataset.py:174-184`, `configs/config.yaml:28-35`).

5. **[NEW] Embeddings are saved as one pickle file per image, not as batched `.pt/.npy/.npz` files.** The save loop iterates each embedding in the batch, creates a safe file name by replacing `/` and `\\` with `_`, skips existing files, and pickles `{"embedding": embedding.numpy(), "label": labels[i].item(), "image_path": img_path}` (`scripts/extract_embeddings.py:32-44`, `scripts/extract_embeddings.py:92-123`). The docs repeat this schema and filename mapping (`EXPERIMENT_FEATURES.md:133-145`).

6. **[NEW] Extracted embeddings are laid out under split directories and optionally model-specific directories.** `extract_embeddings.py` writes train embeddings to `<output_dir>/train` and validation embeddings to `<output_dir>/val` (`scripts/extract_embeddings.py:200-208`, `scripts/extract_embeddings.py:234-242`). `extract_embeddings_all_models.py` derives `<base_dir>/<model-name-with-dashes-replaced>` (`scripts/extract_embeddings_all_models.py:22-35`) and invokes `extract_embeddings.py --output_dir <that_dir>` for each model (`scripts/extract_embeddings_all_models.py:136-147`).

7. **[NEW] `extract_embeddings_all_models.py` mutates the YAML config on disk and covers B/L/7B, not the requested S+/B/H+ set.** It rewrites `config["model"]["name"]`, `config["experiment"]["embeddings_dir"]`, and `config["model"]["embedding_dim"] = None` (`scripts/extract_embeddings_all_models.py:38-57`). Its built-in `MODELS` list is `vitb16`, `vitl16`, and `vit7b16` (`scripts/extract_embeddings_all_models.py:13-19`), while the current config uses `vith16plus` (`configs/config.yaml:10-15`).

8. **[NEW] Model loading is via Hugging Face `AutoModel` with eager attention and explicit login/access failure text.** `DINOv3LinearClassifier` calls `AutoModel.from_pretrained(model_name, attn_implementation="eager")` and `AutoImageProcessor.from_pretrained(model_name)` (`models/dinov3_model.py:44-53`). Load failure raises a `RuntimeError` instructing `huggingface-cli login` and model access request at the HF URL (`models/dinov3_model.py:54-60`).

9. **[NEW] The wrapper obtains embeddings from the CLS token.** Both `forward()` and `get_embeddings()` use `outputs.last_hidden_state[:, 0, :]` when available, with tensor and pooler fallbacks (`models/dinov3_model.py:120-146`, `models/dinov3_model.py:148-173`).

10. **[NEW] `DINOv3LinearClassifier` is a backbone plus single linear head.** The constructor stores `self.backbone`, optionally freezes it, creates `self.classifier = nn.Linear(embedding_dim, num_classes)`, and initializes classifier weights/biases (`models/dinov3_model.py:84-95`). `models/__init__.py` exports only this class (`models/__init__.py:1-5`).

11. **[NEW] Backbone freezing sets all backbone params non-trainable and forces eval mode.** `_freeze_backbone()` loops over `self.backbone.parameters()` setting `requires_grad=False`, then calls `self.backbone.eval()` (`models/dinov3_model.py:97-102`). `unfreeze_backbone()` reverses this and calls `.train()` (`models/dinov3_model.py:104-109`).

12. **[NEW] Embedding dimension is auto-detected from HF config, with incomplete fallback mappings.** If `embedding_dim` is `None`, the wrapper uses `backbone.config.hidden_size`, then `backbone.embed_dim`, then fallback substrings for `vit7b`, `vitb`/`vit-base`, and `vitl`/`vit-large` (`models/dinov3_model.py:62-83`). It has no fallback branch for `vits16plus` or `vith16plus`; those rely on `hidden_size` existing.

13. **[NEW] The image dataset hardcodes the official Amazon 36-class mapping and filters unknown labels.** `AMAZON_RAINFOREST_CLASS_NAMES` contains 36 names including `Unknown`; the mapping is built from that list (`data/dataset.py:17-29`). The dataset filters rows whose `Label` is not in this list, raises if none remain, and always uses the official mapping (`data/dataset.py:72-108`).

14. **[NEW] Dataset image loading returns `(image_tensor, label_idx, image_path)` and verifies missing images at init.** `_verify_images()` removes missing paths (`data/dataset.py:113-130`), `__getitem__` opens `root_dir / img_path`, converts RGB, applies transforms, and returns image, label, and the original relative path string (`data/dataset.py:135-160`).

15. **[NEW] Cached embedding consumption is implemented by `EmbeddingDataset` with the same CSV and filename scheme.** It loads the same `Label`/`Photo_ID` columns, filters to the official classes, uses safe filenames ending `.pkl`, skips rows whose embedding file is missing, and raises if no cached embeddings are found (`data/embedding_dataset.py:35-53`, `data/embedding_dataset.py:59-87`, `data/embedding_dataset.py:88-145`).

16. **[NEW] `EmbeddingDataset.__getitem__` accepts dict/NumPy/tensor payloads but returns only embedding, label, image path.** It unpickles each file, uses the `embedding` key when the payload is a dict, accepts raw `np.ndarray` or `torch.Tensor`, converts to `torch.float`, and returns `(embedding, label, img_path)` (`data/embedding_dataset.py:151-186`). It does not validate that the pickle's stored `label` matches the CSV label.

17. **[NEW] Training data loaders branch on `experiment.use_cached_embeddings`.** Cached mode resolves `experiment.embeddings_dir`, uses `<dir>/train` and `<dir>/val`, and builds `EmbeddingDataset`; non-cached mode builds image datasets with train/val transforms (`scripts/train.py:123-225`).

18. **[NEW] Training and validation skip the DINO backbone in cached mode.** In `train_epoch`, cached data is moved to the device as embeddings and logits are `model.classifier(embeddings)`; non-cached mode calls `model(images)` (`scripts/train.py:330-373`). Validation mirrors the same branch (`scripts/train.py:453-480`).

19. **[NEW] Optimizer/scheduler choices are AdamW or SGD and cosine or step.** When the backbone is frozen, only `model.classifier.parameters()` are optimized; otherwise all model parameters are used (`scripts/train.py:258-267`). Optimizers are AdamW or SGD with momentum 0.9 (`scripts/train.py:272-287`); schedulers are `CosineAnnealingLR` or `StepLR` (`scripts/train.py:289-304`).

20. **[NEW] Loss is cross entropy with label smoothing from config.** `train.py` supports only `loss.name == "cross_entropy"` and constructs `nn.CrossEntropyLoss(label_smoothing=...)` (`scripts/train.py:814-819`). `train_peft.py` uses the same loss handling (`scripts/train_peft.py:244-248`). Config currently sets `label_smoothing: 0.1` (`configs/config.yaml:56-59`).

21. **[NEW] Noisy same-class mixup is embedding-only augmentation.** `train_epoch` disables it if cached embeddings are not used (`scripts/train.py:330-340`), then calls `noisy_same_class_mixup()` on embeddings with `mixup_alpha`, `noise_std`, `noise_type`, and `mixup_dropout_prob` (`scripts/train.py:350-365`). The mixup utility mixes embeddings with another same-class item, adds Gaussian/dropout/both noise, and leaves labels unchanged (`utils/mixup.py:12-98`).

22. **[NEW] Validation metrics include accuracy, top-5, macro/micro/weighted F1, precision, recall, and inference timing.** `calculate_metrics()` computes these metrics (`utils/metrics.py:33-137`), and training validation adds batch inference timing stats (`scripts/train.py:515-539`). `validate.py` adds classification reports, confusion matrices, per-class metrics, plots, and JSON/CSV outputs (`scripts/validate.py:133-309`).

23. **[NEW] Checkpoint format is a Torch checkpoint dict with direct `model.state_dict()`; there is no source-side key-prefix remapping.** `save_checkpoint()` writes keys `epoch`, `model_state_dict`, `optimizer_state_dict`, `metrics`, `config`, and optional `scheduler_state_dict` (`scripts/train.py:588-612`). Resume and validation load `checkpoint["model_state_dict"]` directly (`scripts/train.py:837-845`, `scripts/validate.py:53-66`). Since the model attributes are `backbone` and `classifier`, state_dict keys are expected to be `backbone.*`, `classifier.weight`, and `classifier.bias` (inferred from `models/dinov3_model.py:84-95`).

24. **[NEW] Checkpoint filenames and class count inference are fixed by training code.** Best model goes to `best_model.pt`, periodic checkpoints to `checkpoint_epoch_{N}.pt`, and final checkpoint to `final_model.pt` under `checkpoint.save_dir` (`scripts/train.py:927-973`, `scripts/train.py:1025-1026`). `num_classes` is inferred from the dataset loader and copied into `config["model"]["num_classes"]` before model creation (`scripts/train.py:785-803`).

25. **[NEW] PEFT/LoRA path exists but is not part of cached-embedding linear probing.** `train_peft.py` imports PEFT, builds `LoraConfig`, defaults target modules to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`, supports `lora` or `pissa`, and wraps `model.backbone = get_peft_model(...)` (`scripts/train_peft.py:32-38`, `scripts/train_peft.py:58-91`, `scripts/train_peft.py:94-116`). It forces `use_cached_embeddings` to false because PEFT needs a backbone forward pass (`scripts/train_peft.py:190-195`, `scripts/train_peft.py:231-238`).

26. **[NEW] `validate.py` is image-only and hardcodes the 7B model name when reconstructing the model.** It requires `--checkpoint`, defaults `--config` and `--output_dir`, and accepts `--val_csv`/`--no_attention` (`scripts/validate.py:473-511`). It infers `num_classes` from the checkpoint config or defaults to 36, but constructs `DINOv3LinearClassifier(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", ...)` instead of using the checkpoint model name (`scripts/validate.py:547-566`).

27. **[NEW] Configs expose the knobs needed for port planning.** `configs/config.yaml` contains experiment cache flags/embedding path (`configs/config.yaml:3-8`), model/freeze/classes/dim (`configs/config.yaml:9-16`), PEFT settings (`configs/config.yaml:17-27`), data paths/image size/batch/workers/augmentations (`configs/config.yaml:28-38`), training/mixup/loss/validation/checkpoint/logging/hardware/seed settings (`configs/config.yaml:40-99`).

28. **[NEW] Sweep configs are W&B-specific and optional for the port.** `configs/sweep_models.yaml` tests `vitb16`, `vitl16`, and `vit7b16` with fixed best mixup values (`configs/sweep_models.yaml:10-34`); `configs/sweep_mixup.yaml` performs an 81-run grid over mixup/noise/dropout settings (`configs/sweep_mixup.yaml:4-30`). `run_sweep.py` injects `program: scripts/train.py` and command args into W&B sweep config (`scripts/run_sweep.py:65-88`).

29. **[NEW] Source dependencies are PyTorch/Transformers/PEFT/Pandas/Sklearn/W&B/YAML/plotting; no `timm` dependency is listed.** `requirements.txt` pins minimums `torch>=2.0.0`, `torchvision>=0.15.0`, `transformers>=4.30.0`, `huggingface-hub>=0.15.0`, `peft>=0.10.0`, `pandas>=1.5.0`, `numpy>=1.23.0`, `Pillow>=9.0.0`, `scikit-learn>=1.2.0`, `wandb>=0.15.0`, `pyyaml>=6.0`, `matplotlib>=3.6.0`, `seaborn>=0.13.0`, `tqdm>=4.65.0` (`requirements.txt:1-29`).

30. **[NEW] Source docs explain HF login but mainly for the 7B model, not all requested sizes.** README setup requires `huggingface-cli login` for DINOv3 authentication (`README.md:91-95`) and troubleshooting repeats that requirement (`README.md:212-217`, `README.md:289-295`). QUICKSTART says the 7B model may require terms acceptance/access request (`QUICKSTART.md:30-45`).

## **Port manifest**

| Source artifact | What it provides | Port decision for the 4 requirements |
|---|---|---|
| `scripts/extract_embeddings.py` | CLI and function-level behavior for batched DINOv3 inference, CLS extraction through `model.get_embeddings()`, per-image `.pkl` save, split dirs, timing file (`scripts/extract_embeddings.py:47-124`, `scripts/extract_embeddings.py:127-286`). | **Essential behavior, replace implementation shape as needed.** Requirement 3 needs inference plus embedding save; requirement 4 needs precompute-then-train workflow. |
| `scripts/extract_embeddings_all_models.py` | Model loop wrapper and model-specific embedding dirs; mutates config and currently covers B/L/7B (`scripts/extract_embeddings_all_models.py:13-19`, `scripts/extract_embeddings_all_models.py:38-57`, `scripts/extract_embeddings_all_models.py:123-155`). | **Replaceable.** Need safer non-mutating orchestration and requested S+/B/H+ list. |
| `models/dinov3_model.py` | HF loading, login/access error text, eager attention, embedding-dim detection, frozen backbone, linear head, `forward()`, `get_embeddings()`, attention maps (`models/dinov3_model.py:30-228`). | **Essential.** Core model wrapper for all requirements; must be adapted to target model family conventions. |
| `models/__init__.py` | Exports `DINOv3LinearClassifier` (`models/__init__.py:1-5`). | **Replaceable/trivial.** Mirror target package exports if target has model registry. |
| `data/dataset.py` | Amazon class list/mapping, CSV loading, missing-image filtering, RGB image loading, transforms (`data/dataset.py:17-29`, `data/dataset.py:31-160`, `data/dataset.py:174-210`). | **Partly essential.** Reuse class mapping and preprocessing decisions, but integrate with target datasets without breaking current pipeline. |
| `data/embedding_dataset.py` | Cached embedding dataset that maps CSV rows to safe `.pkl` files and returns tensors (`data/embedding_dataset.py:20-186`). | **Essential for training from embeddings.** May be adapted to target dataset API or replaced by a target-native cached-embedding dataset. |
| `scripts/train.py` | Data-loader branch, train/val loops, cached embedding classifier path, optimizer/scheduler/loss, metrics logging, checkpoint save/resume (`scripts/train.py:123-304`, `scripts/train.py:307-612`, `scripts/train.py:785-1035`). | **Essential behavior.** Port linear-probe-from-embeddings path without breaking existing classifiers. |
| `scripts/train_peft.py` | Optional PEFT LoRA/PiSSA finetuning path and PEFT checkpoint reuse (`scripts/train_peft.py:58-159`, `scripts/train_peft.py:190-380`). | **Optional for the 4 stated requirements.** Include in plan as future/optional unless team model requires PEFT adapters. |
| `scripts/validate.py` | Standalone image-validation script, metrics, plots, JSON/CSV outputs, checkpoint load (`scripts/validate.py:53-66`, `scripts/validate.py:69-309`, `scripts/validate.py:473-615`). | **Optional/replaceable.** Useful evaluation logic; hardcoded 7B reconstruction must not be copied as-is. |
| `utils/metrics.py` | Accuracy, top-5, F1/precision/recall metrics (`utils/metrics.py:10-167`). | **Essential or target-equivalent.** Use target metrics if existing; otherwise port. |
| `utils/mixup.py` | Embedding-level noisy same-class mixup (`utils/mixup.py:12-172`). | **Optional.** Not required by inference/embedding/classification workflow, but relevant if reproducing SINCHI experiments. |
| `utils/logging_utils.py` | Basic console/file logger (`utils/logging_utils.py:9-54`). | **Replaceable.** Prefer target logging conventions. |
| `configs/config.yaml` | Defaults for model, data, cached embeddings, training, loss, validation, checkpoint, W&B, hardware (`configs/config.yaml:3-99`). | **Essential as requirements source, not as a literal config format.** Translate knobs into target config/CLI. |
| `configs/sweep_models.yaml` | W&B model sweep over B/L/7B (`configs/sweep_models.yaml:10-34`). | **Optional and incomplete.** Replace with S+/B/H+ model selection if target supports sweeps. |
| `configs/sweep_mixup.yaml` | W&B grid for mixup/noise params (`configs/sweep_mixup.yaml:4-30`). | **Optional.** Only needed for experiment replication. |
| `scripts/run_sweep.py`, `scripts/run_sweep_agent.py`, `scripts/extract_validation_metrics.py` | W&B sweep initialization/agent and log parsing (`scripts/run_sweep.py:65-132`, `scripts/run_sweep_agent.py:11-59`, `scripts/extract_validation_metrics.py:18-100`). | **Optional.** Not needed for core port; avoid adding W&B coupling unless target already uses it. |
| `scripts/verify_peft_target_modules.py` | PEFT target-module introspection (`scripts/verify_peft_target_modules.py:22-47`, `scripts/verify_peft_target_modules.py:69-119`). | **Optional diagnostic.** Useful if PEFT support is included. |
| `README.md`, `QUICKSTART.md`, `EXPERIMENT_FEATURES.md`, `SWEEP_README.md`, `MODEL_SWEEP_README.md`, `MODEL_TESTING_GUIDE.md` | Intended workflow, HF login notes, data format, checkpoint docs, embedding schema, model/sweep notes (`README.md:91-165`, `QUICKSTART.md:30-88`, `EXPERIMENT_FEATURES.md:13-151`, `MODEL_SWEEP_README.md:13-31`, `MODEL_TESTING_GUIDE.md:15-80`, `SWEEP_README.md:17-101`). | **Essential documentation inputs.** Target README must explain login, auto-download/check behavior, and two-stage workflow for all model sizes. |
| `requirements.txt` | Dependency floor versions (`requirements.txt:1-29`). | **Essential dependency inventory.** Target dependency changes need version verification, especially Transformers DINOv3 support. |

## **Embedding file format**

- **Producer:** `scripts/extract_embeddings.py::extract_embeddings()`.
- **Inputs:** `AnimalClassificationDataset` rows from `data.train_csv` or `data.val_csv` under `data.root_dir`; dataset returns `(image_tensor, label_idx, image_path)` (`scripts/extract_embeddings.py:193-208`, `scripts/extract_embeddings.py:227-242`, `data/dataset.py:135-160`).
- **Transform:** validation transform only: bicubic resize to `image_size`, `ToTensor`, ImageNet normalize (`scripts/extract_embeddings.py:181-183`, `data/dataset.py:201-210`).
- **Batching:** `DataLoader(..., batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"], pin_memory=True, drop_last=False)` (`scripts/extract_embeddings.py:70-78`, `scripts/extract_embeddings.py:201-208`, `scripts/extract_embeddings.py:235-242`).
- **Embedding value:** `model.get_embeddings(images)` returns CLS-token features (`scripts/extract_embeddings.py:85-91`, `models/dinov3_model.py:148-173`).
- **On-disk layout:** `<output_dir>/train/<safe_image_path>.pkl` and `<output_dir>/val/<safe_image_path>.pkl`; all-model wrapper uses `<base_embedding_dir>/<model_id>/train|val/*.pkl` (`scripts/extract_embeddings.py:200-208`, `scripts/extract_embeddings.py:234-242`, `scripts/extract_embeddings_all_models.py:22-35`).
- **Filename scheme:** `image_path.replace("/", "_").replace("\\", "_") + ".pkl"` (`scripts/extract_embeddings.py:32-44`, `data/embedding_dataset.py:133-145`).
- **Payload:** Python pickle of a dict:
  ```python
  {
      "embedding": numpy.ndarray,  # shape (embedding_dim,), from CPU tensor .numpy()
      "label": int,                # label from dataset at extraction time
      "image_path": str,           # original relative image path
  }
  ```
  Evidence: `scripts/extract_embeddings.py:107-118`; schema also documented in `EXPERIMENT_FEATURES.md:133-145`.
- **Skip/error behavior:** existing `.pkl` files are skipped; save exceptions are logged and counted (`scripts/extract_embeddings.py:102-123`). `EmbeddingDataset` later filters rows without an expected file and raises if none are found (`data/embedding_dataset.py:88-125`).
- **Extra file:** `extract_embeddings.py` also writes `<output_dir>/extraction_timing.txt` with model, output dir, split timings, and total time (`scripts/extract_embeddings.py:271-286`).

## **Checkpoint format**

- **Producer:** `scripts/train.py::save_checkpoint()`, reused by `train_peft.py` (`scripts/train.py:588-612`, `scripts/train_peft.py:47-55`).
- **File names:** `best_model.pt` when validation metric improves, `checkpoint_epoch_{epoch+1}.pt` periodically, `final_model.pt` at completion (`scripts/train.py:927-973`, `scripts/train.py:1025-1026`; PEFT mirrors this at `scripts/train_peft.py:320-378`).
- **Container:** `torch.save(checkpoint, save_path)` where `checkpoint` is:
  ```python
  {
      "epoch": int,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "metrics": dict,
      "config": dict,
      # optional
      "scheduler_state_dict": scheduler.state_dict(),
  }
  ```
  Evidence: `scripts/train.py:600-611`.
- **State dict layout:** no explicit prefix translation is performed; resume and validate load direct (`scripts/train.py:837-845`, `scripts/validate.py:53-66`). Because `DINOv3LinearClassifier` attributes are `backbone` and `classifier`, direct PyTorch keys are expected to be `backbone.*`, `classifier.weight`, `classifier.bias`; this key layout is inferred from the module attributes (`models/dinov3_model.py:84-95`). PEFT checkpoints additionally include PEFT-wrapped `backbone.*` adapter keys because `model.backbone = get_peft_model(...)` before saving (`scripts/train_peft.py:113-116`, `scripts/train_peft.py:320-378`).
- **Class count:** `train.py` sets `config["model"]["num_classes"] = label_info["num_classes"]` before saving (`scripts/train.py:785-803`); `validate.py` reads that value if present or defaults to 36 (`scripts/validate.py:547-553`).
- **Caveat:** `validate.py` ignores the saved model name and reconstructs with hardcoded `facebook/dinov3-vit7b16-pretrain-lvd1689m` (`scripts/validate.py:555-566`), so S+/B/H+ checkpoints need a corrected loader in the target.

## **Sources**

- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/extract_embeddings.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/extract_embeddings_all_models.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/train.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/train_peft.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/validate.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/run_sweep.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/run_sweep_agent.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/extract_validation_metrics.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/scripts/verify_peft_target_modules.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/models/dinov3_model.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/models/__init__.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/data/dataset.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/data/embedding_dataset.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/data/__init__.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/utils/metrics.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/utils/logging_utils.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/utils/__init__.py`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/configs/config.yaml`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/configs/sweep_models.yaml`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/configs/sweep_mixup.yaml`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/README.md`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/QUICKSTART.md`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/EXPERIMENT_FEATURES.md`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/SWEEP_README.md`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/MODEL_SWEEP_README.md`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/MODEL_TESTING_GUIDE.md`
- `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/requirements.txt`

## **Cross-Area Connections**

- **researcher-2 / target-architecture:** The source's cached-embedding path expects classifier training to accept raw embedding tensors and call only `model.classifier(embeddings)` (`scripts/train.py:346-368`). The target plan should add this as a DINOv3-specific extension point without changing current image-based classifier inference/training paths.
- **researcher-2 / target-architecture:** The source embedding schema is per-image pickle with `embedding`, `label`, and `image_path` keys (`scripts/extract_embeddings.py:107-118`). If target prefers batch `.npz`/`.pt`, add a compatibility loader or migration path; otherwise existing source training semantics depend on the safe filename scheme (`data/embedding_dataset.py:133-145`).
- **researcher-2 / target-architecture:** Source datasets assume `Photo_ID`/`Label` and the official 36-class Amazon mapping (`data/dataset.py:17-29`, `data/dataset.py:37-57`). Target integration must adapt to target dataset abstractions and avoid forcing this mapping onto unrelated MegaDetector-Classifier pipelines.
- **researcher-3 / prior-art-branch:** Validate any existing DINOv3 branch against the source caveats: source `validate.py` hardcodes 7B (`scripts/validate.py:555-566`), all-model extraction covers B/L/7B (`scripts/extract_embeddings_all_models.py:13-19`), and config currently uses H+ (`configs/config.yaml:10-15`).
- **researcher-4 / dinov3-model-access:** Source docs and model wrapper require Hugging Face login/access handling (`models/dinov3_model.py:54-60`, `README.md:91-95`, `QUICKSTART.md:30-45`) but do not implement an explicit preflight check/download function beyond `AutoModel.from_pretrained()`.
- **researcher-4 / dinov3-model-access:** Source dependency floor `transformers>=4.30.0` may be insufficient for current DINOv3 HF model classes; verify exact target version needed before porting (`requirements.txt:1-6`).
- **researcher-4 / dinov3-model-access:** Source does not list the requested small-plus/base/huge-plus trio together. It mentions B/L/7B in sweeps (`configs/sweep_models.yaml:15-17`) and H+ in config (`configs/config.yaml:10-15`); exact S+/B/H+ repo IDs and dims must be confirmed externally.

## **Gaps**

1. Source all-model extraction and model sweep cover B/L/7B, not the requested S+/B/H+ set (`scripts/extract_embeddings_all_models.py:13-19`, `configs/sweep_models.yaml:15-17`).
2. Source config uses `facebook/dinov3-vith16plus-pretrain-lvd1689m` but comments call it 7B, while docs also describe 7B as current default (`configs/config.yaml:10-15`, `README.md:1-12`). This inconsistency must be corrected in target docs/configs.
3. `validate.py` reconstructs only `facebook/dinov3-vit7b16-pretrain-lvd1689m` and will not reliably validate S+/B/H+ checkpoints as-is (`scripts/validate.py:555-566`).
4. Source model wrapper explains HF login on load failure but has no explicit weight-availability preflight/download/check function required by the target workflow (`models/dinov3_model.py:44-60`).
5. Source docs explain authentication mainly for the 7B model, not for all required S+/B/H+ variants (`README.md:91-95`, `QUICKSTART.md:30-45`).
6. `extract_embeddings_all_models.py` mutates `configs/config.yaml` on disk while running (`scripts/extract_embeddings_all_models.py:38-57`); target should avoid this to protect existing pipeline behavior.
7. Source `EmbeddingDataset` trusts CSV labels and ignores the stored pickle `label` field, so stale or mismatched embedding files are not detected (`data/embedding_dataset.py:164-186`).
8. Source dependency floor only says `transformers>=4.30.0`; exact DINOv3 support/version constraints remain unverified (`requirements.txt:1-6`).

## **Confidence**

| Finding(s) | Confidence | Basis |
|---|---|---|
| 1-6 | HIGH | Verified by reading extraction scripts, embedding dataset, and docs with line citations. |
| 7 | HIGH | Verified source model lists/config mutation directly; requested S+/B/H+ mismatch is based on source search and cited lines. |
| 8-12 | HIGH | Verified by reading `models/dinov3_model.py` and `models/__init__.py`. |
| 13-18 | HIGH | Verified by reading dataset, cached dataset, and train data-loader/loop branches. |
| 19-24 | HIGH | Verified by reading train optimizer/loss/validation/checkpoint/save paths. State_dict prefix details are partly inferred from PyTorch module attribute names and direct `model.state_dict()` usage. |
| 25 | HIGH | Verified by reading `train_peft.py`; whether PEFT is needed for the team's Amazon model is unknown. |
| 26 | HIGH | Verified by reading `validate.py`. |
| 27-30 | HIGH | Verified by reading configs, requirements, README, and QUICKSTART. Exact modern DINOv3 dependency requirements are delegated to model-access research. |

STATUS: NEW=30 GAPS=8
