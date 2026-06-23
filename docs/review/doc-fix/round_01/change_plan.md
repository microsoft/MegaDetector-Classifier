# Doc-fix round 1 — change plan (Documenter)

Scope: 10 target docs in `docs/review/doc-fix/SCOPE_LEDGER.json`. Every claim below was
verified against the current source on branch `feat/dinov3-port`. The authoritative spec
(`DINOv3_classification_port_plan.md`) was read but is NOT an output and was NOT edited.

Editable files (ledger): README.md, SECURITY.md, docs/index.md, docs/installation.md,
docs/usage.md, docs/tags.md, docs/cite.md, docs/build_mkdocs.md, docs/dinov3.md,
docs/dinov3_experimental.md. `mkdocs.yml` is NOT editable (flagged under NEEDS DECISION).

## Verification anchors (code checked)
- `src/models/dinov3_registry.py`: `DINOV3_MODEL_IDS` (5 sizes), `DINOV3_DIMS`, `resolve_model_id`, `ensure_model_available` (gated-error text matches docs).
- `src/models/classification/dinov3_base/base_classifier.py`: `DINOv3BaseInference`, `_extract_state_dict` (flavours A/B/C/D implemented), `batch_image_classification`, `batch_image_classification_with_embeddings`, `extract_embeddings`, `results_generation` dict schema.
- `src/models/classification/dinov3_base/amazon.py`: `DINOv3AmazonRainforest(model_size="base", weights=None, device="cpu")`, `DINOV3_AMAZON_MANIFEST` (head_url/sha256 = None → NotImplementedError without local weights), 36 CLASS_NAMES.
- `src/models/classification/dinov3_base/custom_weights.py`: `DINOv3CustomWeights`, `torch.load(..., weights_only=False)`, PEFT flavour-C inference path.
- `src/models/classification/dinov3_base/transforms.py`: `DINOv3Transform` (bicubic 518, ImageNet norm).
- `src/models/dinov3.py`: `DINOv3Classifier`, `get_embeddings` (CLS = last_hidden_state[:,0,:]); `freeze_backbone=False` → NotImplementedError("Full FT lands in Phase 6").
- `src/algorithms/dinov3_linear.py`: `DINOv3Linear` (main.py-routable; `_wrap_with_peft` → NotImplementedError "PEFT lands in Phase 7") and `DINOv3LinearProbe` (cached-embedding probe; `export_classifier_only`).
- `src/datasets/dinov3_embedding.py`: `safe_name`, `build_label_mapping(label_col="Label")`, `DINOv3EmbeddingDataset(photo_col="Photo_ID", label_col="Label")`, `DINOv3EmbeddingDataModule`.
- `scripts/extract_embeddings.py`: flags `--model-size {small_plus,base,huge_plus} --csv --root --out --split {train,val,both} [--batch-size --num-workers --device]`; CSV image-path col `Photo_ID`, label col `Label`.
- `scripts/train_dinov3.py`: flag `--config` (default `configs/dinov3_linear.yaml`); writes `<output_dir>/best_model.pt` (classifier-only flavour D).
- `configs/dinov3_linear.yaml`: keys model_size, embeddings_dir, train_csv, val_csv, num_classes, epochs, lr, label_smoothing, batch_size, num_workers, output_dir, image_size.
- `requirements-dinov3.txt`: `transformers>=4.56,<5`, `peft>=0.10.0,<0.14.0`, `huggingface_hub>=0.20,<1`.
- `src/models/classification/__init__.py`: guarded DINOv3 import gates on `transformers`+`huggingface_hub` (not `peft`); actionable ImportError via `__getattr__`.
- `mkdocs.yml` nav: only index/installation/usage/tags/cite/build_mkdocs — DINOv3 pages absent.

---

## Change items

ITEM-001 | docs/dinov3_experimental.md:1-5 | Retitle to "DINOv3 fine-tuning notes (experimental)" and replace the phase-roadmap status blockquote (Phase 5 MVP; "Full fine-tune lands in Phase 6; PEFT/LoRA in Phase 7; SINCHI ckpts in Phase 8"; "Full docs land in Phase 9.") with an accurate one-line status that links `docs/dinov3.md` as the canonical install/auth/inference/two-stage guide. | Internal phase labels are stale and user-facing; `docs/dinov3.md` already exists, so "Full docs land in Phase 9" is false; the orphan page should point to the canonical doc.

ITEM-002 | docs/dinov3_experimental.md:35 | Replace "Full memory table lands in `docs/dinov3.md` (P9)." with a self-contained statement (larger backbones — ViT-B/L/H+ — need proportionally more VRAM). | `docs/dinov3.md` exists and contains no memory table; the forward reference is a stale, unfulfilled promise.

ITEM-003 | docs/dinov3_experimental.md:39-40 | Fix the example `Custom_Crop(image_size=518)` → `Custom_Crop(conf, image_size=518)`. | `Custom_Crop.__init__(self, conf, image_size=224)` requires the positional `conf` (src/datasets/custom.py:267); the example as written would TypeError.

ITEM-004 | docs/dinov3_experimental.md:42-54 | Rewrite "Scope (P5 MVP)" + "Not included (later phases)" to current reality: the DINOv3 inference surface (`DINOv3AmazonRainforest`, `DINOv3BaseInference`, `batch_image_classification_with_embeddings`), the SINCHI checkpoint loaders (flavours A–D in `_extract_state_dict`), the two-stage `scripts/extract_embeddings.py`+`scripts/train_dinov3.py`, and `configs/dinov3_linear.yaml` now exist; only full-backbone fine-tuning and PEFT/LoRA *training* remain `NotImplementedError` stubs. Point to `docs/dinov3.md` for the supported workflow. | Verified implemented: flavours B/C/D in base_classifier.py; amazon.py, scripts/*, configs/dinov3_linear.yaml present. Still deferred: `DINOv3Classifier(freeze_backbone=False)` and `DINOv3Linear._wrap_with_peft` raise NotImplementedError. The original "Not included: SINCHI checkpoint flavours (P8), sample configs & full docs (P9)" is now false.

ITEM-005 | docs/dinov3_experimental.md:46-48 | Correct "reload a `.ckpt` trained by this repo" → "load a local DINOv3 checkpoint (`.pt`/`.ckpt`)"; keep the trusted-source / `torch.load(weights_only=False)` security caveat. | Stage-2 emits `best_model.pt`; `DINOv3CustomWeights` calls `torch.load(weights, ..., weights_only=False)` on a path (custom_weights.py:170) — the security caveat is accurate, but ".ckpt"-only phrasing is not.

ITEM-006 | README.md:141 | Add a bullet to "Available loaders" for `DINOv3AmazonRainforest`, marked optional/experimental, noting it currently needs a local checkpoint (`weights=<path>`) until the Zenodo weights publish, cross-linking the "DINOv3 classification (experimental)" section (`#dinov3-classification-experimental`). | `DINOv3AmazonRainforest` is exported from `src.models.classification` when DINOv3 deps are present; amazon.py raises NotImplementedError without local weights, so the caveat is required for accuracy.

ITEM-007 | README.md:176 | Add `dinov3_base/  # DINOv3 inference (optional, experimental)` under the `classification/` subpackage tree in Repository Structure (alongside resnet_base/timm_base/speciesnet_base). | `src/models/classification/` contains `dinov3_base/`; the tree currently omits it while listing the other three families.

ITEM-008 | docs/dinov3.md:91 | After the stage-1 command, add a note that the CSV must provide an image-path column `Photo_ID` and a class-name column `Label` (overridable via `photo_col`/`label_col` in the stage-2 config), which differ from the base pipeline's `path`/`classification`/`label` columns; optionally mention the split column (`split`/`Split`/`dataset`/…) used by `--split`. | extract_embeddings.py DEFAULT_PHOTO_COL="Photo_ID"/DEFAULT_LABEL_COL="Label"; DINOv3EmbeddingDataset & build_label_mapping default to Photo_ID/Label; `_split_column` lists the accepted split-column names. Without this, a user running the doc verbatim with the base `annotations.csv` hits `KeyError: ... 'Photo_ID'`.

ITEM-009 | docs/dinov3.md:100 | Add that stage-2 writes a classifier-only (flavour D) checkpoint to `<output_dir>/best_model.pt`, and list the key `configs/dinov3_linear.yaml` fields (`model_size`, `embeddings_dir`, `train_csv`, `val_csv`, `output_dir`, `epochs`, `lr`, `batch_size`). | train_dinov3.py saves `output_dir/best_model.pt` via `export_classifier_only()`; config fields verified in configs/dinov3_linear.yaml.

ITEM-010 | docs/dinov3.md:1-5 | Add a one-line pointer in the Overview to `docs/dinov3_experimental.md` for GPU-memory guidance and experimental main.py-based training notes. | Connects the two DINOv3 pages and reduces orphan isolation in rendered output (both pages currently stand alone).

ITEM-011 | docs/installation.md:62-67 | Add a short "Optional: DINOv3 classification" subsection before "Next Steps": `pip install -r requirements-dinov3.txt`, note the deps are optional and the base pipeline works without them, and link `docs/dinov3.md`. | requirements-dinov3.txt exists and README documents the same optional path; installation.md currently omits any DINOv3 install path (completeness/consistency with README).

## NEEDS DECISION (cannot fix — outside editable ledger)

ITEM-012 | mkdocs.yml:71-78 | `docs/dinov3.md` and `docs/dinov3_experimental.md` are NOT in the mkdocs `nav:`, so they are orphan pages; `mkdocs build --strict` (the command docs/build_mkdocs.md tells contributors to run) fails on "pages exist but are not in nav" warnings. Recommend adding both under the "MegaDetector-Classifier" nav group (e.g. "DINOv3 (experimental): dinov3.md"). | mkdocs.yml is not in the editable ledger, so this cannot be fixed in this round. Flagged for the lead/maintainer. Verified: nav lists only index/installation/usage/tags/cite/build_mkdocs while both DINOv3 pages exist on disk.

## Notes / non-issues (verified accurate; no change)
- README DINOv3 section (lines 205-214), docs/dinov3.md §§2–5 & 7, install command, gated-access steps, model-size table (S+/B/H+ dims 384/768/1280), HF auth options, and troubleshooting all match the code.
- `transformers>=4.56.0` (doc) vs `transformers>=4.56,<5` (requirements-dinov3.txt) are equivalent floors — no change.
- docs/index.md, usage.md, tags.md, cite.md, build_mkdocs.md, SECURITY.md contain no DINOv3 references and no detected drift against current code.

STATUS: PLAN-READY
