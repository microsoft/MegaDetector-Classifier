# Documenter change plan — doc-fix round 2 (CONFIRMING round)

Branch: `feat/dinov3-port`. Convergence mode: `editing`. This is the confirming
(no-change) round that editing-mode requires after round 1 applied fixes. I
re-read the current on-disk state of all 10 ledger docs and re-verified every
applied and pre-existing claim against the actual source/config. No docs were
edited in this turn.

## Re-verification basis (code/config re-read this round)
- `scripts/extract_embeddings.py` — `DEFAULT_PHOTO_COL="Photo_ID"`,
  `DEFAULT_LABEL_COL="Label"` (L15–16); `--model-size choices=["small_plus","base","huge_plus"]`
  (L21); `_split_column` accepts `split/Split/dataset/Dataset/subset/Subset` (L36–40).
- `scripts/train_dinov3.py` — stage-2 `photo_col`/`label_col` config overrides (L51–52);
  writes `<output_dir>/best_model.pt` via `model.export_classifier_only()` (L123–127).
- `configs/dinov3_linear.yaml` — all doc-listed keys present (`model_size`, `embeddings_dir`,
  `train_csv`, `val_csv`, `output_dir`, `epochs`, `lr`, `batch_size`).
- `src/algorithms/dinov3_linear.py` — `DINOv3Linear.__init__(self, conf,
  train_class_counts=None, id_to_labels=None)` (main.py-routable, L34–47);
  `DINOv3LinearProbe` is the stage-2 cached-embedding class (module docstring L1–10,
  `__all__` L22); `_wrap_with_peft` raises `NotImplementedError("PEFT lands in Phase 7")`
  (L114–116); `export_classifier_only` emits `format: "dinov3_classifier_only"` (L181–184).
- `src/algorithms/__init__.py` — `from .dinov3_linear import *` re-exports `DINOv3Linear`.
- `src/models/dinov3.py` — `DINOv3Classifier(freeze_backbone=False)` raises
  `NotImplementedError("Full FT lands in Phase 6")` (L37–38).
- `src/models/dinov3_registry.py` — `small_plus`/`base`/`huge_plus` dims 384/768/1280 and
  repo ids match the docs/dinov3.md §4 table.
- `src/models/classification/__init__.py` — lazy DINOv3 import gated on
  `transformers`+`huggingface_hub`; `__getattr__` raises actionable `ImportError`.
- `src/models/classification/dinov3_base/__init__.py` — `__all__` exports
  `DINOv3AmazonRainforest`, `DINOv3BaseInference`, `DINOv3CustomWeights`, `DINOv3Transform`.
- `src/models/classification/dinov3_base/amazon.py` — every manifest entry has
  `head_url=None`/`sha256=None` (TODO-Zenodo, L56–88); `_validate_manifest(...,
  require_published=weights is None)` raises `NotImplementedError` when unpublished
  (L102–126), confirming the README/dinov3.md `weights=<path>` caveat.
- `src/models/classification/dinov3_base/base_classifier.py` — inference methods
  `single_image_classification(img, …)` (L386), `batch_image_classification(…)` (L417),
  `batch_image_classification_with_embeddings(…)` (L493).
- `src/datasets/custom.py` — `Custom_Crop.__init__(self, conf, image_size: int = 224)`
  (L267), confirming the `Custom_Crop(conf, image_size=518)` example.
- `requirements-dinov3.txt` — `transformers>=4.56,<5`, `peft>=0.10.0,<0.14.0`,
  `huggingface_hub>=0.20,<1`.

## Per-file confirmation
- **README.md** — `DINOv3AmazonRainforest` loader bullet, the `dinov3_base/` tree node, and
  the `#dinov3-classification-experimental` anchor (heading at L207) all resolve and match code. OK.
- **SECURITY.md** — unedited; no code-coupled or DINOv3 claims. OK.
- **docs/index.md** — DINOv3 pointer to `dinov3.md` resolves. OK.
- **docs/installation.md** — "Optional: DINOv3 classification" subsection + `requirements-dinov3.txt`
  install line + `dinov3.md` link all accurate. OK.
- **docs/usage.md** — unedited; documents the base ResNet/timm `main.py` workflow; no drift. OK.
- **docs/tags.md** — unedited MkDocs tags stub; no code-coupled claims. OK.
- **docs/cite.md** — unedited citation page; no code-coupled claims. OK.
- **docs/build_mkdocs.md** — unedited; its own instructions are accurate. (The mkdocs.yml
  orphan-nav matter is out of ledger scope per the task brief and not re-proposed here.)
- **docs/dinov3.md** — Overview cross-link, §4 model-sizes table, §5 inference example
  signatures, §6 stage-1 CSV columns / `--split` names / stage-2 output + config keys all
  verified against code. OK.
- **docs/dinov3_experimental.md** — EXPERIMENTAL status line, GPU-memory note with the
  size-name guardrail, `Custom_Crop(conf, image_size=518)`, and the Implemented-vs-Deferred
  scope (deferred stubs still raise `NotImplementedError`) all verified. OK.

## Result
All 10 ledger docs re-verified accurate, consistent, and complete against the current
code/config at this HEAD. Round 1's fixes (ITEM-001..011 + ITEM-INQ-002/003) remain correctly
applied; no new drift found; no in-scope doc defects remain open. The two non-blocking LOW
clarity suggestions the inquisitor noted last round (an optional usage.md → dinov3.md pointer;
the stage-1/stage-2 `photo_col`/`label_col` parenthetical in dinov3.md §6) were explicitly
recorded as "NOT drift" and out of necessity scope — re-editing in this confirming round would
re-open the editing loop and is therefore not warranted. Out-of-ledger items (mkdocs.yml nav
orphans; the flavour-D code matter) are excluded by the task brief and not re-proposed.

Nothing to change.

STATUS: NOTHING-TO-DO
