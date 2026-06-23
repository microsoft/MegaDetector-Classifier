# Documenter report — doc-fix round 1 (APPLY phase)

Branch: `feat/dinov3-port`. Applied only inquisitor-APPROVED items plus the
in-ledger independent finding (ITEM-INQ-002). Every claim re-verified against
source before writing. No source code, `mkdocs.yml`, or non-ledger files touched.

## Changes Applied

| ITEM | File | What changed | Why |
|---|---|---|---|
| ITEM-001 | docs/dinov3_experimental.md | Retitled `# DINOv3 fine-tuning notes (experimental)`; replaced the stale phase-roadmap status blockquote with a one-line EXPERIMENTAL status that links `docs/dinov3.md` as the canonical guide. | The Phase 6/7/8/9 labels were false: SINCHI flavours already exist (`_extract_state_dict` flavours A–D, base_classifier.py:80–136) and `docs/dinov3.md` already exists, so "Full docs land in Phase 9" was wrong. |
| ITEM-002 | docs/dinov3_experimental.md | Replaced "Full memory table lands in `docs/dinov3.md` (P9)." with a self-contained VRAM statement (larger ViT-B/L/H+ backbones need proportionally more VRAM). Also added an ITEM-INQ-003 guardrail clarifying these size names apply to the `main.py`/`DINOv3Linear` route (any HF repo id), while the two-stage scripts accept only `small_plus`/`base`/`huge_plus`. | The forward reference was unfulfilled — `docs/dinov3.md` contains no memory table. Guardrail prevents the memory note being misread as the two-stage scripts accepting ViT-S/L. |
| ITEM-003 | docs/dinov3_experimental.md | `Custom_Crop(image_size=518)` → `Custom_Crop(conf, image_size=518)`. | `Custom_Crop.__init__(self, conf, image_size=224)` requires the positional `conf` (src/datasets/custom.py:267); the old example would `TypeError`. |
| ITEM-004 | docs/dinov3_experimental.md | Rewrote "Scope (P5 MVP)" + "Not included (later phases)" into "Implemented and supported today" (inference surface, flavours A–D, two-stage scripts + config) vs "Still deferred (raise `NotImplementedError`)" (full-FT `DINOv3Classifier(freeze_backbone=False)`, dinov3.py:38; PEFT `DINOv3Linear._wrap_with_peft`, dinov3_linear.py:110). Points to `docs/dinov3.md`. | The old "Not included" list was false; verified the deferred stubs still raise `NotImplementedError`. |
| ITEM-005 | docs/dinov3_experimental.md | (Applied within the ITEM-004 rewrite.) "reload a `.ckpt` trained by this repo" → "load a local DINOv3 checkpoint (`.pt`/`.ckpt`)"; kept the trusted-source / `torch.load(weights_only=False)` caveat (custom_weights.py:170). | `.ckpt`-only phrasing was inaccurate (stage-2 emits `best_model.pt`). Per ITEM-INQ-001 guardrail, the new wording does NOT assert the repo's own stage-2 output is reloadable through the inference loader. |
| ITEM-006 | README.md | Added a `DINOv3AmazonRainforest` bullet to "Available loaders", marked optional/experimental, noting it needs `weights=<path>` until Zenodo publishes, cross-linking `#dinov3-classification-experimental`. | `DINOv3AmazonRainforest` is exported from `src.models.classification` (`__init__.py:20`); amazon.py:86–93 raises `NotImplementedError` when `weights=None` and the manifest head_url is None. |
| ITEM-007 | README.md | Added `dinov3_base/  # DINOv3 inference (optional, experimental)` as the final child under `classification/`; changed `speciesnet_base/` from the final-child connector (`└──`) to a mid-child (`├──`) so box-drawing stays valid. | `src/models/classification/dinov3_base/` exists but the tree omitted it. |
| ITEM-008 | docs/dinov3.md | After the stage-1 command, documented the required CSV columns `Photo_ID` / `Label` (override via `photo_col`/`label_col`), contrasted with the base pipeline's `path`/`classification`/`label`, and listed the accepted `--split` column names. | extract_embeddings.py:16–17 defaults; `_split_column` (L37–41) accepts split/Split/dataset/Dataset/subset/Subset. Prevents a verbatim-copy `KeyError`. |
| ITEM-009 | docs/dinov3.md | Added that stage-2 writes a classifier-only (flavour D) checkpoint to `<output_dir>/best_model.pt` and listed key `configs/dinov3_linear.yaml` fields. | train_dinov3.py:125–126 writes `best_model.pt` via `export_classifier_only()`; config keys verified in configs/dinov3_linear.yaml. States only what stage-2 WRITES (no reloadability claim — ITEM-INQ-001 guardrail). |
| ITEM-010 | docs/dinov3.md | Added a one-line Overview pointer to `docs/dinov3_experimental.md` for GPU-memory + experimental `main.py` training notes. | Connects the two DINOv3 pages, reducing orphan isolation. |
| ITEM-011 | docs/installation.md | Added an "Optional: DINOv3 classification" subsection before "Next Steps": `pip install -r requirements-dinov3.txt`, optional-deps note, link to `docs/dinov3.md`. | requirements-dinov3.txt exists; README documents the same optional path; installation.md previously had none. |
| ITEM-INQ-002 | docs/index.md | Added a one-line pointer from the "Get Started" section to `docs/dinov3.md`. | In-ledger discoverability fix; mitigates orphan isolation of the DINOv3 pages ahead of the out-of-scope nav fix. |

## Approval Decisions

| ITEM | Decision | Notes |
|---|---|---|
| ITEM-001 | APPROVED → applied | — |
| ITEM-002 | APPROVED → applied (extended) | Folded in ITEM-INQ-003 size-name guardrail. |
| ITEM-003 | APPROVED → applied | — |
| ITEM-004 | APPROVED → applied | Merged with ITEM-005 into one Scope-section rewrite. |
| ITEM-005 | APPROVED (with caveat) → applied | Honoured the ITEM-INQ-001 caveat: no reloadability claim re-introduced. |
| ITEM-006 | APPROVED → applied | — |
| ITEM-007 | APPROVED → applied | Box-drawing connectors preserved. |
| ITEM-008 | APPROVED → applied | — |
| ITEM-009 | APPROVED → applied | — |
| ITEM-010 | APPROVED → applied | — |
| ITEM-011 | APPROVED → applied | — |
| ITEM-012 | REJECTED (out-of-ledger) → not applied | mkdocs.yml not editable; see Cross-Boundary Findings. |
| ITEM-INQ-001 | Out-of-scope (code defect) → not applied | Honoured as a doc guardrail; see Cross-Boundary Findings. |
| ITEM-INQ-002 | In-ledger independent finding → applied | docs/index.md (ledger). |
| ITEM-INQ-003 | In-ledger guardrail → applied via ITEM-002 | No standalone edit; folded into the memory note wording. |

## Cross-Boundary Findings (no fix applied — surface to maintainer)

- **ITEM-012 — mkdocs nav orphan pages (out-of-ledger).** `mkdocs.yml` nav
  (L71–78) lists only index/installation/usage/tags/cite/build_mkdocs, while
  `docs/dinov3.md` and `docs/dinov3_experimental.md` exist on disk as orphan
  pages. `docs/build_mkdocs.md` instructs contributors to run
  `mkdocs build --strict`, which fails on pages-not-in-nav. Recommend the
  maintainer add both pages under the nav group (e.g.
  `DINOv3 (experimental): dinov3.md`, `dinov3_experimental.md`). `mkdocs.yml` is
  not in the editable ledger this round.

- **ITEM-INQ-001 — flavour-D train→infer roundtrip is broken (CODE defect).**
  `DINOv3LinearProbe.export_classifier_only()` (dinov3_linear.py:175–189) emits
  `classifier.weight`/`classifier.bias` + metadata at top level with NO `meta`
  key, but the flavour-D branch of `_extract_state_dict` requires
  `isinstance(ckpt.get("meta"), dict)` (base_classifier.py:94–98). Loading the
  stage-2 `best_model.pt` via the inference loaders therefore hits
  `raise ValueError("Unrecognised checkpoint format")` (base_classifier.py:141–143).
  Out of doc-fix scope (source code). Doc guardrail honoured: ITEM-005 and
  ITEM-009 describe only what stage-2 *writes* and do NOT assert the repo's own
  output is reloadable through the inference path. Already tracked in
  docs/review/audit-fix/round_01 (ITEM-REV-001). Surface to maintainer.

## Skipped

- ITEM-012 (REJECTED, out-of-ledger) — recorded above, not edited.
- ITEM-INQ-001 (code defect, out-of-ledger) — recorded above, not edited.

## Re-proposed in Future Round

- None. All in-ledger approved items were applied this round. The two
  out-of-ledger items (ITEM-012 nav fix; ITEM-INQ-001 code defect) require a
  maintainer/code change outside the doc-fix editable scope and should be
  handled in their respective workflows, not re-proposed here.
