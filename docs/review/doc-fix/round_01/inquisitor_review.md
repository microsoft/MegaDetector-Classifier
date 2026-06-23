# Inquisitor independent review — doc-fix round 1 (FINAL REVIEW / Phase 2)

Branch: `feat/dinov3-port`. HEAD: `3244405` (doc-fix round-1 apply commit).
Convergence mode: `editing`. I am READ-ONLY on docs/code; I re-read the **current
on-disk state** of all 10 ledger docs and re-verified every applied and
pre-existing claim against the actual source/config at this HEAD. The lead did
not pre-filter these inputs. Severity tags: CRITICAL / WARNING / SUGGESTION / OK.

## Changes reviewed (apply commit 3244405)
- README.md (+`DINOv3AmazonRainforest` loader bullet; +`dinov3_base/` tree node)
- docs/dinov3.md (+experimental cross-link; +stage-1 CSV columns; +stage-2 output/config keys)
- docs/dinov3_experimental.md (retitle; status rewrite; GPU-memory rewrite; `Custom_Crop(conf, …)`; Scope rewrite)
- docs/index.md (+DINOv3 pointer)
- docs/installation.md (+"Optional: DINOv3 classification" subsection)
- (5 ledger docs unchanged: SECURITY.md, docs/usage.md, docs/tags.md, docs/cite.md, docs/build_mkdocs.md)

## Verification basis (code/config re-read this round)
- `src/models/classification/dinov3_base/{__init__,amazon,base_classifier,custom_weights,transforms}.py`
- `src/models/classification/__init__.py`
- `src/models/dinov3.py`, `src/models/dinov3_registry.py`
- `src/algorithms/dinov3_linear.py`, `src/algorithms/__init__.py`
- `scripts/extract_embeddings.py`, `scripts/train_dinov3.py`
- `configs/dinov3_linear.yaml`, `requirements-dinov3.txt`
- `src/datasets/custom.py`, `main.py`

---

## Per-ledger-file verification

<a name="README-md"></a>
### README.md — OK (accurate)
- **ITEM-006** `DINOv3AmazonRainforest` loader bullet (README.md:142): VERIFIED. The class is
  star-exported by `dinov3_base/__init__.py` (`__all__`) and re-exported through
  `src/models/classification/__init__.py` only when `transformers`+`huggingface_hub`
  are importable; absent deps raise an actionable `ImportError` via `__getattr__`
  (mirrors the SpeciesNet optional pattern). `amazon.py` `DINOv3AmazonRainforest.__init__`
  calls `_validate_manifest(..., require_published=weights is None)`; every manifest
  `head_url`/`sha256` is `None` (TODO-Zenodo), so `weights=None` ⇒ `NotImplementedError`.
  The "needs `weights=<path>` until the Zenodo weights publish" caveat is required and correct.
- The intra-doc anchor `#dinov3-classification-experimental` resolves to the heading
  `## DINOv3 classification (experimental)` (README.md:207). OK.
- **ITEM-007** repo tree: `src/models/classification/dinov3_base/` exists on disk; the tree now
  lists it as the final child and demotes `speciesnet_base/` from `└──` to `├──`. Box-drawing
  remains valid. OK.

<a name="SECURITY-md"></a>
### SECURITY.md — OK (no drift)
- Unedited. Contains no DINOv3/phase references; nothing in the round-1 code or doc changes
  affects it. Consistent. OK.

<a name="docs-index-md"></a>
### docs/index.md — OK (accurate)
- **ITEM-INQ-002** one-line DINOv3 pointer added after the Get-Started section, linking
  `[DINOv3 classification](dinov3.md)`. Relative target `docs/dinov3.md` exists. The
  discoverability finding I raised in the approval phase is now resolved in-ledger. OK.

<a name="docs-installation-md"></a>
### docs/installation.md — OK (accurate)
- **ITEM-011** "Optional: DINOv3 classification" subsection added before "Next Steps":
  `pip install -r requirements-dinov3.txt` (file exists: `transformers>=4.56,<5`,
  `peft>=0.10.0,<0.14.0`, `huggingface_hub>=0.20,<1`), optional-deps note, and a link to
  `dinov3.md` (resolves). Consistent with the README optional path. OK.

<a name="docs-usage-md"></a>
### docs/usage.md — OK (no drift)
- Unedited. Documents the base `main.py` ResNet/timm workflow; contains no DINOv3 claims, so no
  drift against current code. SUGGESTION (LOW, non-blocking, NOT counted as drift): a one-line
  pointer to `dinov3.md` would mirror index.md and aid discoverability — same optional
  enhancement flagged under ITEM-INQ-002; out of necessity scope.

<a name="docs-tags-md"></a>
### docs/tags.md — OK (no drift)
- Unedited, MkDocs tags-index stub. No code-coupled claims. Consistent. OK.

<a name="docs-cite-md"></a>
### docs/cite.md — OK (no drift)
- Unedited citation page. No DINOv3/code-coupled claims. Consistent. OK.

<a name="docs-build-mkdocs-md"></a>
### docs/build_mkdocs.md — OK as doc text; cross-boundary carryover noted
- Unedited. Its own instructions are accurate (`mkdocs build --strict`, "Add the page to the
  `nav:` section in `mkdocs.yml`"). CARRYOVER (cross-boundary, NOT doc-editable this round):
  `mkdocs.yml` nav (L71–78) still lists only index/installation/usage/tags/cite/build_mkdocs;
  `docs/dinov3.md` and `docs/dinov3_experimental.md` remain orphan pages, so the `--strict`
  build this page mandates would still fail on pages-not-in-nav. `mkdocs.yml` is **not** one of
  the 10 ledger files, so it cannot be edited this round (ITEM-012). Surface to maintainer; this
  does not block doc convergence.

<a name="docs-dinov3-md"></a>
### docs/dinov3.md — OK (accurate)
- **ITEM-010** Overview cross-link to `dinov3_experimental.md` (resolves). OK.
- **ITEM-008** stage-1 CSV columns: VERIFIED `DEFAULT_PHOTO_COL="Photo_ID"`,
  `DEFAULT_LABEL_COL="Label"` (extract_embeddings.py:15-16); `ImageCSVDataset` raises
  `KeyError` on a missing photo column (L110-111); `_split_column` accepts
  `split/Split/dataset/Dataset/subset/Subset` (L37). The base-pipeline contrast
  (`path`/`classification`/`label`) matches `src/datasets/custom.py`. SUGGESTION (LOW,
  non-blocking): the parenthetical "the stage-2 config can override these via
  `photo_col`/`label_col`" sits under the stage-1 command, but the override exists only in the
  **stage-2** config (`train_dinov3.py:51-52`); **stage-1 has no `--photo-col`/`--label-col`
  flag** and hardwires the defaults. The sentence is technically true (these are stage-2 config
  keys) but could be read as if stage-1 honoured the override. A one-clause clarification would
  remove the ambiguity. Not drift — the statement is accurate as written.
- **ITEM-009** stage-2 output: VERIFIED `train_dinov3.py:125-126` writes
  `<output_dir>/best_model.pt` via `model.export_classifier_only()` (format
  `dinov3_classifier_only`, flavour D). All listed config keys (`model_size`,
  `embeddings_dir`, `train_csv`, `val_csv`, `output_dir`, `epochs`, `lr`, `batch_size`)
  exist in `configs/dinov3_linear.yaml`. The text states only what stage-2 *writes* and makes
  no reloadability claim (ITEM-INQ-001 guardrail honoured). OK.
- Pre-existing §4/§5 re-verified (ledger doc, so re-audited): model-sizes table
  (`small_plus`/`base`/`huge_plus` with dims 384/768/1280, repo ids) matches
  `dinov3_registry.py`; inference example methods exist with matching signatures —
  `single_image_classification(img, …)` (L386), `batch_image_classification(data_path=…)`
  (L417), `batch_image_classification_with_embeddings(data_path=…, save_embeddings_dir=…)`
  (L493). OK.

<a name="docs-dinov3-experimental-md"></a>
### docs/dinov3_experimental.md — OK (accurate)
- **ITEM-001** retitled `# DINOv3 fine-tuning notes (experimental)`; stale Phase-6/7/8/9 status
  blockquote replaced with a one-line EXPERIMENTAL status linking `dinov3.md`. The former
  claims were false (SINCHI flavours A–D already exist in `_extract_state_dict`; `dinov3.md`
  already exists). No `Phase N`/`P#`/`MVP` strings remain in any edited doc (grep-clean). OK.
- **ITEM-002 + ITEM-INQ-003** GPU-memory note: forward reference to a non-existent memory table
  removed; replaced with a self-contained VRAM statement plus the size-name guardrail —
  ViT-S/B/L/H+ names describe the **`main.py`-routable `DINOv3Linear`** path (accepts any HF
  repo id), while the two-stage scripts accept only `small_plus`/`base`/`huge_plus`. VERIFIED:
  `dinov3_registry.py` defines 5 sizes (small, small_plus, base, large, huge_plus);
  `extract_embeddings.py:21` restricts `--model-size` to `choices=["small_plus","base","huge_plus"]`.
  The discrepancy I raised (ITEM-INQ-003) is now defused in the text. OK.
- **ITEM-003** `Custom_Crop(conf, image_size=518)`: VERIFIED `src/datasets/custom.py:267`
  `def __init__(self, conf, image_size: int = 224)` — the old `Custom_Crop(image_size=518)`
  would `TypeError` on the missing positional `conf`. Fixed correctly. OK.
- **ITEM-004/005** Scope rewrite ("Implemented and supported today" vs "Still deferred"):
  VERIFIED the deferred stubs still raise — `DINOv3Classifier(freeze_backbone=False)` ⇒
  `NotImplementedError("Full FT lands in Phase 6")` (dinov3.py:37-38);
  `DINOv3Linear._wrap_with_peft` ⇒ `NotImplementedError("PEFT lands in Phase 7")`
  (dinov3_linear.py:114-116). The "load a local DINOv3 checkpoint (`.pt`/`.ckpt`)" wording with
  the trusted-source / `torch.load(weights_only=False)` caveat matches
  `custom_weights.py:220`. The `.ckpt`-only over-narrowing is gone, and the wording does NOT
  assert the repo's own stage-2 output is reloadable (ITEM-INQ-001 guardrail honoured). The
  "routable via `main.py`" claim is VERIFIED: `main.py:109` dispatches
  `algorithms.__dict__[conf.algorithm](conf=…, train_class_counts=…, id_to_labels=…)`,
  `algorithms/__init__.py` does `from .dinov3_linear import *`, `__all__` exports
  `DINOv3Linear`, and `DINOv3Linear.__init__(self, conf, train_class_counts=None,
  id_to_labels=None)` matches that call exactly. OK.

---

## Cross-path / bug-propagation audit
- Searched all 10 ledger docs for residual `Phase N` / `P#` / `lands in` / `MVP` strings →
  none remain (the antipattern fixed in dinov3_experimental.md does not persist elsewhere). OK.
- Searched the 5 unedited docs for DINOv3 references → none; no stale parallel claims to fix. OK.
- Relative `.md` links in edited docs all resolve (corroborates round-1 verification.txt): README→docs/dinov3.md
  and intra-doc `#dinov3-classification-experimental`; index→dinov3.md/usage.md; installation→dinov3.md;
  dinov3.md↔dinov3_experimental.md. OK.

## ITEM-INQ resolution status (my three approval-phase findings)
- **ITEM-INQ-002 (discoverability, LOW)** — RESOLVED in-ledger: index.md now links dinov3.md
  (usage.md pointer remains an optional, non-blocking enhancement).
- **ITEM-INQ-003 (size-name surface mismatch, LOW)** — RESOLVED in-ledger: the experimental
  GPU-memory note now scopes ViT-S/B/L/H+ to the `main.py`/`DINOv3Linear` route and points
  two-stage users to `small_plus`/`base`/`huge_plus`.
- **ITEM-INQ-001 (flavour-D train→infer roundtrip, MEDIUM — CODE, out-of-ledger)** — UPDATED:
  this defect now appears **RESOLVED in current code**. Commit `c9fc011` ("harden checkpoint
  validation", parent of the doc-apply commit) rewrote `_extract_state_dict`'s flavour-D
  detection: `has_classifier_only` now accepts `classifier_format == "dinov3_classifier_only"`
  (top-level, via `_classifier_only_meta` which merges top-level keys) and no longer *requires*
  a `meta` dict. Tracing `export_classifier_only()`'s exact output through the current
  dispatcher reaches the classifier-only branch and passes `_validate_classifier_only_meta`
  (declared=True; all required keys present; optional `transform`/`transform_version` skipped).
  (Static trace only — `torch` is not installed in this environment, so I could not execute the
  roundtrip; the logic is unambiguous on read.) Either way the docs remain conservative and
  correct: ITEM-005/009 describe only what stage-2 *writes* and assert no reloadability. The
  prior carryover is effectively CLOSED in code; no doc change is warranted.

## Carryover / cross-boundary (NOT doc-editable — do not block doc convergence)
- **ITEM-012 — mkdocs.yml nav orphan pages (OPEN, out-of-ledger).** `mkdocs.yml` is not a
  ledger file; the two DINOv3 pages remain absent from the nav, so the `mkdocs build --strict`
  that `docs/build_mkdocs.md` mandates would still fail. Recommend the maintainer add both pages
  under a nav group. Surface to maintainer.
- **ITEM-INQ-001 — flavour-D code defect (now RESOLVED in code; out-of-ledger).** See above;
  recorded for the code/audit workflow, not doc-fix.

## Code-quality / consistency observations (doc scope)
- No dead links, no contradictory parallel claims, no duplicated stale phase language across the
  ledger. Edited docs integrate cleanly with existing heading numbering. OK.

## Outcome
All 10 ledger docs independently re-verified and accurate against the current code/config. All
in-ledger drift items (ITEM-001..011) and my approval-phase findings (ITEM-INQ-002/003) are
applied and confirmed; ITEM-INQ-001 is a now-resolved code matter outside doc scope. No
in-scope doc defects remain open (two non-blocking LOW clarity suggestions noted on
docs/usage.md and docs/dinov3.md). Per editing-mode rules, because this round APPLIED changes it
cannot be declared final this round — a confirming round is required.

---

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=10/10 UNCOVERED=[] NEW=0
