# Inquisitor independent review — doc-fix round 2 (CONFIRMING round)

Branch: `feat/dinov3-port`. HEAD: `9829e74`. Convergence mode: `editing`. I am
READ-ONLY on docs/code. This is the confirming (no-change) round that
editing-mode requires after round 1 applied fixes. The documenter re-scanned and
reported a no-change outcome (no changes proposed/applied this round; the
documenter `change_plan.md` records a nothing-to-do outcome, the
`documenter_report.md` records a done/no-commit outcome, and `verification.txt`
reports a clean link scan).

I did **not** simply trust round 1 or the documenter's self-report. I re-read the
current on-disk state of all 10 ledger docs and re-derived every code-coupled
claim from the actual source/config at this HEAD. Code/doc state has not changed
since the round-1 apply commit `3244405` (current HEAD `9829e74` only added
round-1 review artifacts), so a clean confirming round is expected — but I
verified it rather than assumed it.

## Verification basis (code/config re-read this round, independently)
- `scripts/extract_embeddings.py` — `IMAGE_SIZE=518` (L14); `DEFAULT_PHOTO_COL="Photo_ID"`,
  `DEFAULT_LABEL_COL="Label"` (L15–16); `--model-size choices=["small_plus","base","huge_plus"]`,
  `required=True` (L21); `--split choices=["train","val","both"]` (L25); `_split_column`
  accepts `split/Split/dataset/Dataset/subset/Subset` (L36–40). Stage-1 has **no**
  `--photo-col`/`--label-col` flag — defaults are hardwired.
- `scripts/train_dinov3.py` — stage-2 reads `photo_col`/`label_col` from the config
  (defaults `"Photo_ID"`/`"Label"`, L51–52); writes `<output_dir>/best_model.pt` via
  `torch.save(model.export_classifier_only(), …)` (L123–126).
- `configs/dinov3_linear.yaml` — every doc-listed key present (`model_size`, `embeddings_dir`,
  `train_csv`, `val_csv`, `output_dir`, `epochs`, `lr`, `batch_size`).
- `src/models/dinov3_registry.py` — `DINOV3_MODEL_IDS`/`DINOV3_DIMS` give
  `small_plus`→384/`vits16plus`, `base`→768/`vitb16`, `huge_plus`→1280/`vith16plus`
  (matches dinov3.md §4 exactly), plus two more sizes (`small`/`large`) not exposed to
  the two-stage CLI. `ensure_model_available` raises the gated-repo guidance the §7
  troubleshooting table describes (`GatedRepoError` → accept license + `hf auth login`/`HF_TOKEN`).
- `src/models/classification/dinov3_base/__init__.py` — `__all__` exports
  `DINOv3AmazonRainforest`, `DINOv3BaseInference`, `DINOv3CustomWeights`, `DINOv3Transform`.
- `src/models/classification/dinov3_base/amazon.py` — every manifest entry has
  `head_url=None`/`sha256=None` (TODO-Zenodo, L56–84); `DINOv3AmazonRainforest.__init__(self,
  model_size="base", weights=None, device="cpu")` (L185) calls
  `_validate_manifest(..., require_published=weights is None)` (L192), which raises
  `NotImplementedError` when unpublished and `weights is None` (L120–121). Confirms the
  README/dinov3.md `weights=<path>` caveat.
- `src/models/classification/dinov3_base/base_classifier.py` — inference surface
  `single_image_classification(self, img, img_id=None, id_strip=None)` (L386),
  `batch_image_classification(…)` (L417),
  `batch_image_classification_with_embeddings(self, data_path=None, det_results=None,
  save_embeddings_dir=None, …)` (L493) — signatures match dinov3.md §5 keyword usage.
  `_extract_state_dict` docstring enumerates "Supported structural flavours: A…D" (L230–237);
  A (md_cls), D (classifier-only), C (sinchi_peft), B (sinchi_raw) are all handled and return
  state (L240–306).
- `src/models/classification/__init__.py` — lazy DINOv3 surface gated on
  `transformers`+`huggingface_hub`; `__getattr__` raises an actionable `ImportError` listing the
  four DINOv3 names (L20–30).
- `src/algorithms/dinov3_linear.py` — `DINOv3Linear(pl.LightningModule).__init__(self, conf,
  train_class_counts=None, id_to_labels=None)` (L34–47, main.py-routable);
  `_wrap_with_peft` raises `NotImplementedError("PEFT lands in Phase 7")` (L114–116);
  `DINOv3LinearProbe` is the stage-2 cached-embedding class (L119+);
  `export_classifier_only` emits `"format": "dinov3_classifier_only"` (L181–184).
- `src/algorithms/__init__.py` — `from .dinov3_linear import *` (re-exports `DINOv3Linear`).
- `src/models/dinov3.py` — `DINOv3Classifier.__init__(..., freeze_backbone=True)`; passing
  `freeze_backbone=False` ⇒ `NotImplementedError("Full FT lands in Phase 6")` (L37–38).
- `src/datasets/custom.py` — `Custom_Crop.__init__(self, conf, image_size: int = 224)` (L267),
  confirming the `Custom_Crop(conf, image_size=518)` example.
- `main.py` — `algorithms.__dict__[conf.algorithm](conf=conf,
  train_class_counts=…, id_to_labels=…)` (L109–111) matches the `DINOv3Linear` signature exactly.
- `requirements-dinov3.txt` — `transformers>=4.56,<5`, `peft>=0.10.0,<0.14.0`,
  `huggingface_hub>=0.20,<1` (matches the docs' "`transformers>=4.56.0`" minimum + optional `peft`).

---

## Per-ledger-file verification

<a name="README-md"></a>
### README.md — OK (accurate, unchanged this round)
- `DINOv3AmazonRainforest` loader bullet (L142) resolves: class is `__all__`-exported by
  `dinov3_base/__init__.py` and re-exported via the lazy `classification` surface; the
  "needs `weights=<path>` until the Zenodo weights publish" caveat is enforced by
  `_validate_manifest(require_published=weights is None)`. Intra-doc anchor
  `#dinov3-classification-experimental` resolves to the heading at L207. Repo tree node
  `dinov3_base/` exists on disk (`amazon.py`, `base_classifier.py`, `custom_weights.py`,
  `transforms.py`, `__init__.py`). No drift. OK.

<a name="SECURITY-md"></a>
### SECURITY.md — OK (no drift)
- Boilerplate MSRC policy; no code-coupled or DINOv3 claims. Unaffected by any round-1/2
  code or doc change. OK.

<a name="docs-index-md"></a>
### docs/index.md — OK (accurate)
- One-line DINOv3 pointer (L45) links `[DINOv3 classification](dinov3.md)`; relative target
  `docs/dinov3.md` exists. Base-pipeline description (ResNet-18/50, three split strategies,
  `python main.py`) matches the repo. OK.

<a name="docs-installation-md"></a>
### docs/installation.md — OK (accurate)
- "Optional: DINOv3 classification" subsection (L64–72): `pip install -r requirements-dinov3.txt`
  (file exists with the three pinned deps), optional-deps note, and `dinov3.md` link (resolves).
  Consistent with the README/dinov3.md optional path. OK.

<a name="docs-usage-md"></a>
### docs/usage.md — OK (no drift)
- Documents the base `main.py` ResNet/timm workflow; no DINOv3 claims, so nothing can drift
  against the DINOv3 surfaces. The optional usage.md → dinov3.md pointer remains a LOW,
  non-blocking discoverability suggestion already recorded in round 1 as **NOT drift** — not a
  defect, and re-editing in a confirming round would needlessly re-open the editing loop. OK.

<a name="docs-tags-md"></a>
### docs/tags.md — OK (no drift)
- MkDocs tags-index stub (`[TAGS]` macro). No code-coupled claims. OK.

<a name="docs-cite-md"></a>
### docs/cite.md — OK (no drift)
- Citation page (PyTorch-Wildlife BibTeX + `citation.cff` pointer). No DINOv3/code-coupled
  claims. OK.

<a name="docs-build-mkdocs-md"></a>
### docs/build_mkdocs.md — OK as doc text (cross-boundary nav item out of ledger scope)
- Its own instructions are accurate (`mkdocs serve`, `mkdocs build --strict`, "Add the page to
  the `nav:` section in `mkdocs.yml`", `gh-deploy`). The `mkdocs.yml` orphan-nav matter (the two
  DINOv3 pages absent from `nav:`) is an out-of-ledger maintainer/code item per the task brief
  and does NOT block doc convergence; not counted as a doc issue. OK.

<a name="docs-dinov3-md"></a>
### docs/dinov3.md — OK (accurate)
- §2 install/deps: `transformers>=4.56.0`, `huggingface_hub`, conditional `peft` — matches
  `requirements-dinov3.txt`. §3 gated-access steps + the three model pages match the registry
  repo ids. §4 model-sizes table (`small_plus`/`base`/`huge_plus` → 384/768/1280, patch 16,
  image 518) matches `dinov3_registry.py` and `IMAGE_SIZE=518`. §5 inference example
  (`DINOv3AmazonRainforest(model_size="base", weights=…)` + the three classification methods)
  matches `amazon.py`/`base_classifier.py` signatures; the Zenodo placeholder table is correctly
  flagged TBD. §6 two-stage: stage-1 columns `Photo_ID`/`Label`, `--split` column names, stage-2
  output `<output_dir>/best_model.pt` (flavour-D classifier-only), and all config keys all verify
  against `extract_embeddings.py`/`train_dinov3.py`/`dinov3_linear.yaml`. §7 troubleshooting
  matches `ensure_model_available`'s gated/offline/cache handling. The §6 parenthetical
  "(the stage-2 config can override these via `photo_col`/`label_col`)" sits under the stage-1
  command, but the override is a **stage-2** config key — recorded in round 1 as a LOW, technically
  accurate, non-drift clarity note, not a defect. OK.

<a name="docs-dinov3-experimental-md"></a>
### docs/dinov3_experimental.md — OK (accurate)
- EXPERIMENTAL status line links `dinov3.md`; no stale `Phase N`/`MVP` status language remains in
  the doc body (grep-clean across the ledger). GPU-memory note correctly scopes ViT-S/B/L/H+ names
  to the `main.py`-routable `DINOv3Linear` path (registry accepts any HF repo id / 5 sizes) and
  points two-stage users to `small_plus`/`base`/`huge_plus`. `Custom_Crop(conf, image_size=518)`
  matches `custom.py:267`. Scope "Implemented today" list verifies: `DINOv3Classifier` (frozen) +
  `DINOv3Linear` (routable via `main.py`), the `DINOv3AmazonRainforest`/`DINOv3BaseInference`/
  `batch_image_classification_with_embeddings` surface, the flavour A–D loaders in
  `_extract_state_dict` (the code docstring itself enumerates A–D as the supported structural
  flavours; all four return state — the `NotImplementedError("…Phase 8")` at L309 guards a
  *different*, unsupported backbone-only SINCHI variant, not A–D), `DINOv3CustomWeights`
  (`.pt`/`.ckpt`, `torch.load(weights_only=False)` trusted-source caveat), the two-stage workflow,
  and the lazy optional-deps surface. "Still deferred" list verifies: `DINOv3Classifier(
  freeze_backbone=False)` and `DINOv3Linear._wrap_with_peft` both still raise. OK.

---

## Cross-path / bug-propagation audit
- Re-grepped all 10 ledger docs for residual stale phase/roadmap status language and forward
  references to non-existent tables → none in the doc bodies; the antipattern fixed in
  dinov3_experimental.md does not persist elsewhere. OK.
- Re-grepped the 5 non-DINOv3 docs (SECURITY/usage/tags/cite/build_mkdocs) for DINOv3 references →
  none; no stale parallel claims. OK.
- Independent relative `.md` link scan across all ledger docs → every relative target resolves
  (corroborates `verification.txt`): README→docs/dinov3.md + intra-doc anchor; index/installation→
  dinov3.md (+usage.md); dinov3.md↔dinov3_experimental.md. OK.
- The §5 keyword-call signatures and §4/registry dim+repo-id parity were re-derived from source
  this round (not carried over): consistent. OK.

## Missed issues
- I actively searched for drift the round-1 review or the documenter might have missed:
  signature mismatches (none — re-derived `single_image_classification`/`batch_image_classification`/
  `batch_image_classification_with_embeddings`, `DINOv3AmazonRainforest.__init__`,
  `DINOv3Linear.__init__`, `Custom_Crop.__init__`); registry dim/repo-id table accuracy (none);
  config-key completeness in `dinov3_linear.yaml` (none); the deferred-stub claims still raising
  (both confirmed); the flavour A–D claim against the actual `_extract_state_dict` control flow,
  including the Phase-8 `NotImplementedError` (confirmed it guards an out-of-set variant, so the
  "A–D implemented" claim is correct, not over-stated); dependency pins (match). **No previously
  missed in-scope doc defect was found.** Zero new doc issues.
- The two LOW non-blocking clarity suggestions from round 1 (usage.md → dinov3.md pointer; the
  dinov3.md §6 stage-1/stage-2 `photo_col`/`label_col` parenthetical placement) are explicitly
  **NOT drift** — the statements are accurate as written. They are optional polish, out of
  necessity scope, and correctly left unedited in this confirming round. They do not count as
  open doc defects.
- Out-of-ledger, non-blocking (per task brief; not doc issues): the `mkdocs.yml` DINOv3 nav-orphan
  item (maintainer/`mkdocs.yml` is not one of the 10 ledger files) and the flavour-D
  train→infer code matter (code/audit workflow, appears resolved in current code). Neither blocks
  doc convergence and neither is counted here.

## Coverage analysis
- Scope ledger lists 10 files; all 10 independently re-read and re-verified against current
  code/config this round. One COVERAGE_LOG.jsonl `verified` entry per file is appended for round 2,
  each pointing at the HTML anchor emitted before its heading above (anchor existence guarded with
  `grep -qF` before logging). `scope_check.sh` is run as the deterministic cross-check.
- Result: zero new doc issues, no changes applied this round, full 10/10 coverage, verification
  clean → editing-mode convergence criteria satisfied.

---

STATUS: CONVERGED SCOPE_CHECK=PASS COVERED=10/10 UNCOVERED=[]
