# Inquisitor approvals — round 1

Scope: independently verified each ITEM-001..012 against source on branch
`feat/dinov3-port`. Read in full: dinov3_registry.py, dinov3.py, dinov3_linear.py,
dinov3_embedding.py, classification/__init__.py, dinov3_base/{__init__,amazon,
base_classifier,custom_weights,transforms}.py, scripts/{extract_embeddings,
train_dinov3}.py, configs/dinov3_linear.yaml, requirements-dinov3.txt,
src/datasets/custom.py, and all 10 ledger docs + mkdocs.yml.

## Verdicts
ITEM-001 | APPROVED — docs/dinov3_experimental.md:1-5 status blockquote is stale/false. SINCHI checkpoint flavours are already implemented in base_classifier.py `_extract_state_dict` (md_cls L80-92, classifier_only L99-112, sinchi_peft L123-131, sinchi_raw L133-136), so "SINCHI ckpts in Phase 8" is wrong; docs/dinov3.md already exists, so "Full docs land in Phase 9" is wrong. Retitle + one-line status linking docs/dinov3.md is accurate and necessary.

ITEM-002 | APPROVED — Read docs/dinov3.md in full: it contains NO memory table. Line 35 "Full memory table lands in `docs/dinov3.md` (P9)" is an unfulfilled forward reference. Self-contained replacement (larger ViT backbones need proportionally more VRAM) is accurate.

ITEM-003 | APPROVED — Verified src/datasets/custom.py:267 `def __init__(self, conf, image_size: int = 224)`. The example `Custom_Crop(image_size=518)` omits the required positional `conf` and would TypeError; `Custom_Crop(conf, image_size=518)` is the correct call.

ITEM-004 | APPROVED — The "Not included (P8/P9)" list is false: flavours implemented (see ITEM-001 line refs); amazon.py, scripts/extract_embeddings.py, scripts/train_dinov3.py, configs/dinov3_linear.yaml all present. Still genuinely deferred, as the rewrite states: `DINOv3Classifier(freeze_backbone=False)` -> NotImplementedError("Full FT lands in Phase 6") (dinov3.py:38) and `DINOv3Linear._wrap_with_peft` -> NotImplementedError("PEFT lands in Phase 7") (dinov3_linear.py:110). Rewrite is accurate; pointing to docs/dinov3.md is correct.

ITEM-005 | APPROVED (with caveat) — Verified custom_weights.py:170 `torch.load(weights, ..., weights_only=False)` on a path; the trusted-source caveat is accurate. Broadening ".ckpt" -> ".pt/.ckpt" is accurate because DINOv3CustomWeights loads SINCHI raw/PEFT `.pt` (model_state_dict layout) and md_cls `.ckpt`. The new wording correctly DROPS "trained by this repo". CAVEAT to editors: do not re-introduce any claim that the repo's own stage-2 `best_model.pt` is reloadable via this path — that roundtrip is currently broken (see ITEM-INQ-001). As written, ITEM-005 does not assert it, so the edit is sound.

ITEM-006 | APPROVED — Verified DINOv3AmazonRainforest is star-exported from dinov3_base (`__all__`) and re-exported by src/models/classification/__init__.py when transformers+huggingface_hub are present; absent deps raise an actionable ImportError via `__getattr__` (mirrors the SpeciesNet optional pattern). amazon.py raises NotImplementedError when weights is None (manifest head_url/sha256 = None), so the "needs `weights=<path>`" caveat is required for accuracy. Anchor `#dinov3-classification-experimental` matches the heading at README.md:205.

ITEM-007 | APPROVED — Verified src/models/classification/dinov3_base/ exists; the README tree (README.md:174-176) lists resnet_base/timm_base/speciesnet_base but omits dinov3_base. Adding it is correct. Editor must preserve the tree's box-drawing connectors (the current `speciesnet_base/` uses the final-child connector).

ITEM-008 | APPROVED — Verified extract_embeddings.py DEFAULT_PHOTO_COL="Photo_ID"/DEFAULT_LABEL_COL="Label" (L16-17); ImageCSVDataset raises `KeyError` on a missing photo column (L111-112); `_split_column` accepts split/Split/dataset/Dataset/subset/Subset (L37-41). Confirmed the contrast: the base pipeline reads `path`/`classification`/`label` (custom.py:105-107). The override note is precise: photo_col/label_col are stage-2 CONFIG keys (train_dinov3.py:51-52), not stage-1 CLI flags (stage-1 hardwires the defaults at L159). Necessary to prevent a verbatim-copy KeyError.

ITEM-009 | APPROVED — Verified train_dinov3.py:125-126 writes `<output_dir>/best_model.pt` via `export_classifier_only()`; the listed config keys exist in configs/dinov3_linear.yaml. "classifier-only (flavour D)" matches the export `format: "dinov3_classifier_only"` (dinov3_linear.py:178). The proposed text only states what stage-2 WRITES and does not claim reloadability — correct (see ITEM-INQ-001).

ITEM-010 | APPROVED — Cross-link from docs/dinov3.md Overview to docs/dinov3_experimental.md is accurate (the experimental page holds the GPU-memory + main.py/DINOv3Linear notes) and reduces orphan isolation.

ITEM-011 | APPROVED — Verified requirements-dinov3.txt exists (`transformers>=4.56,<5`, `peft>=0.10.0,<0.14.0`, `huggingface_hub>=0.20,<1`) and README documents the same optional path; installation.md currently has no DINOv3 install path. Adding an "Optional: DINOv3 classification" subsection before "Next Steps" (installation.md:64-67) is consistent and complete.

ITEM-012 | REJECTED: out-of-ledger (surface to maintainer) — mkdocs.yml is not in the 10-doc editable ledger, so it cannot be edited this round. The underlying finding is REAL and corroborated: mkdocs.yml nav (L71-78) lists only index/installation/usage/tags/cite/build_mkdocs; docs/dinov3.md and docs/dinov3_experimental.md are orphan pages on disk, and build_mkdocs.md:33-36 instructs contributors to run `mkdocs build --strict` (warnings-as-errors), which fails on pages-not-in-nav. Surface to the lead/maintainer for a follow-up nav edit.

## Independent findings (issues the documenter missed)
ITEM-INQ-001 | MEDIUM — Flavour-D train->infer roundtrip is broken in current code, and the documenter did not flag it. `DINOv3LinearProbe.export_classifier_only()` (dinov3_linear.py:175-189) emits `classifier.weight`/`classifier.bias` plus metadata at TOP LEVEL with NO `meta` key, but the flavour-D branch of `_extract_state_dict` requires `isinstance(ckpt.get("meta"), dict)` (base_classifier.py:94-98). Loading the stage-2 `best_model.pt` via DINOv3CustomWeights/DINOv3AmazonRainforest therefore falls through to `raise ValueError("Unrecognised checkpoint format")` (base_classifier.py:141-143). This is a CODE defect, already noted in docs/review/audit-fix/round_01/reviewer_plan.md (ITEM-REV-001) and reviewer_subtask_amazon.md:77, and is OUT of the doc-fix editable scope — surface to maintainer. Doc guardrail this round: ITEM-005/009 must not assert the repo's own stage-2 output is reloadable through the inference loader; as written they do not, so no proposed text is blocked.

ITEM-INQ-002 | LOW — Discoverability: docs/index.md (Overview landing page) and docs/usage.md contain no pointer to the DINOv3 docs, and (per ITEM-012) the DINOv3 pages are absent from mkdocs nav. A one-line link from index.md -> docs/dinov3.md is within the editable ledger and would mitigate orphan isolation even before the out-of-scope nav fix. Optional enhancement, not drift — noted for editor consideration, not required.

ITEM-INQ-003 | LOW — Size-name surface mismatch: the two-stage CLI (extract_embeddings.py `--model-size`) and the Amazon manifest expose only {small_plus, base, huge_plus}, while dinov3_registry defines 5 sizes (adds `small`, `large`). docs/dinov3_experimental.md "GPU memory" references ViT-S/ViT-B/ViT-L. Those are valid for the main.py/DINOv3Linear route (accepts any HF repo id) but NOT for the two-stage scripts. ITEM-004's pointer to docs/dinov3.md mitigates; editors should ensure the experimental memory note is not read as the two-stage scripts accepting ViT-S/L.

STATUS: APPROVALS-DONE NEW=3
