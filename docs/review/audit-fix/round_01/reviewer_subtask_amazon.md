# Reviewer subtask — Amazon DINOv3 behavior (round 01)

## Scope read
- Read `~/.copilot/skills/_shared/iterative-anti-drift.md`; ledger convergence mode is `editing` and this subtask is read-only.
- Read `docs/review/audit-fix/SCOPE_LEDGER.json` and `docs/review/audit-fix/COVERAGE_LOG.jsonl`.
- Read `DINOv3_classification_port_plan.md`, especially §6, §8, §11, §12, and §9.4 because the Amazon wrapper is the planned round-trip inference surface.
- Focused files: `src/models/classification/dinov3_base/amazon.py`, direct construction call sites, and direct loader dependencies in `custom_weights.py`, `base_classifier.py`, and `dinov3_linear.py`.

## Direct construction calls checked
- `rg "DINOv3AmazonRainforest\\(" src scripts configs tests` found only:
  - `src/models/classification/dinov3_base/amazon.py:36` class definition.
  - `tests/integration/test_dinov3_amazon_placeholder.py:11` placeholder test constructing `DINOv3AmazonRainforest(model_size="base")` and expecting `NotImplementedError`.
- No production source call site currently exercises successful Amazon DINOv3 construction.

## Findings

### SUB-AMZ-001 — Amazon manifest path cannot ever auto-load, and SHA256 is never verified

**Evidence**
- `src/models/classification/dinov3_base/amazon.py:10-31` keeps every manifest `head_url` and `sha256` as `None` and has no `revision`, `num_classes`, or manifest-level class order.
- `src/models/classification/dinov3_base/amazon.py:86-94` raises `NotImplementedError` for default `weights=None` whenever `head_url` is `None`.
- `src/models/classification/dinov3_base/amazon.py:117-122` implements `_download_and_verify_head()` as an unconditional `NotImplementedError`; even a populated `head_url` would not download or hash-check anything.
- `tests/integration/test_dinov3_amazon_placeholder.py:23-24` asserts that all manifest `head_url` and `sha256` values are `None`, so the existing direct test locks in the placeholder behavior.
- Plan mismatch: `DINOv3_classification_port_plan.md:187-198` requires per-size URL/path, sha256, HF id + revision, dim, num_classes, class order, and sha256 verification before load; §8 says `DINOv3AmazonRainforest(... weights=None ...)` resolves manifest, downloads/verifies the head, builds the backbone, and loads (`DINOv3_classification_port_plan.md:213`). §12 expects a manifest sha256 guard test (`DINOv3_classification_port_plan.md:265-268`).

**Behavioral impact**
- Requirement 2 cannot succeed through the named product surface with default arguments. The class can only load caller-supplied local weights, and no built-in artifact integrity check exists.

**Proposed behavioral fix**
- Replace the placeholder dict with a versioned manifest entry per size containing `model_name`, `revision`, `embedding_dim`, `num_classes`, `class_names`, `flavor`, `head_url`/local artifact URI, and `sha256`.
- Implement `_download_and_verify_head()` to select the manifest entry, download/cache the head, compute SHA256 before `torch.load`, and raise a clear `ValueError`/`RuntimeError` on hash mismatch or missing metadata.
- Replace the placeholder integration test with tests for successful manifest selection using a synthetic local artifact and for hash mismatch rejection.

### SUB-AMZ-002 — Amazon class ordering is not validated against checkpoint metadata

**Evidence**
- `src/models/classification/dinov3_base/amazon.py:98-105` passes `class_names=list(self.CLASS_NAMES.values())` into `DINOv3CustomWeights` before load-time metadata validation.
- `src/models/classification/dinov3_base/custom_weights.py:112-124` uses explicit `class_names` ahead of `meta.get("class_names")` and only checks length, not equality or order.
- `src/models/classification/dinov3_base/amazon.py:106-111` then installs the same static class list on the inference wrapper.
- Plan mismatch: classifier-only metadata carries ordered `class_names` (`DINOv3_classification_port_plan.md:142-152`), load-time validation must carry class-name ordering (`DINOv3_classification_port_plan.md:155`), and the Amazon manifest is the source of the 36-class order (`DINOv3_classification_port_plan.md:187-198`).

**Behavioral impact**
- A checkpoint with the same 36 classes in a different order can load without error. Logits would then be decoded using the hard-coded Amazon order rather than the checkpoint's trained order, producing plausible but wrong species labels.

**Proposed behavioral fix**
- For `DINOv3AmazonRainforest`, treat the manifest/SINCHI order as `expected_class_names`, not as a blind override.
- If checkpoint metadata includes `class_names`, require exact sequence equality with the expected Amazon order and raise `ValueError` on mismatch.
- If metadata lacks `class_names`, allow only legacy SINCHI RAW checkpoints whose provenance is known to use the Amazon order, and keep an explicit validation path for `num_classes == len(expected_class_names)`.
- Add a unit test where a synthetic classifier-only checkpoint has reversed class names and verify Amazon construction rejects it.

### SUB-AMZ-003 — Requested `model_size` and manifest metadata are only partially validated

**Evidence**
- `src/models/classification/dinov3_base/amazon.py:98-105` supplies `model_size`, `model_name`, and `embedding_dim` from the selected manifest entry to `DINOv3CustomWeights`.
- `src/models/classification/dinov3_base/custom_weights.py:65-75` validates only requested/resolved `model_name` against `meta["model_name"]` when that field exists.
- `src/models/classification/dinov3_base/custom_weights.py:86-89` returns an explicit `embedding_dim` without checking whether `meta["embedding_dim"]` disagrees.
- No `custom_weights.py` path checks `meta["model_size"]`, `meta["image_size"]`, `meta["feature_type"]`, `meta["transform"]`, or manifest `revision` against the Amazon request.
- Plan mismatch: load-time validation must assert requested `model_size`/`model_name` matches checkpoint, `num_classes == classifier.weight.shape[0]`, and classifier input dim matches backbone hidden size (`DINOv3_classification_port_plan.md:155`); classifier-only metadata includes `model_size`, `embedding_dim`, `image_size`, `feature_type`, and transform fields (`DINOv3_classification_port_plan.md:142-152`).

**Behavioral impact**
- Some wrong-artifact cases are caught by `model_name` or tensor shape, but same-shape or incomplete metadata cases can bypass the Amazon manifest's selected size. That weakens artifact selection and makes failures depend on tensor shape rather than explicit metadata.

**Proposed behavioral fix**
- Add an Amazon-specific validation layer after `_extract_state_dict()` and before model construction, or extend `DINOv3CustomWeights` with `expected_model_size`, `expected_embedding_dim`, `expected_num_classes`, `expected_image_size`, `expected_transform`, and `expected_class_names`.
- Reject any classifier-only checkpoint whose metadata conflicts with the selected manifest entry, including `model_size`, resolved `model_name`, `embedding_dim`, `num_classes`, `image_size`, `feature_type`, transform name/version, and optional HF revision.
- Keep tensor shape validation as a second line of defense, not the primary artifact-selection check.

### SUB-AMZ-004 — The stage-2 classifier-only artifact cannot round-trip into the Amazon loader

**Evidence**
- Plan §9.4 requires the stage-2 classifier-only checkpoint to load through `DINOv3BaseInference`/`DINOv3AmazonRainforest` (`DINOv3_classification_port_plan.md:234-235`).
- `src/algorithms/dinov3_linear.py:175-189` exports `format`, `version`, `model_name`, `model_size`, metadata fields, and classifier tensors all at the top level.
- `src/models/classification/dinov3_base/base_classifier.py:94-112` recognizes classifier-only checkpoints only when top-level `classifier.weight`/`classifier.bias` are present and `ckpt.get("meta")` is a dict.
- `src/models/classification/dinov3_base/amazon.py:98-105` delegates local Amazon weight loading to `DINOv3CustomWeights`, which delegates format detection to `_extract_state_dict()`.

**Behavioral impact**
- The training script's own exported `best_model.pt` is not recognized as flavour D by the inference loader path that Amazon uses. The intended train-head → load-Amazon-wrapper round trip fails before construction.

**Proposed behavioral fix**
- Make `DINOv3LinearProbe.export_classifier_only()` emit `{"classifier.weight": ..., "classifier.bias": ..., "meta": {...}}` matching `_extract_state_dict()`; or update `_extract_state_dict()` to accept the current top-level export schema and normalize it into `meta`.
- Add a round-trip test with a synthetic exported head, mocked backbone, and `DINOv3AmazonRainforest(weights=..., model_size=...)` to verify successful load and metadata validation.

## Gated dependency check
- No Amazon-specific PEFT import leak found. `peft` is imported only inside `_wrap_peft_backbone()` (`src/models/classification/dinov3_base/custom_weights.py:133-142`) and invoked only for `meta["flavor"] == "sinchi_peft"` (`src/models/classification/dinov3_base/custom_weights.py:217-218`).
- `transformers` is imported lazily inside `DINOv3Classifier.__init__()` (`src/models/dinov3.py:40-47`), and `src.models.classification` has an optional-dependency guard (`src/models/classification/__init__.py:6-17`).

STATUS: DONE
