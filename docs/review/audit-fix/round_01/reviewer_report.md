# Reviewer report — round 1

## Changes Applied

<a name="ITEM-REV-001"></a>
ITEM-REV-001 | src/models/classification/dinov3_base/base_classifier.py:117-198,255-280 | Accepted top-level `dinov3_classifier_only` exports, normalized metadata, and validated version/shape/CLS/transform fields when present | The stage-2 exporter writes top-level metadata, so the loader now accepts the real flavour-D artifact while preserving nested-meta support.

<a name="ITEM-REV-002"></a>
ITEM-REV-002 | src/models/classification/dinov3_base/custom_weights.py:226-324 | Added conditional expected-* metadata validation for model size/name, shape-derived dimensions, image size, CLS feature type, transform, and class order | Validation now fires when metadata or caller expectations exist without rejecting RAW/PEFT checkpoints that lack optional flavour-D fields.

<a name="ITEM-REV-003"></a>
ITEM-REV-003 | src/models/classification/dinov3_base/base_classifier.py:56-82; src/models/classification/dinov3_base/custom_weights.py:362-388 | Validated classifier weight/bias presence, rank, row/column consistency, and wrapped load failures as `ValueError` with flavour/model context | Malformed heads now fail with explicit errors before obscure indexing or `load_state_dict` exceptions.

<a name="ITEM-REV-004"></a>
ITEM-REV-004 | src/models/classification/dinov3_base/custom_weights.py:164-192 | Normalized SINCHI PEFT config by accepting `alpha` as `lora_alpha`, rejecting non-LoRA methods, and requiring LoRA fields | Real PEFT checkpoints using `alpha` now reach `LoraConfig` with PEFT's expected keyword names.

<a name="ITEM-REV-005"></a>
ITEM-REV-005 | src/models/dinov3.py:85-107 | Removed pooler/2-D fallbacks and required a 3-D `last_hidden_state` tensor for CLS embeddings | Embeddings now match §10 parity: `last_hidden_state[:, 0, :]` only.

<a name="ITEM-REV-006"></a>
ITEM-REV-006 | src/models/classification/dinov3_base/base_classifier.py:463-606 | Replaced `extract_embeddings()` with an embedding-only loop and switched saved pickle paths to canonical `safe_name(image).pkl` with in-run collision detection | Stage-1 extraction no longer requires a classifier or computes logits, and saved names round-trip with `DINOv3EmbeddingDataset`.

<a name="ITEM-REV-007"></a>
ITEM-REV-007 | src/models/classification/dinov3_base/base_classifier.py:338-341 | Normalized `self.device` to `torch.device` and moved injected `net`/`predictor` during base inference construction | Direct construction with a non-CPU device no longer sends inputs to a different device than the model.

<a name="ITEM-REV-008"></a>
ITEM-REV-008 | src/models/dinov3_registry.py:53-108 | Resolved size aliases before Hub calls, normalized snapshot paths, and rejected incomplete local caches missing preprocessor or weights | `ensure_model_available("base")` now uses the full HF repo id and local-only mode no longer reports config-only caches as usable.

<a name="ITEM-REV-009"></a>
ITEM-REV-009 | src/models/classification/dinov3_base/amazon.py:56-232 | Added manifest fields, manifest validation, local/URL head resolution, SHA256 verification for published heads, and expected metadata handoff to `DINOv3CustomWeights` | Amazon loaders now validate artifact identity while preserving placeholder `None` errors and local user-supplied RAW checkpoint loading.

<a name="ITEM-REV-010"></a>
ITEM-REV-010 | src/models/classification/dinov3_base/transforms.py:30-35 | Converted PIL/array inputs to RGB inside `DINOv3Transform` before bicubic 518 resize and ImageNet normalization | Direct grayscale/RGBA transform use now yields 3-channel tensors without changing SINCHI preprocessing parity.

## Tests Added/Updated

- No tests edited; test files are outside reviewer ownership.
- Verification run:
  - `uv run python -m py_compile src/models/classification/dinov3_base/base_classifier.py src/models/classification/dinov3_base/custom_weights.py src/models/classification/dinov3_base/amazon.py src/models/classification/dinov3_base/transforms.py src/models/dinov3.py src/models/dinov3_registry.py` — PASS.
  - `git --no-pager diff --check -- <owned files>` — PASS.
  - `code-auditor` static post-diff audit — STATUS: OK.
- Requested targeted pytest command was attempted, but `uv run pytest ... | tail -25` produced no output after ~16 minutes; process inspection showed pytest blocked in kernel `rpc_wait_bit_killable` during torch import. A unit-only retry and a direct `uv run python -c 'import torch'` reproduced the same environment hang. No test failure output was produced.

## Cross-Scope Findings

- `tests/unit/test_dinov3_extract_state_dict.py`: add classifier-only top-level export acceptance, malformed classifier-only metadata, class-name numeric dict ordering, classifier bias/shape errors, and PEFT `alpha` alias cases.
- `tests/unit/test_dinov3_registry.py`: add alias resolution for `ensure_model_available("base")` and incomplete local-cache rejection.
- `tests/unit/test_dinov3_model.py`: add strict CLS extraction rejection for pooler-only and 2-D payloads.
- `tests/unit/test_dinov3_embedding_dataset.py` or a base-inference unit test: add canonical `safe_name(...).pkl` save and collision behavior coverage for `extract_embeddings()`.
- `tests/unit` or integration: add Amazon manifest SHA mismatch, local RAW path, and class-order mismatch cases.
- `tests/unit`: add `DINOv3Transform` direct grayscale/RGBA input coverage.

## Skipped

- No APPROVED reviewer item was skipped.
- No test files were edited because the APPLY ownership list only allowed reviewer source files.

STATUS: DONE COMMIT=c9fc011179d796854cf81b6b6ee89f82ac615b37
