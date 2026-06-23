# Reviewer subtask: DINOv3 checkpoint flavors

Scope: behavioral correctness review of `src/models/classification/dinov3_base/base_classifier.py` and `src/models/classification/dinov3_base/custom_weights.py` against `DINOv3_classification_port_plan.md` §2.4 and §6. Read-only; no source edits.

## Findings

### SUB-CKPT-001 — PEFT config from SINCHI can be passed to `LoraConfig` with the wrong keyword

**Evidence**
- Plan §2.4 describes SINCHI PEFT metadata as `peft={lora, r=8, alpha=8, target_modules=[...]}` (`DINOv3_classification_port_plan.md:59`).
- Plan §6 requires flavor C to wrap the HF backbone with `LoraConfig(**config.peft)` (`DINOv3_classification_port_plan.md:139`).
- `_wrap_peft_backbone()` only removes `method`; it passes the remaining checkpoint metadata directly to `LoraConfig(**peft_kwargs)` (`src/models/classification/dinov3_base/custom_weights.py:144-148`).

**Behavioral impact**
If the verified SINCHI checkpoint stores LoRA scaling as `alpha` rather than PEFT's expected `lora_alpha`, flavor C loading fails before weights are loaded. The current unit test uses `lora_alpha`, so it does not exercise the plan-described metadata spelling.

**Proposed behavioral fix**
Normalize and validate SINCHI PEFT metadata before constructing `LoraConfig`: accept `alpha` as an alias for `lora_alpha`, reject unsupported `method` values, require `r`, `lora_alpha`, and `target_modules`, and raise an explicit `ValueError` naming the invalid metadata field.

### SUB-CKPT-002 — Classifier-only checkpoints are accepted without validating the authoritative format/version metadata

**Evidence**
- Plan §6 defines flavor D as classifier tensors plus an authoritative metadata block containing `format: "dinov3_classifier_only"`, `version: 1`, `model_name`, `model_size`, `embedding_dim`, `num_classes`, `image_size`, `feature_type`, transform metadata, and ordered `class_names` (`DINOv3_classification_port_plan.md:140-155`).
- `_extract_state_dict()` classifies any checkpoint with `classifier.weight`, `classifier.bias`, and any dict-valued `meta` as classifier-only; it does not check `meta["format"]`, `version`, or required fields (`src/models/classification/dinov3_base/base_classifier.py:94-112`).
- If class names are absent, `_normalise_class_names()` silently falls back to numeric strings (`src/models/classification/dinov3_base/custom_weights.py:112-115`).

**Behavioral impact**
A malformed flavor-D file can load if the caller supplies enough constructor arguments. That violates the plan's “metadata is authority” rule and can silently produce a model with numeric labels instead of the checkpoint's intended class order.

**Proposed behavioral fix**
For flavor D, require `meta.format == "dinov3_classifier_only"`, supported `version`, `model_name`, `embedding_dim`, `num_classes`, `image_size`, `feature_type == "cls"`, and ordered `class_names` unless an explicit caller override is intentionally allowed. Missing or invalid metadata should raise `ValueError` before model construction.

### SUB-CKPT-003 — Requested `model_size` is not validated against checkpoint `model_size`

**Evidence**
- Plan §6 requires load-time validation that the requested `model_size`/`model_name` matches the checkpoint (`DINOv3_classification_port_plan.md:155`).
- `_resolve_model_name()` validates only the resolved model name against `meta["model_name"]`; it never receives or checks `meta["model_size"]` (`src/models/classification/dinov3_base/custom_weights.py:45-76`, `src/models/classification/dinov3_base/custom_weights.py:178-182`).
- `_resolve_embedding_dim()` uses the caller's `model_size` to choose an embedding dimension before falling back to the checkpoint/model registry, without checking it against `meta["model_size"]` (`src/models/classification/dinov3_base/custom_weights.py:79-97`, `src/models/classification/dinov3_base/custom_weights.py:197-202`).

**Behavioral impact**
If a checkpoint has `model_size` metadata but missing/ambiguous `model_name`, or a caller passes conflicting `model_size` and explicit `model_name`, the loader may proceed until a later shape mismatch or may load the wrong 384-dimensional family member without an explicit model-size error.

**Proposed behavioral fix**
Validate `meta["model_size"]` whenever present: requested `model_size` must equal it, and explicit `model_name` must resolve to the same registry entry. Run this validation before choosing `embedding_dim` or constructing `DINOv3Classifier`.

### SUB-CKPT-004 — Classifier shape validation is partial and can surface non-ValueError failures

**Evidence**
- Plan §6 says load-time mismatches should produce explicit `ValueError`s (`DINOv3_classification_port_plan.md:155`).
- `DINOv3CustomWeights` checks only `classifier.weight` presence, `classifier.weight.shape[1]`, and `classifier.weight.shape[0]` (`src/models/classification/dinov3_base/custom_weights.py:229-242`).
- It does not validate that `classifier.weight` is 2-D, that `classifier.bias` exists, or that `classifier.bias.shape == (num_classes,)` before calling `self.net.load_state_dict(...)` (`src/models/classification/dinov3_base/custom_weights.py:244-245`).

**Behavioral impact**
Bad or corrupt checkpoint heads can raise `IndexError`, `AttributeError`, or raw `RuntimeError` from `load_state_dict` instead of the explicit `ValueError` required by the plan. For flavor D, `strict=False` also makes pre-load validation more important because missing unexpected keys are not treated the same way as full checkpoints.

**Proposed behavioral fix**
Before `load_state_dict`, validate tensor type/shape for `classifier.weight` and `classifier.bias`: weight must be rank-2 `(num_classes, hidden_size)`, bias must be rank-1 `(num_classes,)`. Wrap `load_state_dict` shape/key failures in `ValueError` with checkpoint flavor and model id context.

### SUB-CKPT-005 — Numeric-string `class_names` dictionaries can be reordered lexicographically

**Evidence**
- Plan §6 requires carrying class-name ordering at load time (`DINOv3_classification_port_plan.md:155`).
- `_normalise_class_names()` converts dicts with `[class_names[i] for i in sorted(class_names)]` (`src/models/classification/dinov3_base/custom_weights.py:116-118`).

**Behavioral impact**
If checkpoint metadata stores class names as a JSON-style dict with keys like `"0"`, `"1"`, ..., `"10"`, lexicographic sorting places `"10"` before `"2"`, corrupting the class index order while still passing the length check.

**Proposed behavioral fix**
When class-name metadata is a dict, accept integer keys or numeric-string keys and sort by numeric value. Reject sparse, duplicate-after-coercion, or non-numeric keys with an explicit `ValueError`. Prefer a list for classifier-only metadata.

## No source edits performed

This subtask wrote only this notes file under `docs/review/audit-fix/round_01/` and did not modify source files or commit anything.
