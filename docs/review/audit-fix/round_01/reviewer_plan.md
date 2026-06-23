# Reviewer change plan — round 1

Scope: behavioral defects only in the reviewer-owned DINOv3 files. I read the anti-drift protocol, scope ledger, coverage log, DINOv3 port plan, all owned files, and the five read-only subtask notes written under this round directory.

<a name="ITEM-REV-001"></a>
## ITEM-REV-001 | src/models/classification/dinov3_base/base_classifier.py:94 | Normalize and validate flavour-D classifier-only checkpoints | Plan §6 defines an authoritative `meta` block, while the repo's current stage-2 exporter emits the same metadata at top level. `_extract_state_dict()` should accept both layouts by normalizing top-level metadata into `meta`, then require `format == "dinov3_classifier_only"`, supported `version`, `model_name`, `embedding_dim`, `num_classes`, `image_size`, `feature_type == "cls"`, transform metadata when present, and ordered `class_names`; malformed classifier-only files must fail before model construction.

<a name="ITEM-REV-002"></a>
## ITEM-REV-002 | src/models/classification/dinov3_base/custom_weights.py:45 | Enforce requested checkpoint identity and metadata invariants | Plan §6 requires load-time validation that requested `model_size`/`model_name` matches the checkpoint and that metadata is authoritative. Add validation for `meta.model_size`, resolved `meta.model_name`, `embedding_dim`, `num_classes`, `image_size`, `feature_type`, transform name/version, and class-name order; add expected-* arguments so Amazon manifest construction can reject wrong artifacts explicitly rather than relying on later tensor-shape errors.

<a name="ITEM-REV-003"></a>
## ITEM-REV-003 | src/models/classification/dinov3_base/custom_weights.py:229 | Validate classifier tensor shape and wrap load failures as `ValueError` | Current validation assumes `classifier.weight` is rank-2 and does not check `classifier.bias` before `load_state_dict()`. Validate tensor presence/rank/shape for weight `(num_classes, hidden_size)` and bias `(num_classes,)`, then wrap state-dict load errors with checkpoint flavour/model context to satisfy the plan's explicit mismatch-error requirement.

<a name="ITEM-REV-004"></a>
## ITEM-REV-004 | src/models/classification/dinov3_base/custom_weights.py:133 | Normalize SINCHI PEFT metadata before `LoraConfig` | Plan §2.4 describes PEFT config using `alpha`, while PEFT expects `lora_alpha`. Accept `alpha` as an alias, reject unsupported `method`, require `r`, `lora_alpha`, and `target_modules`, and raise an actionable `ValueError` for invalid PEFT metadata before constructing the LoRA wrapper.

<a name="ITEM-REV-005"></a>
## ITEM-REV-005 | src/models/dinov3.py:85 | Make embedding extraction strict about CLS token semantics | Plan §10 requires CLS embeddings from `last_hidden_state[:, 0, :]`. Remove fallback to `pooler_output` or 2-D payloads in `_cls_embedding_from_output()`; accept only `last_hidden_state` or a tuple/list/dict containing a 3-D last-hidden-state tensor and raise a clear `TypeError` otherwise.

<a name="ITEM-REV-006"></a>
## ITEM-REV-006 | src/models/classification/dinov3_base/base_classifier.py:280 | Split embedding-only extraction from prediction+embedding inference and align saved pickle names | `extract_embeddings()` currently requires a classifier and computes logits. Implement an embedding-only loop that only calls `get_embeddings()` under `eval()`/`inference_mode()`. For saved records, use the same SINCHI-safe `<safe_name>.pkl` convention as the cached dataset and detect duplicate safe names with an explicit collision error or manifest strategy, instead of hash-suffixed filenames that do not round-trip.

<a name="ITEM-REV-007"></a>
## ITEM-REV-007 | src/models/classification/dinov3_base/base_classifier.py:158 | Make direct base-inference construction device-safe | The base constructor records `device` but does not move an injected `net`; direct `DINOv3BaseInference(net=cpu_net, device="cuda")` sends inputs to CUDA while the model stays on CPU. Normalize `self.device`, move `self.net`/`self.predictor` to it in `__init__`, and keep batch/single inference inputs on the model device.

<a name="ITEM-REV-008"></a>
## ITEM-REV-008 | src/models/dinov3_registry.py:43 | Resolve size keys and harden local/offline availability checks | `ensure_model_available("base")` currently calls Hugging Face with the raw string `base`. Resolve registry aliases at entry, raise the existing actionable size error before Hub calls for invalid values, normalize returned snapshot paths, and avoid reporting local-only success from a cache containing only `config.json` without weights/preprocessor files.

<a name="ITEM-REV-009"></a>
## ITEM-REV-009 | src/models/classification/dinov3_base/amazon.py:10 | Validate Amazon manifest artifacts and SHA256 before loading | Plan §7/§8 requires a versioned manifest with URL/path, sha256, HF id/revision, dim, num_classes, class order, and flavour. Add manifest-field validation, implement download/local-path resolution plus SHA256 verification for populated entries, and pass expected metadata/class order into `DINOv3CustomWeights` so wrong-size or wrong-order heads are rejected explicitly. Placeholder `None` entries should still raise an actionable unpublished-artifact error until real heads are supplied.

<a name="ITEM-REV-010"></a>
## ITEM-REV-010 | src/models/classification/dinov3_base/transforms.py:28 | Put RGB conversion inside `DINOv3Transform` | Plan §10 assigns RGB conversion to the transform itself. Convert PIL/array inputs to RGB inside `__call__` before resize/tensor/normalize so direct public use on grayscale or RGBA images produces a 3-channel `(3, 518, 518)` tensor instead of normalization shape errors.

## Needed tests outside reviewer ownership

- `tests/unit/test_dinov3_extract_state_dict.py`: malformed classifier-only metadata, top-level stage-2 export acceptance, class-name dict numeric ordering, classifier bias/shape errors, and PEFT `alpha` alias.
- `tests/unit/test_dinov3_registry.py`: `ensure_model_available("base")` resolves to the full HF repo id and local-only incomplete-cache failure is actionable.
- `tests/unit` or `tests/integration`: Amazon manifest hash mismatch and class-order mismatch rejection with a synthetic head.
- `tests/unit`: `DINOv3Transform` RGB/L/RGBA direct inputs and `DINOv3Classifier.get_embeddings()` strict CLS behavior.

STATUS: PLAN-READY
