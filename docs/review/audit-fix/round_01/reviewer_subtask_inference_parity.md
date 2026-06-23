# Reviewer subtask — DINOv3 inference parity

Scope: behavioral correctness only for `src/models/classification/dinov3_base/base_classifier.py` and `src/models/dinov3.py` against plan §8/§10/§12, with supporting cross-file evidence where needed.

## Findings

### SUB-INF-001 — `DINOv3BaseInference` does not provide the planned inference constructor/loading API

**Evidence**
- Plan §8 requires full API parity in `DINOv3BaseInference`: `__init__(weights, class_names, device, model_size|model_name, transform=None)` and `_load_model(...)` that builds the backbone via `ensure_model_available` and loads the head (`DINOv3_classification_port_plan.md:204-207`).
- The implementation constructor only accepts an already-built `net`, optional `transform`, `device`, and `class_names` (`src/models/classification/dinov3_base/base_classifier.py:158-164`).
- The class stores that injected net directly (`src/models/classification/dinov3_base/base_classifier.py:166-172`); there is no `def _load_model` in this file.

**Behavioral impact**
Existing callers cannot use the base inference class as the planned parity surface for local weights/model-size/model-name loading. The load path is pushed into `DINOv3CustomWeights`, so `DINOv3BaseInference` itself does not match the TIMM/ResNet-style base loader promised by the plan.

**Proposed behavioral fix**
Add the planned constructor signature and `_load_model(...)` to `DINOv3BaseInference`: resolve/download the HF backbone with `ensure_model_available`, call `_extract_state_dict(..., return_meta=True)`, construct `DINOv3Classifier`, validate classifier shape/class-name ordering/model identity, load the head/state, move to `device`, and call `eval()`. If keeping dependency injection for tests, make `net` an explicit test-only/advanced optional path that still normalizes device/eval state.

### SUB-INF-002 — Direct `DINOv3BaseInference(device=...)` can put inputs and model on different devices

**Evidence**
- The base constructor records `self.device` but does not move `self.net`/`self.predictor` to that device (`src/models/classification/dinov3_base/base_classifier.py:166-172`).
- Inference methods move image tensors to `self.device` before calling the net (`src/models/classification/dinov3_base/base_classifier.py:223-227`, `src/models/classification/dinov3_base/base_classifier.py:267-270`, `src/models/classification/dinov3_base/base_classifier.py:333-337`).
- `DINOv3CustomWeights` compensates after `super().__init__` by calling `self.to(device)` (`src/models/classification/dinov3_base/custom_weights.py:244-248`), but the base class itself remains unsafe for direct use.

**Behavioral impact**
A direct `DINOv3BaseInference(net=cpu_net, device="cuda")` will send inputs to CUDA while the net remains on CPU, causing device mismatch at inference. This is an inference API parity issue because the plan makes the base class the public inference surface.

**Proposed behavioral fix**
Normalize `self.device = torch.device(device)` and call `self.to(self.device)` or `self.net.to(self.device)` inside `DINOv3BaseInference.__init__` after assigning the net. In batch/single methods, move floating inputs with both device and model dtype when appropriate, e.g. derive `param_dtype = next(self.net.parameters()).dtype` and use `imgs.to(device=self.device, dtype=param_dtype)` for floating tensors.

### SUB-INF-003 — Embedding pickle names saved by inference are not compatible with the stage-2 cached-embedding dataset

**Evidence**
- Plan §9.1 requires SINCHI-compatible per-image pickles under split directories using the safe name and calls out collision handling via a manifest or preserved directory structure (`DINOv3_classification_port_plan.md:219-223`).
- The inference helper writes `<safe_name>.<sha1>.pkl` (`src/models/classification/dinov3_base/base_classifier.py:291-294`) and `batch_image_classification_with_embeddings` uses that helper when saving (`src/models/classification/dinov3_base/base_classifier.py:343-347`).
- The cached-embedding dataset expects exactly `<embeddings_dir>/<split>/<safe_name(Photo_ID)>.pkl` (`src/datasets/dinov3_embedding.py:45-47`, `src/datasets/dinov3_embedding.py:96-98`).

**Behavioral impact**
Embeddings saved through `batch_image_classification_with_embeddings(save_embeddings_dir=...)` cannot be consumed by `DINOv3EmbeddingDataset`/stage-2 training without renaming, even though the plan requires the stage-1/stage-2 embedding-save path to round-trip.

**Proposed behavioral fix**
Use the same `safe_name(image_path) + ".pkl"` naming convention as `DINOv3EmbeddingDataset`, and implement the planned collision strategy: write a manifest (`image_path,pkl_path,label`) and raise on duplicate safe names, or preserve relative directory structure. If hash-suffixed files are kept, update the dataset to resolve through the manifest rather than hard-coded safe-name paths.

### SUB-INF-004 — `extract_embeddings()` is not embedding-only; it requires a classifier and computes logits

**Evidence**
- Plan §8 defines `extract_embeddings(...)` as an embedding-only convenience wrapper used by stage 1 (`DINOv3_classification_port_plan.md:209-211`).
- Current `extract_embeddings()` delegates to `batch_image_classification_with_embeddings(return_embeddings=True)` (`src/models/classification/dinov3_base/base_classifier.py:361-379`).
- That path calls `_embeddings_and_logits`, which requires both `get_embeddings()` and `classifier`, then computes `self.net.classifier(embeddings)` (`src/models/classification/dinov3_base/base_classifier.py:280-288`, `src/models/classification/dinov3_base/base_classifier.py:333-337`).

**Behavioral impact**
Embedding-only extraction fails for a backbone-only/embedding-only wrapper and does unnecessary classifier work even when predictions are not requested. This diverges from the plan's stage-1 embedding-only API.

**Proposed behavioral fix**
Implement `extract_embeddings()` as its own loop that only calls `self.net.get_embeddings(imgs)` under `self.eval()` and `torch.inference_mode()`, saves/returns embedding records, and does not require `self.net.classifier`. Keep `_embeddings_and_logits()` for the prediction-plus-embedding API.

### SUB-INF-005 — `get_embeddings()` can silently return non-CLS `pooler_output`, weakening the required CLS-token semantics

**Evidence**
- The plan requires CLS embeddings and the review scope specifies `last_hidden_state[:, 0, :]` semantics (`DINOv3_classification_port_plan.md:209-211`, `DINOv3_classification_port_plan.md:219-222`).
- `forward()` uses the correct CLS token path (`src/models/dinov3.py:79-83`).
- `_cls_embedding_from_output()` also uses `last_hidden_state[:, 0, :]` when present, but falls back to `pooler_output`, dict `pooler_output`, or a 2-D tensor payload (`src/models/dinov3.py:86-103`). `get_embeddings()` uses this helper (`src/models/dinov3.py:105-114`).

**Behavioral impact**
The prediction path and embedding-save path can diverge for any backbone/output wrapper that does not expose `last_hidden_state` but does expose `pooler_output` or a 2-D tensor. Saved embeddings would no longer be the planned CLS token embeddings, yet no error would be raised.

**Proposed behavioral fix**
Make `_cls_embedding_from_output()` strict: accept `last_hidden_state[:, 0, :]` or a tuple/list whose first tensor is a 3-D last-hidden-state tensor, and raise a clear `TypeError` otherwise. Do not fall back to `pooler_output` for DINOv3 parity unless a golden parity test explicitly proves the output is identical.

### SUB-INF-006 — Classifier-only checkpoint construction/loading is inconsistent with the current stage-2 exporter

**Evidence**
- Plan §9.4 says the stage-2 `classifier_only` checkpoint must load through `DINOv3BaseInference`/Amazon for inference (`DINOv3_classification_port_plan.md:234-235`).
- `_extract_state_dict()` only recognizes classifier-only checkpoints when top-level `classifier.weight`/`classifier.bias` are present and `ckpt.get("meta")` is a dict (`src/models/classification/dinov3_base/base_classifier.py:94-112`).
- The current stage-2 exporter returns metadata fields at top level, not under `meta` (`src/algorithms/dinov3_linear.py:175-189`), and `scripts/train_dinov3.py` saves that dict directly (`scripts/train_dinov3.py:123-127`).

**Behavioral impact**
A classifier-only checkpoint produced by the repository's own stage-2 training path will not satisfy `_extract_state_dict()`'s flavour-D signature, so inference construction from the trained head can fail before model/head loading.

**Proposed behavioral fix**
Unify the flavour-D schema. Either change `DINOv3LinearProbe.export_classifier_only()` to save `{"classifier.weight": ..., "classifier.bias": ..., "meta": {...}}`, or extend `_extract_state_dict()` to also accept the current top-level metadata layout and normalize it into `meta`. Keep load-time validation for `embedding_dim`, `num_classes`, `feature_type == "cls"`, and class-name order.

## Checked without new finding

- Confidence output schema in `results_generation()` includes `img_id`, `prediction`, `class_id`, `confidence`, and per-row `all_confidences` (`src/models/classification/dinov3_base/base_classifier.py:186-211`), matching plan §8's schema requirement.
- Default DINOv3 transform selection in the base wrapper uses `DINOv3Transform(target_size=518)` (`src/models/classification/dinov3_base/base_classifier.py:176-181`); the transform itself uses bicubic resize, `ToTensor`, and ImageNet normalization (`src/models/classification/dinov3_base/transforms.py:17-29`), matching plan §10.
- Single/batch prediction paths call `self.eval()` and use `torch.inference_mode()` (`src/models/classification/dinov3_base/base_classifier.py:223-227`, `src/models/classification/dinov3_base/base_classifier.py:266-270`, `src/models/classification/dinov3_base/base_classifier.py:332-337`).

STATUS: DONE
