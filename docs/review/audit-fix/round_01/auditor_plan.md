# Auditor change plan — round 1 (STRUCTURAL only)

Scope: simplification, dead-code removal, deduplication (3+ occurrences), naming,
readability. NO behavioral changes. Behavioral issues are deferred to "Cross-Scope
Findings (deferred)" below and recorded for round N+1 in the Step-2 report.

Owned files audited (7):
- src/datasets/dinov3_embedding.py
- src/algorithms/dinov3_linear.py
- scripts/extract_embeddings.py
- scripts/train_dinov3.py
- src/models/classification/__init__.py
- src/models/classification/dinov3_base/__init__.py
- configs/dinov3_linear.yaml

---

## Proposed structural changes

<a name="ITEM-AUD-001"></a>
ITEM-AUD-001 | src/algorithms/dinov3_linear.py:14 | Remove the unused import `efficient_batch_same_class_mixup` from `from src.utils.mixup import ...` (keep `noisy_same_class_mixup`). | Dead code: only `noisy_same_class_mixup` is referenced (line 80); `efficient_batch_same_class_mixup` is never used (verified by grep + AST scan).

<a name="ITEM-AUD-002"></a>
ITEM-AUD-002 | src/algorithms/dinov3_linear.py:39,46-52,99-106 | Introduce a module-level helper `def _conf_get(obj, key, default=None): return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)` and use it to collapse the three duplicated "mapping-or-namespace" access blocks: (a) line 39 `dinov3_cfg = ...`; (b) the `peft_enabled` if/else at lines 47-50 → `peft_enabled = _conf_get(peft_cfg, "enable", False)`; (c) the `lr` ternary at lines 101-105 in `configure_optimizers` → `lr = _conf_get(self.conf, "lr_classifier", 1e-3)`. Also re-express the nested `_cfg` closure (lines 41-44) in terms of `_conf_get`. LEAVE the required `num_classes` lookup at line 64 unchanged (it must raise on missing). | Deduplication: the same dict-or-attr idiom appears in 5 places; a single helper removes ~9 lines and centralizes the pattern. Semantics preserved: `hasattr(obj,"get")` is True for both `dict` and Munch (dict subclass) → uses `.get`; plain namespaces fall through to `getattr` with default — identical behavior for all realistic config types. Existing tests exercise dict configs only and remain green.

<a name="ITEM-AUD-003"></a>
ITEM-AUD-003 | src/algorithms/dinov3_linear.py:1-4 | Update the module docstring to describe BOTH classes the module now hosts: `DINOv3Linear` (image-based linear probe over a live backbone, Phase 4) and `DINOv3LinearProbe` (linear head trained on cached embeddings, Phase 5). | Readability/accuracy: the docstring still says only "(Phase 4 MVP)" and omits `DINOv3LinearProbe`, which is the class actually used by `scripts/train_dinov3.py`. LOW priority.

<a name="ITEM-AUD-004"></a>
ITEM-AUD-004 | scripts/extract_embeddings.py:9 | Remove the unused `Iterable` name from `from typing import Dict, Iterable, List, Optional`. | Dead code: `Iterable` is never referenced (verified by grep + AST scan).

<a name="ITEM-AUD-005"></a>
ITEM-AUD-005 | scripts/extract_embeddings.py:181-183 | Simplify the payload embedding assignment: drop the redundant `np.asarray(embedding, dtype=np.float32)` wrapper and store `embedding` directly. | Simplification: `embedding` is produced by `embeddings[idx].detach().cpu().to(dtype=torch.float32).numpy()`, i.e. it is already a contiguous float32 `np.ndarray`; `np.asarray(x, dtype=np.float32)` on it is a provable no-op (returns the same array). Behavior identical. LOW priority.

---

<a name="NO-FINDINGS"></a>
## Owned files with NO structural findings (audited, no change proposed)

- **src/datasets/dinov3_embedding.py** — clean; imports all used; `DINOv3EmbeddingDataModule` is unused by the scripts but is a planned public-API deliverable (plan §5 "DINOv3EmbeddingDataset (+ DataModule)"), so it is intentionally retained (NOT dead code to remove).
- **scripts/train_dinov3.py** — clean; imports all used; control flow readable.
- **src/models/classification/__init__.py** — guarded-import split (plan §11) is correct and readable; left as-is.
- **src/models/classification/dinov3_base/__init__.py** — pure re-export module; the "unused import" AST hits are intentional `__all__` re-exports.
- **configs/dinov3_linear.yaml** — consistent with `train_dinov3.py` keys and §10 image_size=518.

---

## Cross-Scope Findings (deferred — behavioral, NOT fixed this round)

These are recorded here for visibility and will be carried into the Step-2 report's
"Cross-Scope Findings" section for round N+1. They are BEHAVIORAL and out of structural scope.

- **C1 (extract_embeddings.py:177-180) safe_name collision silently reuses embeddings.** Plan §9.1 explicitly warns that `safe_name` (`/`,`\`→`_`) can collide across nested dirs and requires a `manifest.csv` collision guard OR preserving directory structure. Current code does neither: on a name collision it hits `if embedding_path.exists(): continue` and skips, silently reusing the FIRST image's embedding for a different image. Behavioral gap vs plan.
- **C2 (extract_embeddings.py:44-48 vs build_label_mapping) divergent label maps.** Stage-1 `_label_mapping` builds indices from the FULL CSV (all splits); stage-2 `build_label_mapping` builds from train_csv only. The stored pickle `label` can therefore disagree with the CSV-derived label. Currently harmless because `DINOv3EmbeddingDataset.__getitem__` returns `self.labels[idx]` (CSV-derived) and ignores the pickle `label`, but the stored label is misleading and the two implementations duplicate near-identical logic.
- **C3 (dinov3_embedding.py:91-94) val labels absent from train mapping raise KeyError.** `DINOv3EmbeddingDataModule`/`train_dinov3.py` derive `label_to_idx` from train only and reuse it for val; a val-only label raises `KeyError` at dataset construction.

STATUS: PLAN-READY
