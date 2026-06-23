# Auditor report — round 1 (APPLY phase, Step 2)

Scope: STRUCTURAL changes only (dead-code removal, deduplication, naming,
readability). No behavioral changes. All five approved items (ITEM-AUD-001..005)
were applied. Owned files only; the concurrent doc-fix dirty docs were left
untouched.

Commit: `c3460fb9fb5b1435505d7bc08b917d2c639939a5`
Files changed (committed via `git commit --only`):
- `src/algorithms/dinov3_linear.py`
- `scripts/extract_embeddings.py`

(The other five owned files had NO structural findings in the approved plan and
were left unchanged: `src/datasets/dinov3_embedding.py`, `scripts/train_dinov3.py`,
`src/models/classification/__init__.py`,
`src/models/classification/dinov3_base/__init__.py`, `configs/dinov3_linear.yaml`.)

---

## Changes Applied

<a name="ITEM-AUD-001"></a>
### ITEM-AUD-001 — src/algorithms/dinov3_linear.py:14 (import line)
- before: `from src.utils.mixup import efficient_batch_same_class_mixup, noisy_same_class_mixup`
- after:  `from src.utils.mixup import noisy_same_class_mixup`
- why: Dead-import removal. `efficient_batch_same_class_mixup` appeared only on the
  import line and was never referenced (verified by grep + AST). `noisy_same_class_mixup`
  is still used (training_step). Both symbols continue to exist in `src/utils/mixup.py`,
  so the trimmed import stays valid. No behavioral effect.

<a name="ITEM-AUD-002"></a>
### ITEM-AUD-002 — src/algorithms/dinov3_linear.py (config-access dedup)
- Added a module-level helper after `__all__`:
  ```python
  def _conf_get(obj: Any, key: str, default: Any = None) -> Any:
      return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)
  ```
- Collapsed the duplicated "mapping-or-namespace" access blocks:
  - (a) `dinov3_cfg = conf.get("dinov3", {}) if hasattr(conf, "get") else getattr(conf, "dinov3", {})`
        → `dinov3_cfg = _conf_get(conf, "dinov3", {})`
  - (a') the nested `_cfg` closure body (`if isinstance(dinov3_cfg, dict): ... else getattr`)
        → `return _conf_get(dinov3_cfg, key, default)` (closure retained as a thin wrapper to
        keep call sites `_cfg("model_name")` etc. unchanged)
  - (b) the `peft_enabled` if/else (`isinstance(peft_cfg, dict)` → `.get` else `getattr`)
        → `peft_enabled = _conf_get(peft_cfg, "enable", False)`
  - (c) the `lr` ternary in `configure_optimizers`
        (`getattr ... if hasattr ... else (.get if isinstance(dict) else 1e-3)`)
        → `lr = _conf_get(self.conf, "lr_classifier", 1e-3)`
- LEFT UNCHANGED: the required `num_classes` lookup (now line 72-74) still raises on a
  missing top-level `num_classes` (`conf["num_classes"]` / `getattr(conf, "num_classes")`
  with no default).
- why: The same dict-or-attr idiom appeared in 5 places; one helper centralizes it and
  removes branching. Semantics-preserving: `hasattr(obj,"get")` is True for `dict` and
  Munch (dict subclass) → `.get`; plain namespaces fall through to `getattr` with the
  same default. I proved equivalence vs the original branching with a standalone test
  across dict (present/missing), Munch (present/missing), and `SimpleNamespace`
  (present/missing) for keys `dinov3/peft/enable/freeze_backbone/model_name/use_mixup`
  and the `lr_classifier` path → `EQUIVALENCE-OK` for every case.

<a name="ITEM-AUD-003"></a>
### ITEM-AUD-003 — src/algorithms/dinov3_linear.py:1-10 (module docstring)
- before: `"""DINOv3 linear-probe Lightning algorithm (Phase 4 MVP). ... """` (described
  only the Phase-4 class)
- after: docstring now documents BOTH classes the module hosts — `DINOv3Linear`
  (image-based linear probe over a live backbone, Phase 4 MVP) and `DINOv3LinearProbe`
  (linear head trained on cached embeddings, Phase 5; the class used by
  `scripts/train_dinov3.py`).
- why: Readability/accuracy only. No code/behavior change.

<a name="ITEM-AUD-004"></a>
### ITEM-AUD-004 — scripts/extract_embeddings.py:9 (typing import)
- before: `from typing import Dict, Iterable, List, Optional`
- after:  `from typing import Dict, List, Optional`
- why: Dead-import removal. `Iterable` was never referenced (verified by grep + AST);
  `Dict/List/Optional` remain used.

<a name="ITEM-AUD-005"></a>
### ITEM-AUD-005 — scripts/extract_embeddings.py (payload simplification + numpy import)
- before:
  ```python
  import numpy as np
  ...
  "embedding": np.asarray(embedding, dtype=np.float32),
  ```
- after:
  ```python
  # (numpy import removed)
  ...
  "embedding": embedding,
  ```
- why: `embedding` is produced by `embeddings[idx].detach().cpu().to(dtype=torch.float32).numpy()`,
  i.e. an already-contiguous float32 `np.ndarray`; `np.asarray(x, dtype=np.float32)` on it is a
  provable no-op (returns the same object). Storing `embedding` directly is bit-identical, and
  the payload is pickled immediately inside the loop so the pre-existing tensor-buffer aliasing
  (which the no-op `asarray` never decoupled) is unchanged.
- Necessary corollary (non-behavioral): `np.asarray` was the ONLY `np.` usage in the file
  (verified: `.numpy()` on line 180 is a torch tensor method, not the `numpy` module), so
  `import numpy as np` became dead. Leaving it would contradict the dead-import cleanup of
  ITEM-AUD-004 in the same file, so the now-unused `import numpy as np` was removed. No other
  `numpy`/`np` references exist.

---

## Tests Added/Updated

- No new test files added (structural-only changes; the lead runs the full suite later).
- Sanity checks run this round:
  - `ast.parse` over all 6 owned `.py` files → `syntax-ok`.
  - Standalone equivalence test for ITEM-AUD-002's `_conf_get` vs the original
    branching across dict / Munch / `SimpleNamespace` (present + missing keys, incl. the
    `lr_classifier` ternary) → `EQUIVALENCE-OK`.

---

## Cross-Scope Findings (behavioral — deferred to round 2)

These were discovered during the audit but are BEHAVIORAL and out of structural scope.
They are recorded for the next round / lead.

- **C1 — scripts/extract_embeddings.py (safe_name collision silently reuses embeddings).**
  On a `safe_name` collision across nested dirs, the loop hits `if embedding_path.exists(): continue`
  and skips, silently reusing the FIRST image's embedding for a different image. Plan §9.1 requires
  a `manifest.csv` collision guard OR preserving directory structure. NOTE: this is the same-policy
  twin of reviewer ITEM-REV-006 (which fixes `base_classifier.extract_embeddings` this round);
  per cross-file note C1 both stage-1 save paths must converge on ONE `safe_name` + collision policy.
- **C2 — extract_embeddings.py:`_label_mapping` vs `build_label_mapping` (divergent label maps).**
  Stage-1 `_label_mapping` indexes from the FULL CSV (all splits); stage-2 `build_label_mapping`
  indexes from train only. The stored pickle `label` can therefore disagree with the CSV-derived
  label. Currently harmless because `DINOv3EmbeddingDataset.__getitem__` returns the CSV-derived
  `self.labels[idx]` and ignores the pickle `label`, but the stored label is misleading and the two
  near-identical implementations should be unified.
- **C3 — src/datasets/dinov3_embedding.py:91-94 (val-only label raises KeyError).**
  `DINOv3EmbeddingDataModule` / `train_dinov3.py` derive `label_to_idx` from train only and reuse it
  for val; a val-only label raises `KeyError` at dataset construction.

---

## Skipped

- None. All five approved auditor items (ITEM-AUD-001..005) were applicable and applied; no
  auditor items were REJECTED or MODIFY in the approvals.
- `src/datasets/dinov3_embedding.py`, `scripts/train_dinov3.py`,
  `src/models/classification/__init__.py`, `src/models/classification/dinov3_base/__init__.py`,
  and `configs/dinov3_linear.yaml` had no approved structural findings and were intentionally left
  unchanged (see auditor_plan.md "NO structural findings").

STATUS: DONE COMMIT=c3460fb9fb5b1435505d7bc08b917d2c639939a5
