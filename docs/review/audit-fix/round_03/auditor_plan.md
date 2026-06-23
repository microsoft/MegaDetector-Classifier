# Auditor plan — round 3 (STRUCTURAL, CONFIRMING)

Role: Auditor (STRUCTURAL). Round 3 is the convergence-confirming round: rounds 1-2
applied 18 fixes; the inquisitor verified 13/13 both rounds (NEW=0) and all 56 tests
pass. Editing-mode convergence requires one zero-edit round. I re-read all 7 owned
files and re-ran the structural checks (dead code, needless duplication, naming/
readability). Edited nothing — this is a planning scan.

## Owned files re-scanned this round
- src/datasets/dinov3_embedding.py
- src/algorithms/dinov3_linear.py
- scripts/extract_embeddings.py
- scripts/train_dinov3.py
- src/models/classification/__init__.py
- src/models/classification/dinov3_base/__init__.py
- configs/dinov3_linear.yaml

## Objective checks run
- AST unused-import + syntax scan over all 6 owned .py files: the only flags are
  `from __future__ import annotations` (a required future-import, not dead) and the
  `dinov3_base/__init__.py` re-exports (tracked via `__all__`, intentional). No genuine
  unused imports; all files compile.
- Reference scan: every module-private helper is referenced —
  `_class_names_from_label_mapping` (train_dinov3.py:25→111), `_label_mapping`
  (extract_embeddings.py:43→152), `build_label_mapping` (def + 2 call sites). No orphans.
- Config/script key contract: every `config[...]` required key (embeddings_dir,
  train_csv, val_csv) is present in configs/dinov3_linear.yaml; all `config.get(...)`
  keys carry defaults. Coherent.
- `git status --porcelain`: no modified tracked source (only untracked review artifacts).

## Structural candidates re-evaluated and DECLINED (prior-round rejections, not re-litigated)
- `DINOv3EmbeddingDataModule` unreferenced (dinov3_embedding.py:156 + `__all__`:19 only) —
  this is ITEM-REV-002 (REJECTED R2): an intentional public-API helper, exported on
  purpose. Not dead code to remove. DECLINE.
- `_label_mapping` (stage-1) vs `build_label_mapping` (stage-2) divergence — ITEM-REV-004
  (REJECTED R2): the `{}`-on-missing-Label vs RAISE behaviour is intentional (§9.1
  unlabeled extraction → label=-1). Deduping would risk a behavioral regression. DECLINE.
- Inline train+val union loop (train_dinov3.py:54-60) — the approved ITEM-REV-003 R2 fix;
  explicit, deterministic, and clear. Rewriting it is a marginal readability churn that
  re-litigates an approved fix. DECLINE.

## Other structural checks (all clean)
- Dead code: none — every function/class/helper in the 7 owned files is reachable.
- Needless duplication: the two near-parallel class-name builders operate on different
  inputs (a `label_to_idx` mapping vs a dataset's `idx_to_label`); coupling them is below
  the dedup bar and was not deferred to me. None actionable.
- Naming/readability: helper and variable names are descriptive and consistent; docstrings
  present on public surfaces. No defects.
- configs/dinov3_linear.yaml: keys consistent with the train script; `num_classes: null`
  → inferred from the union mapping. Clean.

<a name="NO-FINDINGS"></a>
## Conclusion
All 7 owned files are structurally clean. No genuine, justified structural defect remains.
The only candidates are prior-round intentional rejections, which this confirming round
must not re-litigate. Reporting NOTHING-TO-DO (zero source edits).

STATUS: NOTHING-TO-DO
