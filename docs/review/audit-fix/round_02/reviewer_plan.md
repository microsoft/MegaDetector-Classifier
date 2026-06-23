# Reviewer plan — round 2 (BEHAVIORAL, swapped ownership)

Scope: behavioral fixes only for the 7 files formerly owned by the round-1 auditor. Round 1 left three explicit behavioral cross-scope findings (C1-C3) in these files. I also checked the stage-2 exporter against the fixed flavour-D loader contract.

## Planned behavioral fixes

<a name="ITEM-REV-001"></a>
ITEM-REV-001 | scripts/extract_embeddings.py:154-187 | Add a per-split safe-name collision guard before/while writing cached embeddings: compute the target `<split>/<safe_name(Photo_ID)>.pkl`, raise `ValueError` when two different image paths map to the same pickle path, and when skipping an existing pickle verify its stored `image_path` matches the current row. Also write/update a `manifest.csv` with `image_path,pkl_path,label` for successful rows. | Fixes C1 and plan §9.1: current `if embedding_path.exists(): continue` silently reuses the first embedding on `safe_name` collisions. The guard preserves resumable same-image skips but rejects ambiguous collisions.

<a name="ITEM-REV-002"></a>
ITEM-REV-002 | src/datasets/dinov3_embedding.py:28-39,182-200 | Add a shared label-mapping helper that can build one stable mapping from one or more CSV/DataFrame inputs, make `build_label_mapping()` delegate to it, and make `DINOv3EmbeddingDataModule.setup()` derive its default `label_to_idx` from train+val labels instead of train only. | Fixes C3 for the DataModule path and supplies one mapping implementation for stage 1 and stage 2. Label-name leakage is not a feature leak; it prevents validation from crashing on a class absent from train but present in val.

<a name="ITEM-REV-003"></a>
ITEM-REV-003 | scripts/train_dinov3.py:54-78,99-108 | Build `label_to_idx` from the union of `train_csv` and `val_csv`, derive `inferred_num_classes`/`class_names` from that mapping, and pass the shared mapping to both datasets. Keep the existing `num_classes` override guard, but compare it against the union size. | Fixes C3 for the CLI path and resolves C2 for the documented train/val split workflow by using the same class-index universe for cached labels, dataset labels, model output size, and exported class order.

<a name="ITEM-REV-004"></a>
ITEM-REV-004 | scripts/extract_embeddings.py:43-58,132-137 | Replace the local `_label_mapping` implementation with the shared dataset helper and keep `_label_index` as the row-level lookup against that shared mapping. | Fixes the implementation half of C2: stage 1 and stage 2 should not maintain near-identical label-index builders with different semantics.

<a name="ITEM-REV-005"></a>
ITEM-REV-005 | src/algorithms/dinov3_linear.py:181-195 | Harden `DINOv3LinearProbe.export_classifier_only()` so it never emits a declared flavour-D checkpoint the loader rejects: raise `ValueError` if required exporter metadata such as `model_name` is missing, and include `transform="dinov3_bicubic_518_imagenet"` plus `transform_version=1` in the exported dict. | Confirms and tightens the exporter/loader contract. The fixed loader accepts top-level metadata and tolerates transform-less checkpoints, but plan §6 defines full classifier-only metadata and `_validate_classifier_only_meta(..., declared=True)` requires `model_name`; the exporter should produce a coherent round-trippable layout by construction.

## Audited with no behavioral source fix planned

- src/models/classification/__init__.py — guarded import still preserves §11/§12 optional-dep invariants; no behavioral change needed.
- src/models/classification/dinov3_base/__init__.py — pure re-export after the parent guard; no behavioral change needed.
- configs/dinov3_linear.yaml — current keys remain consistent with the planned train-script fixes; no config change needed.

STATUS: PLAN-READY
