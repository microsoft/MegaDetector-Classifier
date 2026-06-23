# Reviewer report — round 2 APPLY

## Changes Applied

<a name="ITEM-REV-001"></a>
ITEM-REV-001 | scripts/extract_embeddings.py:61,184-228 | Added resume-skip cache verification, per-split safe-name collision detection, and `manifest.csv` emission with `image_path,pkl_path,label`. | Prevents two distinct image paths that collapse to the same `safe_name()` pickle from silently reusing the first cached embedding while preserving same-image resume skips.

<a name="ITEM-REV-003"></a>
ITEM-REV-003 | scripts/train_dinov3.py:25-27,52-84,111 | Built the stage-2 label mapping from the sorted train+val label union, passed that shared mapping to both datasets, derived `inferred_num_classes` and class names from it, and kept the `num_classes` override guard against the union size. | Prevents validation-only classes from crashing dataset construction and keeps cached labels, dataset labels, output dimension, and exported class order in one label universe.

<a name="ITEM-REV-005"></a>
ITEM-REV-005 | src/algorithms/dinov3_linear.py:181-194 | Made classifier-only export fail fast when `model_name` is missing and included `transform="dinov3_bicubic_518_imagenet"` plus `transform_version=1` in the exported metadata. | Prevents producing a declared flavour-D checkpoint that the loader rejects and carries the canonical transform identity through export.

## Tests

- `TMPDIR=/dev/shm/r2tmp uv run pytest -q --basetemp=/dev/shm/r2bt tests/unit/test_dinov3_embedding_dataset.py tests/unit/test_dinov3_registry.py tests/integration/test_dinov3_linear_probe.py tests/integration/test_dinov3_dataloader_emits_518.py tests/regression/test_invariants.py 2>&1 | tail -20`
  - Result: `19 passed, 2 deselected, 1 warning in 441.56s (0:07:21)`
- `TMPDIR=/dev/shm/r2tmp uv run python -m py_compile scripts/extract_embeddings.py scripts/train_dinov3.py src/algorithms/dinov3_linear.py`
  - Result: exit 0
- `git --no-pager diff --check -- scripts/extract_embeddings.py scripts/train_dinov3.py src/algorithms/dinov3_linear.py`
  - Result: exit 0

## Cross-Scope Findings

None.

## Skipped

- ITEM-REV-002 | REJECTED | Skipped the DataModule/shared-helper refactor because approvals rejected it as over-engineering on an unused path.
- ITEM-REV-004 | REJECTED | Skipped replacing stage-1 `_label_mapping` because approvals rejected it as dedup with regression risk to unlabeled extraction.

STATUS: DONE COMMIT=96e4a38ab9273e58c6bc5e0f6bff3561f5d5b0aa
