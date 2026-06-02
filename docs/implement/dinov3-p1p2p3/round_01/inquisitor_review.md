# Verifier review — DINOv3 P1+P2+P3 round 1

## Per-subtask verification

<a name="p1-bugfix"></a>
### Subtask: p1-bugfix (coder: coder-p1)

- Commit reachability: `19f87baa66879571695dace0a3412343b54047ee` is reachable from HEAD.
- Signature check: `num_layers: int = 50` appears in `src/models/classification/resnet_base/custom_weights.py`.
- Owned-file history from `impl-dinov3-p1p2p3-baseline..HEAD`:
  - `src/models/classification/resnet_base/custom_weights.py` — `19f87ba p1: CustomWeights num_layers kwarg + flip xfail (bug fix)`
  - `tests/regression/test_customweights_num_layers.py` — `19f87ba p1: CustomWeights num_layers kwarg + flip xfail (bug fix)`
- Test evidence: independent full suite passed; `test_customweights_loads_resnet18` is now PASSED.

<a name="p2-mixup"></a>
### Subtask: p2-mixup (coder: coder-p2)

- Commit reachability: `a332e19794fcde3284711cf6b5ae96c8a162c125` is reachable from HEAD.
- Signature check: `efficient_batch_same_class_mixup` appears in `src/utils/mixup.py`.
- Owned-file history from `impl-dinov3-p1p2p3-baseline..HEAD`:
  - `src/utils/mixup.py` — `a332e19 p2: port mixup utilities (noisy + efficient same-class) + 6 unit tests`
  - `tests/unit/test_mixup.py` — `a332e19 p2: port mixup utilities (noisy + efficient same-class) + 6 unit tests`
- Test evidence: independent full suite passed; all six `tests/unit/test_mixup.py` tests passed.

<a name="p3-deps-and-customcrop"></a>
### Subtask: p3-deps-and-customcrop (coder: coder-p3)

- Commit reachability: `2bd85389ab7806ea9ed646212df82a7c3fe03b10` is reachable from HEAD.
- Signature check: `_build_data_transforms` appears in `src/datasets/custom.py`.
- Owned-file history from `impl-dinov3-p1p2p3-baseline..HEAD`:
  - `requirements.txt` — `2bd8538 p3: optional deps file + Custom_Crop image_size kwarg + I11 hash update + 2 tests`
  - `requirements-dinov3.txt` — `2bd8538 p3: optional deps file + Custom_Crop image_size kwarg + I11 hash update + 2 tests`
  - `src/datasets/custom.py` — `2bd8538 p3: optional deps file + Custom_Crop image_size kwarg + I11 hash update + 2 tests`
  - `tests/regression/test_invariants.py` — `2bd8538 p3: optional deps file + Custom_Crop image_size kwarg + I11 hash update + 2 tests`
  - `tests/regression/test_custom_crop_default_is_224.py` — `2bd8538 p3: optional deps file + Custom_Crop image_size kwarg + I11 hash update + 2 tests`
  - `tests/integration/test_dinov3_dataloader_emits_518.py` — `2bd8538 p3: optional deps file + Custom_Crop image_size kwarg + I11 hash update + 2 tests`
- Test evidence: independent full suite passed; I11 and new 224/518 transform tests passed.

## I11 hash update

- Actual `src/datasets/custom.py` SHA: `57a108435edd4889fa06a29dbc019050ab808e0dff4a22981b6185e2f40d65ee`
- Literal `EXPECTED_CUSTOM_DATASET_SHA`: `57a108435edd4889fa06a29dbc019050ab808e0dff4a22981b6185e2f40d65ee`
- Result: MATCH.

## I8/I9/I10 unchanged

The invariant diff only changes `EXPECTED_CUSTOM_DATASET_SHA`:

```diff
-EXPECTED_CUSTOM_DATASET_SHA = "2e9888f9b3d60bc84d69269614edbf938f88614e4b77aaebce8e350e3c35e23b"
+EXPECTED_CUSTOM_DATASET_SHA = "57a108435edd4889fa06a29dbc019050ab808e0dff4a22981b6185e2f40d65ee"
```

No I8/I9/I10 SHA literals appear in the diff.

## P1 xfail flip

`grep -E 'xfail|test_customweights_loads_resnet18' tests/regression/test_customweights_num_layers.py` returns only:

```text
def test_customweights_loads_resnet18(tmp_path: Path) -> None:
```

The xfail marker is absent.

## P3 requirements.txt

`requirements.txt` diff has one added comment line and zero deletions:

```diff
+# DINOv3 training/inference: pip install -r requirements-dinov3.txt
```

## P2 __init__.py untouched

`git diff impl-dinov3-p1p2p3-baseline HEAD -- src/utils/__init__.py` is empty.

## Test re-run

Independent command:

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tee docs/implement/dinov3-p1p2p3/round_01/verifier-pytest.txt
```

Result: `26 passed, 2 deselected in 12.38s`.

Comparison with `verification.txt`: after excluding the lead-added header line and masking elapsed time, the pytest outputs match line-for-line. Raw diff differs only in elapsed time (`21.93s` vs `12.38s`).

## Fixed entries this round (must be ZERO for CONVERGED)

Round 1 has 3 fixed entries:

```jsonl
{"round":1,"agent":"coder-p1","file":"p1-bugfix","action":"fixed","commit":"19f87baa66879571695dace0a3412343b54047ee","signature":"num_layers: int = 50","report_anchor":"coder-p1_report.md#p1-bugfix"}
{"round":1,"agent":"coder-p2","file":"p2-mixup","action":"fixed","commit":"a332e19794fcde3284711cf6b5ae96c8a162c125","signature":"efficient_batch_same_class_mixup","report_anchor":"coder-p2_report.md#p2-mixup"}
{"round":1,"agent":"coder-p3","file":"p3-deps-and-customcrop","action":"fixed","commit":"2bd85389ab7806ea9ed646212df82a7c3fe03b10","signature":"_build_data_transforms","report_anchor":"coder-p3_report.md#p3-deps-and-customcrop"}
```

Because convergence mode is `editing`, round 1 cannot converge.

## Coverage analysis

- Ledger items: 3
- Verified this round: 3
- Uncovered: []

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=3/3 UNCOVERED=[] FIXED=3
