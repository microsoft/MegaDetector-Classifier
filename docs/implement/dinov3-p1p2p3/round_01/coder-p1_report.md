# p1-bugfix

- Agent: coder-p1
- Scope item: p1-bugfix
- Commit: 19f87baa66879571695dace0a3412343b54047ee
- Signature: `num_layers: int = 50`
- Files changed:
  - `src/models/classification/resnet_base/custom_weights.py`
  - `tests/regression/test_customweights_num_layers.py`

## Changes

- Added defaulted `num_layers: int = 50` kwarg to `CustomWeights.__init__` and forwarded it to `PlainResNetInference`.
- Documented the new `num_layers` argument.
- Removed the strict xfail marker from the ResNet-18 regression test and removed the now-unused `pytest` import.

## Verification

- Baseline before fix: `uv run pytest tests/regression/test_customweights_num_layers.py -v 2>&1 | tail -15` reported `1 passed, 1 xfailed`.
- Focused after fix: `uv run pytest tests/regression/test_customweights_num_layers.py -v 2>&1 | tail -15` reported `2 passed`.
- Regression suite after fix: `uv run pytest tests/regression -v 2>&1 | tail -30` reported `17 passed, 2 deselected`.

STATUS: DONE COMMIT=19f87baa66879571695dace0a3412343b54047ee SIGNATURE="num_layers: int = 50" ITEM=p1-bugfix
