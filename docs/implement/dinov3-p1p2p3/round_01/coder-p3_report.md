# coder-p3 report — round 01

<a id="p3-deps-and-customcrop"></a>
## p3-deps-and-customcrop

- Scope item: p3-deps-and-customcrop
- Commit: `2bd85389ab7806ea9ed646212df82a7c3fe03b10`
- Signature: `_build_data_transforms`
- New I11 hash: `57a108435edd4889fa06a29dbc019050ab808e0dff4a22981b6185e2f40d65ee`

## Changes

- Added `requirements-dinov3.txt` with optional DINOv3 dependency ranges only.
- Appended one comment line to `requirements.txt`; no runtime dependency lines changed.
- Added `_build_data_transforms(image_size)` and wired `Custom_Crop(image_size=518)` through instance transforms while preserving module-level `data_transforms` as the 224 fallback.
- Updated only `EXPECTED_CUSTOM_DATASET_SHA` in `tests/regression/test_invariants.py`.
- Added regression coverage for default 224 transforms and integration coverage for 518 transform output tensors.

## Verification

- `uv run python -c "import src.datasets.custom; print('import OK')"`: PASS.
- `uv run pytest tests/regression/test_invariants.py::test_requirements_runtime_lines_hash -v 2>&1 | tail -10`: PASS (I2 unchanged after requirements comment).
- `uv run pytest tests/regression/test_resnet_baseline_predictions.py -v 2>&1 | tail -10`: PASS (6 passed).
- `uv run pytest tests/regression/test_invariants.py -v 2>&1 | tail -15`: PASS (9 passed, 2 isolated_env deselected; I8/I9/I10 unchanged, I11 updated).
- `uv run pytest tests/regression/test_custom_crop_default_is_224.py -v 2>&1 | tail -20`: PASS (2 passed).
- `uv run pytest tests/integration/test_dinov3_dataloader_emits_518.py -v 2>&1 | tail -20`: PASS (1 passed).
- `uv run pytest tests/ -v 2>&1 | tail -30`: PASS (19 passed, 2 deselected, 1 pre-existing xfailed).

STATUS: DONE COMMIT=2bd85389ab7806ea9ed646212df82a7c3fe03b10 SIGNATURE="_build_data_transforms" ITEM=p3-deps-and-customcrop
