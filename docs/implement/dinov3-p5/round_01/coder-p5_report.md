# coder-p5 report — round 01

<a name="p5-loader-v1-and-docs"></a>
## p5-loader-v1-and-docs

STATUS: DONE

### Files changed

1. `src/models/classification/dinov3_base/__init__.py` — new package re-export.
2. `src/models/classification/dinov3_base/base_classifier.py` — `DINOv3BaseInference` and v1 `_extract_state_dict`.
3. `src/models/classification/dinov3_base/custom_weights.py` — `DINOv3CustomWeights` MD-Cls `.ckpt` loader.
4. `src/models/classification/__init__.py` — guarded DINOv3 export and actionable `__getattr__`.
5. `docs/dinov3_experimental.md` — minimal experimental docs stub.
6. `tests/regression/test_dinov3_absent_actionable_error.py` — I13 absent-deps regression.
7. `tests/integration/test_dinov3_custom_weights_mdckpt.py` — MD-Cls checkpoint loader integration.
8. `tests/unit/test_extract_state_dict_v1.py` — v1 state-dict extraction unit tests.

### Commit

`60ac86db9042365cc0b1136c936190c31738bc60`

### SIGNATURE

`class DINOv3CustomWeights`

### Verification

- Package import: `OK` via `uv run python -c "from src.models.classification.dinov3_base import DINOv3CustomWeights, DINOv3BaseInference; print('OK')"`.
- I6/I7/I12 identity invariants: 3 passed (`test_plain_algorithm_class_identity`, `test_plain_resnet_class_identity`, `test_customweights_legacy_call_signature`).
- New unit + integration tests: 8 passed.
- Full default suite: 44 passed, 3 deselected, 1 warning.
- I13 isolated absent-deps regression: 1 passed.

### Notes

- `_extract_state_dict` v1 handles only `state_dict` entries with the `net.` prefix.
- SINCHI raw and PEFT checkpoint flavours raise `NotImplementedError` with `Phase 8` in the message.
- `DINOv3CustomWeights` uses `torch.load(..., weights_only=False)` and the docs state `.ckpt` files must come from trusted sources.

STATUS: DONE COMMIT=60ac86db9042365cc0b1136c936190c31738bc60 SIGNATURE="class DINOv3CustomWeights" ITEM=p5-loader-v1-and-docs
