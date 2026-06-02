# coder-5-fixtures-xfail report

<a name="p0-fixtures-xfail"></a>
## p0-fixtures-xfail

Status: implemented and committed in `803cc5948f7267561d8f6d0138346cdcc7716fe4`.

Files created:
- `tests/fixtures/generate_peft_inputs.py`
- `tests/fixtures/peft_disabled_input.pt`
- `tests/fixtures/peft_disabled_target.pt`
- `tests/regression/test_customweights_num_layers.py`

`tests/fixtures/peft_disabled_golden.pt` is intentionally deferred to P7 because it depends on `DINOv3Linear`, which lands in P4. This P0 subtask only ships the deterministic input/target tensors and the regen script.

Loader confirmation:
- `PlainResNetInference.__init__` loads `clf_weights["state_dict"]` immediately with `strict=True`.
- Because `PlainResNetInference` owns a `net` submodule, synthetic checkpoint keys use `net.feature.*` and `net.classifier.*`.

Fixture verification:
- Generated twice with `uv run --with torch python tests/fixtures/generate_peft_inputs.py`.
- Hashes were byte-identical across both runs:
  - `peft_disabled_input.pt`: `76b27b5b6f9f38a8b1f082f4b2191ff22a3a45c793a590185a2b4255b665cbab`
  - `peft_disabled_target.pt`: `f275fe1de2f37249cd509c44e2625fc627ad114992e68c0e28d5147ccf443021`
- Loaded shapes/dtypes: input `(4, 3, 518, 518)` `torch.float32`; target `(4,)` `torch.int64` with values in `[0, 4)`.

Pytest verification:

```text
tests/regression/test_customweights_num_layers.py::test_customweights_loads_resnet18 XFAIL [ 50%]
tests/regression/test_customweights_num_layers.py::test_customweights_loads_resnet50 PASSED [100%]

======================== 1 passed, 1 xfailed in 37.33s =========================
```

Note: this isolated worktree has no committed pytest/torch dependency configuration yet, so verification used explicit `uv run --with ...` dependencies. Plain `uv run python tests/fixtures/generate_peft_inputs.py` failed before dependency resolution with `ModuleNotFoundError: No module named 'torch'`.

STATUS: DONE COMMIT=803cc5948f7267561d8f6d0138346cdcc7716fe4 SIGNATURE="test_customweights_loads_resnet18" ITEM=p0-fixtures-xfail
