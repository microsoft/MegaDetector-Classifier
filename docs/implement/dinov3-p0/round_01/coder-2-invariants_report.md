# coder-2-invariants report — round 01

<a name="p0-invariants"></a>

## Scope item
- scope_item: p0-invariants
- Owned source file: tests/regression/test_invariants.py
- Phase: 0 only

## Invariant tests created
- I1: test_default_env_imports_clean (@pytest.mark.isolated_env)
- I2: test_requirements_runtime_lines_hash
- I4: test_default_config_yaml_equal
- I5: test_customweights_no_dinov3_deps (@pytest.mark.isolated_env)
- I6: test_plain_algorithm_class_identity
- I7: test_plain_resnet_class_identity
- I8: test_main_unchanged
- I9: test_plain_py_unchanged
- I10: test_plain_resnet_py_unchanged
- I11: test_custom_dataset_byte_or_behaviour
- I12: test_customweights_legacy_call_signature

I3 is not implemented here; it is owned by p0-baseline. I13 is not implemented here; it lands in Phase 5.

## Commit
- Final commit: 5dcfac0fa495d37a38d81553bcd1e2d7014d8dae
- SIGNATURE phrase: test_main_unchanged

## Embedded baselines
- requirements.txt runtime SHA-256: f00b06e6ab206fa5f603305211a267267a11d71b153cb1e854a8a104a0001f3e
- requirements.txt runtime lines: ['PytorchWildlife', 'scikit_learn', 'lightning', 'munch', 'typer', 'timm', 'wget', 'supervision', 'Pillow', 'tqdm']
- main.py SHA-256: 6c2b0225f04ae2527b34c9d41323465805ede6c221a240d0a94b7d397ec46dc7
- src/algorithms/plain.py SHA-256: 340fc47739932debf6a3ca1c406f8d8e21ee1b2cd0bb60a79b769ecb05638c2b
- src/models/plain_resnet.py SHA-256: 33a911f1f95f72d66e4f9999dcbaba3e8cf449f973096585f282cec4f0e7e276
- src/datasets/custom.py SHA-256: 2e9888f9b3d60bc84d69269614edbf938f88614e4b77aaebce8e350e3c35e23b
- configs/config.yaml dict:

```python
{
    "conf_id": "Crop_Res18_plain_071824",
    "algorithm": "Plain",
    "log_dir": "Crop",
    "num_epochs": 30,
    "log_interval": 10,
    "parallel": 0,
    "dataset_root": "./data/imgs",
    "dataset_name": "Custom_Crop",
    "annotation_dir": "./data/imgs",
    "split_path": "./data/imgs/annotation_example.csv",
    "test_size": 0.2,
    "val_size": 0.2,
    "split_data": True,
    "split_type": "location",
    "batch_size": 32,
    "num_workers": 4,
    "num_classes": 2,
    "model_name": "PlainResNetClassifier",
    "num_layers": 18,
    "weights_init": "ImageNet",
    "lr_feature": 0.01,
    "momentum_feature": 0.9,
    "weight_decay_feature": 0.0005,
    "lr_classifier": 0.01,
    "momentum_classifier": 0.9,
    "weight_decay_classifier": 0.0005,
    "step_size": 10,
    "gamma": 0.1,
}
```

## Verification
- Syntax check passed: `uv run python -c "import ast; ast.parse(open('tests/regression/test_invariants.py').read())"`.
- Non-isolated invariant tests passed in this worktree: `uv run --with pytest --with PyYAML --with-requirements requirements.txt pytest tests/regression/test_invariants.py -v -m 'not isolated_env'` → 9 passed, 2 deselected.
- Isolated smoke tests passed: `uv run --with pytest --with PyYAML pytest tests/regression/test_invariants.py::test_default_env_imports_clean tests/regression/test_invariants.py::test_customweights_no_dinov3_deps -m isolated_env -v` → 2 passed.

## Cross-boundary issues
- This isolated worktree does not yet contain coder-1-infra files (pyproject.toml, requirements-dev.txt, marker registrations, default marker exclusions). Exact `uv run pytest tests/regression/test_invariants.py -v` failed before collection with `Failed to spawn: pytest`. The passing commands above add pytest/PyYAML explicitly and use `-m 'not isolated_env'` to emulate coder-1 default exclusions.
- Current CustomWeights requires a weight-loading path; I5/I12 use a synthetic in-memory ResNet-50 checkpoint via monkeypatched `torch.load`, preserving the legacy three-keyword call shape `CustomWeights(weights=w, class_names=c, device=d)` without writing temporary files.

STATUS: DONE COMMIT=5dcfac0fa495d37a38d81553bcd1e2d7014d8dae SIGNATURE="test_main_unchanged" ITEM=p0-invariants
