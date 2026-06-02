# coder-3-baseline report

<a name="p0-baseline"></a>
## p0-baseline

Implemented I3 ResNet baseline behavior snapshot for Phase 0 only.

### Files changed
- `tests/regression/generate_resnet_baseline.py`
- `tests/fixtures/baseline_predictions.json`
- `tests/regression/test_resnet_baseline_predictions.py`

### Implementation notes
- Monkeypatches `src.models.plain_resnet.load_state_dict_from_url` before constructing `PlainResNetClassifier`.
- Synthetic ResNet-18 and ResNet-50 state dicts use `torchvision.models.resnet.*(weights=None)` shape templates, local generators (`42`, `4242`), and no network downloads.
- Calls `feat_init()` so the feature weight-loading path is covered.
- Uses deterministic seeds for model construction and input seeds `[1, 2, 3]`.
- Golden fixture contains 6 predictions: 3 inputs × 2 architectures.

### Verification
- `uv run python tests/regression/generate_resnet_baseline.py` created `tests/fixtures/baseline_predictions.json` with 6 entries.
- Re-running the generator produced identical SHA-256: `edc867f0587dfa4d42c53127653fe4309680722a38a9c4ec4187d3ca25923058`.
- `uv run pytest tests/regression/test_resnet_baseline_predictions.py -v 2>&1 | tail -30` passed: 6 passed in 11.15s.

### Commit
- `94fd71194d09a91f3b5434325724328695e6c6ef`

STATUS: DONE COMMIT=94fd71194d09a91f3b5434325724328695e6c6ef SIGNATURE="test_resnet_baseline_prediction" ITEM=p0-baseline
