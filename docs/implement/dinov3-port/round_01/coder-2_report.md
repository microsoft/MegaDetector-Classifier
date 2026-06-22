# coder-2 report — inference-core

<a name="inference-core"></a>
## Scope item: inference-core

## Files created/modified
- src/models/dinov3.py
- src/models/classification/dinov3_base/base_classifier.py
- src/models/classification/dinov3_base/custom_weights.py
- src/models/classification/dinov3_base/amazon.py
- src/models/classification/dinov3_base/transforms.py
- src/models/classification/dinov3_base/__init__.py
- tests/unit/test_dinov3_extract_state_dict.py
- tests/integration/test_dinov3_amazon_placeholder.py

## Final commit
- 121c6e9e48c7fafb9c4723bdcbf809133946275d

## SIGNATURE
- DINOv3AmazonRainforest

## Tests/validation
- PASS: /shared/home/aahernandez/MegaDetector-Classifier/.venv/bin/python -m py_compile <owned .py files>
- PASS: git diff --check <owned files>
- NOT RUN per task constraint: pytest / torch imports / transformers imports / peft imports.

## Key design decisions
- _extract_state_dict now recognizes MD-Cls, SINCHI raw, SINCHI PEFT, and classifier_only checkpoints and returns metadata when requested.
- DINOv3CustomWeights resolves model identity from explicit args first, then checkpoint metadata, and validates classifier dimensions before loading.
- DINOv3BaseInference mirrors TIMM-style single/batch inference and adds one-backbone-pass embedding extraction with optional SINCHI-schema pickle output.
- Amazon DINOv3 weights remain a Zenodo placeholder; local weights are accepted via weights=<path>.

## Cross-boundary issues
- Registry import is lazy/fallback-compatible in this branch; coder-1's src/models/dinov3_registry.py will be used after merge.

STATUS: DONE COMMIT=121c6e9e48c7fafb9c4723bdcbf809133946275d SIGNATURE="DINOv3AmazonRainforest" ITEM=inference-core
