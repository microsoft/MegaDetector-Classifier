# Round 1 file ownership

## Auditor owns (structural focus: simplification, dead code, naming, readability)
- src/datasets/dinov3_embedding.py
- src/algorithms/dinov3_linear.py
- scripts/extract_embeddings.py
- scripts/train_dinov3.py
- src/models/classification/__init__.py
- src/models/classification/dinov3_base/__init__.py
- configs/dinov3_linear.yaml

## Reviewer owns (behavioral focus: bugs, edge cases, error handling, validation, correctness)
- src/models/classification/dinov3_base/base_classifier.py
- src/models/classification/dinov3_base/custom_weights.py
- src/models/classification/dinov3_base/amazon.py
- src/models/classification/dinov3_base/transforms.py
- src/models/dinov3.py
- src/models/dinov3_registry.py
