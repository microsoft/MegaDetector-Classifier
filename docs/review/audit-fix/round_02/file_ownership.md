# Round 2 file ownership (SWAPPED vs round 1)

Rationale: round-1 editors deferred cross-scope concerns. Auditor flagged BEHAVIORAL
issues in its files (now reviewer's), reviewer flagged STRUCTURAL issues in its files
(now auditor's). Swapping ownership puts the right concern-type agent on each file.

## Auditor owns (structural focus) — formerly reviewer's files
- src/models/classification/dinov3_base/base_classifier.py
- src/models/classification/dinov3_base/custom_weights.py
- src/models/classification/dinov3_base/amazon.py
- src/models/classification/dinov3_base/transforms.py
- src/models/dinov3.py
- src/models/dinov3_registry.py

## Reviewer owns (behavioral focus) — formerly auditor's files
- src/datasets/dinov3_embedding.py
- src/algorithms/dinov3_linear.py
- scripts/extract_embeddings.py
- scripts/train_dinov3.py
- src/models/classification/__init__.py
- src/models/classification/dinov3_base/__init__.py
- configs/dinov3_linear.yaml
