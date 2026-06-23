# Auditor plan — round 2 (STRUCTURAL, confirming/cleanup)

Role: Auditor (STRUCTURAL). Ownership is SWAPPED vs round 1 — I own the 6 files the
reviewer fixed in round 1, to apply any STRUCTURAL cleanup the round-1 reviewer
deferred as cross-scope. Round 1 applied 15 fixes; the inquisitor verified 13/13,
NEW=0, 56 tests pass. This is a confirming round.

## Owned files audited this round
- src/models/dinov3.py
- src/models/dinov3_registry.py
- src/models/classification/dinov3_base/base_classifier.py
- src/models/classification/dinov3_base/custom_weights.py
- src/models/classification/dinov3_base/amazon.py
- src/models/classification/dinov3_base/transforms.py

## Did the round-1 reviewer defer any STRUCTURAL items to me?
No. The reviewer's "Cross-Scope Findings" (round_01/reviewer_report.md) are ALL test
additions in `tests/unit/` and `tests/integration/` (classifier-only export acceptance,
PEFT `alpha` alias, registry alias resolution, strict-CLS rejection, safe_name save,
Amazon SHA mismatch, DINOv3Transform grayscale/RGBA, etc.) — test files outside my
ownership, NOT structural source changes. The inquisitor's round-1 "Missed issues"
recorded NEW=0; the only open items (C1 silent-collision-skip, C2 divergent label maps,
C3 val-only KeyError) are BEHAVIORAL and live in files NOT owned by me this round
(scripts/extract_embeddings.py, src/datasets/dinov3_embedding.py, scripts/train_dinov3.py).
So there is no structural item handed to me.

## Structural candidates I evaluated and DECLINED (with rationale)

### Candidate 1 — `_ordered_class_names` defined in two files
- Locations: base_classifier.py:94-106 and custom_weights.py:108-120. The two bodies are
  functionally identical (only the parameter type annotation differs).
- Why a candidate: genuine identical-body duplication; custom_weights already imports
  private helpers (`_extract_state_dict`, `_validate_classifier_tensors`) from
  base_classifier, so importing this one too would follow the established pattern.
- Why DECLINED: it is a 2-definition duplication, BELOW this round's stated dedup bar
  ("dedup of 3+ occurrences"). The reviewer did not defer it. The inquisitor VERIFIED
  both files clean (NEW=0). De-duplicating would add cross-module coupling for marginal
  gain and is exactly the kind of marginal change the round-2 NOTHING-TO-DO bias exists
  to prevent. Not worth a structural edit in a confirming round.

### Candidate 2 — `_FALLBACK_DINOV3_MODEL_IDS` / `_FALLBACK_DINOV3_DIMS` duplicate the registry
- Location: custom_weights.py:17-30 mirror dinov3_registry.py DINOV3_MODEL_IDS / DINOV3_DIMS.
- Why a candidate: the 5-entry maps are duplicated across modules.
- Why DECLINED: this is an INTENTIONAL defensive fallback used only inside `_registry()`
  when `from ...dinov3_registry import ...` raises ImportError. Removing it would change
  error-handling behavior (a behavioral change, explicitly prohibited this round) and
  weaken the §12 "import src without deps" resilience. Keep as-is.

## Other checks (all clean)
- Dead imports: none in any of the 6 owned files (verified import-by-import).
- Dead code: none; every helper/function is referenced.
- §12 import invariants preserved: repo-wide grep confirms NO module-top-level
  transformers/peft/huggingface_hub import; all heavy deps stay lazy.
- SINCHI parity (bicubic 518, ImageNet norm, CLS=token-0 of last_hidden_state) intact in
  transforms.py and dinov3.py — unchanged.

<a name="NO-FINDINGS"></a>
## Conclusion
The 6 files I own are structurally clean. No genuine, justified structural improvement
meets this round's bar without crossing into behavioral change. Reporting NOTHING-TO-DO.

STATUS: NOTHING-TO-DO
