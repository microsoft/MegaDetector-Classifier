# Inquisitor review — audit-fix round 2 (FINAL REVIEW, Phase 2)

Convergence authority: inquisitor. Round 2 was a confirming round on a 13-file
scope. The auditor reported NOTHING-TO-DO (6 model files structurally clean); the
reviewer APPLIED 3 behavioral fixes (commit `96e4a38`) after the inquisitor approved
3 of 5 planned items and rejected 2. I independently re-read the CURRENT source of
all 13 ledger files, re-derived the plan contract (§6/§9.1/§12/§14), and re-ran the
collision/roundtrip reasoning. All 56 tests still pass (0 failed == baseline).

## Approval recap

| Item | File | Verdict | Independent re-judgement |
|------|------|---------|--------------------------|
| ITEM-REV-001 | scripts/extract_embeddings.py | APPROVED | UPHELD. `if embedding_path.exists(): continue` silently served the first image's vector on any `safe_name` (`/`,`\`→`_`) collision. Plan §9.1 prescribes the exact remedy (manifest.csv + raise on collision). Applied fix matches verbatim. |
| ITEM-REV-002 | src/datasets/dinov3_embedding.py | REJECTED | UPHELD. `DINOv3EmbeddingDataModule` has **0** references in src/scripts/tests/configs (re-grepped: only its own def + `__all__`). The bundled public-API helper refactor touches the tested `build_label_mapping` for marginal gain. Correctly deferred; the real C3 fix lives in REV-003. |
| ITEM-REV-003 | scripts/train_dinov3.py | APPROVED (MODIFY: inline union) | UPHELD. val-only class → `KeyError` at dinov3_embedding.py:94. Inline sorted train+val union, shared mapping to both datasets. Applied fix is self-contained (no REV-002 helper). |
| ITEM-REV-004 | scripts/extract_embeddings.py | REJECTED | UPHELD. Stage-1 `_label_mapping` returns `{}` on a missing Label column (a §9.1 unlabeled-extraction feature → label=-1), whereas `build_label_mapping` RAISES. The divergence is intentional; dedup would risk §9.1. Correctly deferred. |
| ITEM-REV-005 | src/algorithms/dinov3_linear.py | APPROVED | UPHELD. Export omitted `transform`/`transform_version` (plan §6 authoritative keys) and could stamp a declared flavour-D dict with `model_name=None` (unloadable: loader requires it). Both halves minimal + spec-conformant. |

Conclusion: the 3 approvals are genuine correctness fixes; the 2 rejections are
correct (over-engineering on a dead path; intentional-divergence dedup with §9.1
regression risk). No approval/rejection is overturned.

## Fix verification (per file)
<a name="src-models-dinov3-py"></a>
### src/models/dinov3.py

**Unchanged in R2.** Lazy `from transformers import AutoModel` inside `__init__` keeps
`import src` clean (invariant I1/§12). CLS embedding is taken from
`last_hidden_state[:, 0, :]` (token-0) in both `forward` and `_cls_embedding_from_output`
(used by `get_embeddings`), preserving SINCHI parity. No top-level heavy import. Clean.

<a name="src-models-dinov3-registry-py"></a>
### src/models/dinov3_registry.py

**Unchanged in R2.** Pure dict registry (`DINOV3_MODEL_IDS`, `DINOV3_DIMS`) plus helpers;
`huggingface_hub` is imported lazily inside `ensure_model_available`. `resolve_model_id`
(consumed by the R2 train fix and the linear-probe export) returns the canonical HF id for
a size alias. No module-top heavy import (§12 intact). Clean.

<a name="src-models-classification-init-py"></a>
### src/models/classification/__init__.py

**Unchanged in R2.** Guarded `from .dinov3_base import *` gates on `transformers` +
`huggingface_hub` via `importlib.util.find_spec`, re-raises unrelated `ModuleNotFoundError`,
and `__getattr__` returns an actionable `ImportError` for the 4 DINOv3 names when deps are
absent. §11/§12 optional-dep invariant intact.

<a name="src-models-classification-dinov3-base-init-py"></a>
### src/models/classification/dinov3_base/__init__.py

**Unchanged in R2.** Pure re-export of `DINOv3AmazonRainforest`, `DINOv3BaseInference`,
`DINOv3CustomWeights`, `DINOv3Transform`, `_extract_state_dict`, `extract_checkpoint_meta`
behind the parent guard. Clean.

<a name="src-models-classification-dinov3-base-base-classifier-py"></a>
### src/models/classification/dinov3_base/base_classifier.py

**Unchanged source; loader contract re-verified against the R2 export.**
`_CLASSIFIER_ONLY_META_KEYS` already lists `transform` and `transform_version`;
`_classifier_only_meta` hoists top-level export keys into `meta`; `_validate_classifier_only_meta(declared=True)`
requires `model_name`/`embedding_dim`/`num_classes`/`image_size`, and validates
`transform=='dinov3_bicubic_518_imagenet'`, `transform_version==1`, `feature_type=='cls'`,
`num_classes==classifier.weight.rows`. This is exactly the shape `export_classifier_only`
now emits → flavour-D roundtrip is coherent. Clean.

<a name="src-models-classification-dinov3-base-amazon-py"></a>
### src/models/classification/dinov3_base/amazon.py

**Unchanged in R2.** Passes `expected_transform="dinov3_bicubic_518_imagenet"` and
`expected_transform_version=1` to `DINOv3CustomWeights` — exactly the metadata the R2 export
now stamps, so a stage-2 head loads through Amazon without a transform-identity gap.
SHA256 + manifest-field validation intact. Clean.

<a name="src-models-classification-dinov3-base-custom-weights-py"></a>
### src/models/classification/dinov3_base/custom_weights.py

**Unchanged in R2.** Flavour-D load path validates `transform`/`transform_version` against the
canonical pair and uses `strict=False` only for `classifier_only`. Confirms export→load
roundtrip end-to-end. The `_FALLBACK_DINOV3_*` maps inside `_registry()` are an intentional
§12 import-resilience fallback (auditor declined dedup as behavioral — concur). Clean.

<a name="src-models-classification-dinov3-base-transforms-py"></a>
### src/models/classification/dinov3_base/transforms.py

**Unchanged in R2.** Bicubic resize to (518,518), `ToTensor`, ImageNet `Normalize`
(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), RGB convert — exact SINCHI parity (§10).
Clean.

<a name="src-datasets-dinov3-embedding-py"></a>
### src/datasets/dinov3_embedding.py

**Unchanged source (ITEM-REV-002 rejected).** `build_label_mapping` raises `KeyError` on a
missing Label column — intentionally distinct from stage-1 `_label_mapping` which returns `{}`
(basis for the ITEM-REV-004 rejection). `DINOv3EmbeddingDataset` now receives the union mapping
from the R2 train fix, so a val-only class no longer trips the `KeyError` at line 94.
`DINOv3EmbeddingDataModule` remains unreferenced anywhere (grep-confirmed: only its own def +
`__all__`), validating the REV-002 rejection. Clean.

<a name="src-algorithms-dinov3-linear-py"></a>
### src/algorithms/dinov3_linear.py

**FIXED — ITEM-REV-005 (verified correct).** `export_classifier_only` now (a) raises
`ValueError` when `model_name` is falsy (guards against a declared-but-unloadable flavour-D
dict; `not self.model_name` also catches the empty-string case), and (b) emits
`transform="dinov3_bicubic_518_imagenet"` + `transform_version=1`. Both match plan §6's
authoritative block and the base_classifier loader. Minimal, no behavioral change on the
documented path (train_dinov3 always sets `model_name=resolve_model_id(...)`). No regression
(no test constructs/exports `DINOv3LinearProbe`).

<a name="scripts-extract-embeddings-py"></a>
### scripts/extract_embeddings.py

**FIXED — ITEM-REV-001 (verified correct).** Three additions: (1) intra-run guard `seen_targets`
raises when two distinct image paths map to the same pickle within a split; (2) inter-run
`_verify_existing_embedding` re-reads the cached pickle on resume-skip and raises if its stored
`image_path` differs (so a cross-run collision is also caught); (3) per-split `manifest.csv`
(`image_path,pkl_path,label`). Matches plan §9.1 verbatim. `List` is already imported (line 9).
Both collision orderings raise (A-then-B via `seen_targets`; B-then-A via the on-disk verify);
an empty split yields a header-only manifest (no crash). Same-image resume skip is preserved.
No regression (no test exercises this CLI).

<a name="scripts-train-dinov3-py"></a>
### scripts/train_dinov3.py

**FIXED — ITEM-REV-003 (verified correct).** Builds `label_to_idx` from the sorted (by `str`)
union of `build_label_mapping(train_csv)` and `build_label_mapping(val_csv)`, passes the single
shared mapping to both datasets, derives `inferred_num_classes = len(label_to_idx)` and
`class_names` from it, and keeps the `num_classes` override guard against the union size.
`_class_names_from_label_mapping` inverts the mapping in index order, so class-name order matches
dataset label indices and the exported checkpoint (§14 drift mitigation). A val-only class now
trains with an extra output instead of crashing. Self-contained — no REV-002 shared helper. Correct.

<a name="configs-dinov3-linear-yaml"></a>
### configs/dinov3_linear.yaml

**Unchanged in R2.** `num_classes: null` → inferred from the union mapping; `image_size: 518`;
`model_size: base`. Keys remain consistent with the R2 train-script fix. Clean.

## Missed issues

**NEW issues found beyond the applied fixes: 0.**

Independent hunt (beyond what either round flagged):
- **Manifest label vs cached label on resume-skip** — on a resume-skip, the manifest row
  records the *current* run's CSV label while the on-disk pickle keeps its original label. I
  judged this **benign, not a bug**: stage-2 (`DINOv3EmbeddingDataset.__getitem__`) recomputes
  every label from the CSV via the shared `label_to_idx` and never reads the pickle's stored
  `label`, so the only consumer is unaffected; and `_verify_existing_embedding` already raises on
  any *image_path* mismatch, so the cache cannot serve the wrong vector.
- **Union ordering vs stage-1 ordering** — stage-2's union mapping (sorted by `str`) can differ
  from stage-1's full-CSV `_label_mapping`, but stage-2 ignores pickle labels and uses its own
  mapping consistently for dataset labels, head width, and exported `class_names`, so indices stay
  internally coherent. Not an issue.
- **§12 import invariants** — repo-wide grep finds no module-top-level `transformers`/`peft`/
  `huggingface_hub` import in `src/`; all heavy deps stay lazy. The classification package guard
  and `__getattr__` stub are intact.
- **SINCHI parity** — bicubic 518 + ImageNet norm (transforms.py) and CLS=token-0 of
  `last_hidden_state` (dinov3.py) unchanged.
- **flavour-D roundtrip** — `export_classifier_only` (top-level keys incl. transform pair) →
  `_classifier_only_meta` hoist → `_validate_classifier_only_meta(declared=True)` →
  `DINOv3CustomWeights`/Amazon transform checks: every field the exporter emits is required/accepted
  by the loader, and no field the loader requires is missing. Coherent.

## Verification results

From `round_02/verification.txt` (HEAD `96e4a38`, TMPDIR on /dev/shm tmpfs):
- Full suite: **56 passed, 3 deselected, 1 warning** (the pre-existing Lightning `self.log` warning).
- `PYTEST_EXIT=0`. RESULT: ALL CLEAN — 0 failed == baseline. No regressions.
- 3 behavioral fixes applied this round; no linter configured (Lint N/A).

I re-read the result line and the deselect/warning context; the deselects are the gated
integration tests (DINOV3_RUN_INTEGRATION), and the warning is unrelated to the R2 changes.
Confirmed: no regression introduced by commit `96e4a38`.

## Coverage analysis

- Ledger files: **13** (`SCOPE_LEDGER.json`, `convergence_mode: editing`, no narrowing).
- One inquisitor `verified` entry appended to `COVERAGE_LOG.jsonl` per ledger file (round 2),
  each carrying a `report_anchor` that resolves to an `<a name="…"></a>` anchor in this file.
- `~/.copilot/scripts/scope_check.sh docs/review/audit-fix` → `covered=13/13 uncovered=[]`, exit 0.
- **SCOPE_CHECK=PASS, COVERED=13/13, UNCOVERED=[].**

Round 2 APPLIED 3 changes (commit `96e4a38`). Per the `editing` convergence mode, an applied
fix forces NEEDS-MORE for the round even when coverage is complete and zero new issues remain.
A clean confirming round (auditor + reviewer both NOTHING-TO-DO, zero changes, full coverage) is
required before CONVERGED. Therefore this round is NEEDS-MORE with NEW=0.

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=13/13 UNCOVERED=[] NEW=0
