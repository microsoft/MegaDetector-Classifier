# Inquisitor review — round 1 (Phase 2, FINAL REVIEW)

Branch `feat/dinov3-port`; spec `DINOv3_classification_port_plan.md`. I opened the
current source of all 13 ledger files at HEAD `9829e74`, independently re-derived
each editor claim, cross-checked against plan §2.4/§6/§7/§8/§10/§11/§12, and hunted
for issues neither editor found. Coverage of the applied state: auditor commit
`c3460fb` (2 files, ITEM-AUD-001..005), reviewer commit `c9fc011` (6 files,
ITEM-REV-001..010). The two later doc-fix commits (`3244405`, `9829e74`) touched
only docs/review artifacts — confirmed via `git show --stat`; no ledger source file
was changed after the audit-fix commits. Exactly 8 source files differ across
`c3460fb~1..c9fc011`; the other 5 ledger files are byte-unchanged (git-verified).

## Approval recap

All 15 items the inquisitor approved at Phase 1 were applied and are confirmed
correct here:

- Auditor (structural, no behavioral change): AUD-001..005 — all APPROVED at Phase 1,
  all applied in `c3460fb`, all re-verified below.
- Reviewer (behavioral hardening): REV-001..010 — REV-002 was approved as MODIFY
  (make checks CONDITIONAL); the applied code honors that (verified field-by-field
  below). REV-009 approved-with-caveats (placeholder-None still raises; SHA/class-order
  guards conditional) — both caveats honored. All others APPROVED and applied in
  `c9fc011`.

No item was rejected; nothing approved was skipped. The editors' disjoint ownership
(auditor vs reviewer file sets) held — no same-file collisions.

## Fix verification (per file)

<a name="src-models-dinov3-py"></a>
### src/models/dinov3.py  — REV-005 — VERIFIED
`_cls_embedding_from_output` (lines 85-107) now requires a 3-D `last_hidden_state`
(direct tensor, or inside tuple/list/dict) and raises `TypeError` otherwise; the
`pooler_output` and 2-D fallbacks are gone. `forward()` (line 82) independently uses
`out.last_hidden_state[:, 0, :]`, so `get_embeddings` is now consistent with the
classification path — both take CLS = token 0 of last_hidden_state (§10/§2.5 parity
intact). The `transformers` import stays lazy inside `__init__` (lines 40-47); no
top-level heavy import. Real HF DINOv3 always exposes `last_hidden_state`, so no
working path regresses. Exercised by `tests/unit/test_dinov3_model.py` (passing).

<a name="src-models-dinov3-registry-py"></a>
### src/models/dinov3_registry.py  — REV-008 — VERIFIED
`ensure_model_available` resolves aliases via `resolve_model_id(model_id)` at line 58
BEFORE importing `huggingface_hub` (line 60), so `ensure_model_available("base")`
works and a bad size raises `ValueError` (resolve_model_id lines 37-40) with zero Hub
interaction. `_snapshot_has_required_files` (43-50) requires preprocessor + weights, so
a config-only local cache is now correctly rejected with an actionable RuntimeError
(89-94) instead of a false "available". Both return paths are normalized to the
snapshot root (line 88 cache branch, line 108 download branch). `huggingface_hub` stays
lazily imported → §11/§12 import invariants preserved. Exercised by
`tests/unit/test_dinov3_registry.py` (passing).

<a name="src-models-classification-init-py"></a>
### src/models/classification/__init__.py  — NO-FINDINGS — VERIFIED UNCHANGED
Git-confirmed unchanged by both audit-fix commits. The §12 I1 guard is intact: the
optional `dinov3_base` re-export (line 14) is gated behind `find_spec("transformers")`
AND `find_spec("huggingface_hub")` (lines 9-13); a missing dep raises
`ModuleNotFoundError` that is swallowed only when `e.name` is one of those two deps
(lines 15-17) — any other ModuleNotFoundError still propagates. `__getattr__`
(27-33) raises an actionable ImportError for the four DINOv3 names when deps are
absent. So `import src.models.classification` succeeds with NO dinov3 deps. (The
dedicated runtime proof is `test_customweights_no_dinov3_deps`, which is deselected
from the baseline matrix by design — see Verification results.)

<a name="src-models-classification-dinov3-base-init-py"></a>
### src/models/classification/dinov3_base/__init__.py  — NO-FINDINGS — VERIFIED UNCHANGED
Git-confirmed unchanged. Re-exports `DINOv3AmazonRainforest`, `DINOv3BaseInference`,
`DINOv3CustomWeights`, `DINOv3Transform`, `_extract_state_dict`,
`extract_checkpoint_meta`. This module is only reached after the parent guard passes,
and every heavy dep (`transformers`/`peft`) stays lazy inside the submodule bodies —
confirmed by a repo-wide grep showing no module-top-level `transformers/peft/
huggingface_hub` import anywhere in `src/`.

<a name="src-models-classification-dinov3-base-base-classifier-py"></a>
### src/models/classification/dinov3_base/base_classifier.py  — REV-001/003/006/007 — VERIFIED
Flavour detection in `_extract_state_dict` (240-313) accepts all real layouts and
rejects nothing valid:
- A (md_cls): `state_dict` with `net.*` → strip + validate (240-253).
- D (classifier-only, REV-001): `_classifier_only_meta` (109-114) merges top-level
  exporter keys into `meta`; `has_classifier_only` keys on `format ==
  "dinov3_classifier_only"` OR a nested-`meta` dict (257-264); the stage-2 exporter's
  top-level, transform-less dict is accepted with `declared=True`
  (`_validate_classifier_only_meta` 117-198: feature_type tolerated as 'cls', missing
  `transform`/`transform_version` skipped, num_classes/embedding_dim cross-checked
  against the weight shape).
- B (sinchi_raw) and C (sinchi_peft): read classifier tensors from `model_state_dict`,
  NOT top level (282-306), so no collision with flavour-D detection.
- unknown → ValueError (311-313).
REV-003: `_validate_classifier_tensors` (56-82) requires weight rank-2, bias rank-1,
matching rows — applied before any `cls_w.shape[1]` indexing. REV-006:
`extract_embeddings` (556-606) is an embedding-only `eval()`+`inference_mode()` loop
calling only `get_embeddings`; the saved path `_embedding_pickle_path` (463-467) uses
the canonical `safe_name(...).pkl` with NO sha1 suffix and NO `:`-strip (round-trips
with the dataset), and `_register_embedding_output` (477-491) raises on an in-run
filename collision. REV-007: `__init__` (336-341) normalizes `self.device` to
`torch.device` and moves `self.net`/`self.predictor`; default `device="cpu"` keeps it
a no-op for existing callers. Exercised by `test_dinov3_extract_state_dict.py`
(md_cls/raw/peft/classifier_only/unknown) and `test_dinov3_custom_weights_mdckpt.py`
(passing).

<a name="src-models-classification-dinov3-base-amazon-py"></a>
### src/models/classification/dinov3_base/amazon.py  — REV-009 — VERIFIED
Manifest gains `model_name/embedding_dim/num_classes/class_names/head_url/sha256/
flavor/revision`. `_validate_manifest` (102-126) runs `require_published` ONLY when
`weights is None` (line 192), so the placeholder `None` head_url/sha256 entries still
raise the actionable "not yet published" `NotImplementedError` (120-126) — caveat 1
honored. `_verify_sha256` is a no-op when expected is `None` (137-139), and the
local-`weights` path goes straight to `DINOv3CustomWeights` (199-212) without touching
`_resolve_head_source`, so a user-supplied local checkpoint is never SHA-gated. The
expected-* handoff (206-211) inherits REV-002's conditional guard: I traced a local
flavour-B RAW head through `DINOv3CustomWeights` and confirmed `expected_feature_type`/
`expected_transform`/`expected_transform_version`/`expected_class_names` do NOT fire
(meta lacks those fields → the `meta.get(...) is not None` / both-non-None guards skip)
— caveat 2 honored; only `expected_num_classes` (always safe vs weight shape) and
`expected_image_size` (when meta carries it) apply. Exercised by
`test_dinov3_amazon_placeholder.py` (passing).

<a name="src-models-classification-dinov3-base-custom-weights-py"></a>
### src/models/classification/dinov3_base/custom_weights.py  — REV-002/003/004 — VERIFIED
The MODIFY mandate (conditional, never unconditional asserts) is met. Every tightened
check is field- or expectation-gated and therefore does NOT reject the real §2.4
RAW/PEFT heads (whose `_base_meta` carries only model_name/num_classes/image_size/peft/
flavor): model_size (232, both-present), embedding_dim (261, meta-present), image_size
(277, expected+meta both-present), feature_type (288, meta-present), transform (296,
meta-present), transform_version (303, meta-present), class-order (319-321, both-present).
num_classes is always cross-checked against `classifier.weight` rows (250/255/370) — safe
and encouraged. REV-004: `_wrap_peft_backbone` (164-192) accepts `alpha` as a
`lora_alpha` alias (preferring an existing `lora_alpha`), rejects non-lora `method`,
and requires r/lora_alpha/target_modules — all behind the `peft` ImportError guard
(§11 flavour-C gating). REV-003: `cls_w.shape[1]` (364) is safe because rank-2 was
validated at 228; load failures are wrapped as `ValueError` with flavour/model context
(382-388); `strict=False` only for classifier_only (381). Exercised by
`test_dinov3_custom_weights_mdckpt.py` and the flavour tests (passing).

<a name="src-models-classification-dinov3-base-transforms-py"></a>
### src/models/classification/dinov3_base/transforms.py  — REV-010 — VERIFIED
`__call__` (30-35) converts PIL inputs via `img.convert("RGB")` and array/tensor inputs
via `TF.to_pil_image(img).convert("RGB")` before the existing pipeline. The
`Resize((518,518), InterpolationMode.BICUBIC)`, `ToTensor`, and ImageNet
`Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])` (21-28) are untouched, so
SINCHI preprocessing parity is fully preserved; the change is a no-op for existing RGB
callers and yields a (3,518,518) tensor for grayscale/RGBA direct use. Only
torch/torchvision/PIL (base deps) — no new heavy import. Exercised by
`test_dinov3_dataloader_emits_518.py` (passing).

<a name="src-datasets-dinov3-embedding-py"></a>
### src/datasets/dinov3_embedding.py  — NO-FINDINGS — VERIFIED UNCHANGED
Git-confirmed unchanged. `safe_name` (23-25) replaces ONLY `/` and `\` — this is the
canonical round-trip anchor that REV-006 and `scripts/extract_embeddings.py` both reuse;
the dataset reads `<split>/<safe_name(Photo_ID)>.pkl` (96). Independent hunt confirms
the two already-recorded auditor cross-scope behavioral findings remain open (not part
of the applied 15, deferred): C2 (the stored pickle `label` from stage-1 can diverge
from the CSV-derived label, currently harmless because `__getitem__` returns
`self.labels[idx]`, line 144) and C3 (a val-only label raises `KeyError` at 93-94
because `DINOv3EmbeddingDataModule`/`train_dinov3.py` derive `label_to_idx` from train
only). These are real but pre-documented and out of this round's applied scope.

<a name="src-algorithms-dinov3-linear-py"></a>
### src/algorithms/dinov3_linear.py  — AUD-001/002/003 — VERIFIED (structural, no behavioral change)
AUD-001: `efficient_batch_same_class_mixup` removed from the import (line 20);
grep-confirmed it has zero remaining references, `noisy_same_class_mixup` is still used
(line 90), and both symbols still exist in `src/utils/mixup.py`. AUD-002: the
`_conf_get` helper (25-31) replaces the five dict-or-attr sites; equivalence holds
because `hasattr(obj,"get")` is True for dict/Munch (→ `.get`) and False for plain
namespaces (→ `getattr` with the same default) — and the required `num_classes`
lookup (72-74) still raises on a missing top-level key (default preserved by NOT
passing one). AUD-003: the docstring (1-10) now documents both `DINOv3Linear` and
`DINOv3LinearProbe`. No control-flow or value change. `export_classifier_only`
(181-195) emits the flavour-D dict the loader accepts (the train→infer roundtrip).

<a name="scripts-extract-embeddings-py"></a>
### scripts/extract_embeddings.py  — AUD-004/005 — VERIFIED (structural, no behavioral change)
AUD-004: `Iterable` removed from the typing import (line 9); grep-confirmed unused.
AUD-005: `import numpy as np` removed and `np.asarray(embedding, dtype=np.float32)`
replaced by storing `embedding` directly (180-182). This is bit-identical:
`embeddings[idx].detach().cpu().to(dtype=torch.float32).numpy()` already yields a
contiguous float32 ndarray, and `np.asarray` on it was a no-op returning the same
object; the payload is `pickle.dump`-ed in the same loop iteration, so the
pre-existing (and unchanged) tensor-buffer aliasing is harmless. The only `numpy`
token left (line 180) is the torch `.numpy()` method, not the module. Independent note:
the silent `if embedding_path.exists(): continue` collision skip (177-179) is the
already-recorded auditor finding C1 — asymmetric with REV-006's collision-RAISE in
base_classifier; deferred to round 2 by design, not a new issue.

<a name="scripts-train-dinov3-py"></a>
### scripts/train_dinov3.py  — NO-FINDINGS — VERIFIED UNCHANGED
Git-confirmed unchanged. Stage-2 trainer builds `DINOv3LinearProbe` with
model_name/model_size/image_size/class_names (99-108) and saves
`export_classifier_only()` (126) — the exact top-level flavour-D dict the REV-001
loader accepts, closing the train→infer roundtrip exercised by
`tests/integration/test_dinov3_linear_probe.py` (passing). The train-only
`build_label_mapping(train_csv)` reused for val (54) is the source of the recorded C3
KeyError risk above; pre-documented, deferred.

<a name="configs-dinov3-linear-yaml"></a>
### configs/dinov3_linear.yaml  — NO-FINDINGS — VERIFIED UNCHANGED
Git-confirmed unchanged. `num_classes: null` (inferred from the train label mapping),
`image_size: 518`, `lr: 0.001`, `label_smoothing: 0.1`, `batch_size: 256` are all
consistent with `DINOv3LinearProbe` defaults and the stage-2 consumer. No issues.

## Missed issues

I independently hunted across all 13 files (flavour-detection ordering, conditional-
validation gating, device handling, aliasing, import surface, round-trip naming) and
found NO new issue beyond the 15 applied fixes. The only open items are the three
behavioral findings the auditor already documented as cross-scope and deferred
(C1 silent-collision-skip in extract_embeddings.py; C2 divergent stage-1/stage-2 label
maps; C3 val-only-label KeyError). These are pre-existing and pre-documented, NOT newly
discovered this round → NEW=0. They remain open and should be carried into round 2
(they are real and live in ledger files extract_embeddings.py, dinov3_embedding.py,
train_dinov3.py).

## Cross-impact

- Disjoint editor ownership held; no same-file collisions.
- §12 import invariants preserved by construction: the two guarded `__init__` files are
  git-unchanged, and a repo-wide grep confirms NO module-top-level
  `transformers`/`peft`/`huggingface_hub` import was introduced — every heavy dep stays
  lazy (dinov3.py `__init__`, registry function body, custom_weights/amazon submodule
  bodies). `import src` / `import src.models` / `import src.models.classification` stay
  clean without dinov3 deps.
- Round-trip consistency: the flavour-D loader (base_classifier) exactly matches the
  exporter (dinov3_linear.export_classifier_only via train_dinov3) — top-level,
  transform-less, feature_type='cls'. The canonical `safe_name` is shared by both
  stage-1 save paths; the C1 collision-policy asymmetry (skip vs raise) is tracked.
- REV-002's conditional guard is the single source of over-tightening protection that
  REV-009's amazon expected-* handoff relies on — internally consistent, both
  reviewer-owned.

## Verification results

`docs/review/audit-fix/round_01/verification.txt`: FULL pytest suite = **56 passed,
3 deselected, 0 failed**, PYTEST_EXIT=0, in 305s (== baseline; no regression). No
Python linter is configured in the repo (requirements-dev = pytest/pytest-cov/pyyaml;
no `[tool.ruff]`), so the lint gate is N/A per skill policy. The dinov3 behavior is
covered by the passing suite: `test_dinov3_extract_state_dict` (all four flavours +
unknown), `test_dinov3_custom_weights_mdckpt`, `test_dinov3_model`,
`test_dinov3_registry`, `test_dinov3_dataloader_emits_518`,
`test_dinov3_embedding_dataset`, `test_dinov3_amazon_placeholder`, and integration
`test_dinov3_linear_probe`. The dedicated no-dinov3-deps import-guard test
(`test_customweights_no_dinov3_deps`, `@pytest.mark.isolated_env`) is one of the 3
deselected tests — the default `addopts` excludes `isolated_env`/`slow`/`requires_hf`/
`gpu`/`requires_dinov3`. That deselection is the repo's standing baseline-matrix policy,
NOT a regression of this round; the guard itself is preserved by construction as noted
above, and `test_default_env_imports_clean` (which DID run and pass) confirms `import
src` stays clean in a requirements.txt-only isolated env.

## Coverage analysis

- Ledger files: 13 (convergence_mode = editing).
- Verified this round (inquisitor): 13/13 — each has an HTML anchor above its section
  heading matching the path slug, and a `verified` COVERAGE_LOG entry is appended.
- Uncovered: none.
- Editing-mode rule: this round APPLIED 15 changes (`action=="fixed"` entries exist for
  round 1), so the round CANNOT be CONVERGED regardless of cleanliness. A confirming
  round 2 (zero new edits) is required to converge.

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=13/13 UNCOVERED=[] NEW=0
