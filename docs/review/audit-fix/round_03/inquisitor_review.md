# Inquisitor review — audit-fix round 3 (FINAL REVIEW, CONFIRMING)

Convergence authority: inquisitor. Round 3 is the editing-mode confirming round on
the 13-file DINOv3-port scope. Rounds 1-2 applied 18 fixes total (commits `c3460fb`,
`c9fc011`, `96e4a38`); the inquisitor verified 13/13 with NEW=0 in both rounds and the
full suite stayed green (56 passed). In round 3 BOTH editors reported NOTHING-TO-DO.
I independently re-read the CURRENT source of all 13 ledger files against
`DINOv3_classification_port_plan.md` (§6 flavours A-D, §7 registry, §8 inference API,
§10 transform parity, §11 dep gating, §12 invariants), re-derived the flavour-D
export→load round-trip, and actively hunted for any remaining defect. I did not
rubber-stamp the editors' NOTHING-TO-DO — every file below was opened and checked.

## Confirmation recap

- **Zero source edits this round.** `git diff --stat 96e4a38..HEAD` over the 13 ledger
  files = 0 files changed (HEAD `8274bf4` is the round-2 *artifacts* commit; the last
  *source* commit is `96e4a38`). All 13 files are byte-identical to the round-2
  verified-green state. `COVERAGE_LOG.jsonl` has **0** `action=="fixed"` entries for
  round 3 (auditor + reviewer both NOTHING-TO-DO).
- **Auditor (STRUCTURAL) round 3:** NOTHING-TO-DO. Re-scanned its 7 owned files; the
  only candidates are prior-round *intentional rejections* (REV-002 dead-but-public
  `DINOv3EmbeddingDataModule`; REV-004 intentional `{}`-vs-RAISE label-map divergence;
  REV-003 inline union loop) — correctly declined, not re-litigated. Confirmed.
- **Reviewer (BEHAVIORAL) round 3:** NOTHING-TO-DO. Re-verified §6/§7/§8/§10/§12 on its
  6 owned model files; fresh partial run `25 passed, 2 deselected`. Confirmed.
- **Verification (round 3):** ZERO source change ⇒ no regression possible. Round-2 full
  suite at this identical source state = **56 passed, 3 deselected, 0 failed**. Lint
  N/A (no linter configured). Independently re-confirmed the byte-identity via git diff.
- **§12 import invariant (independent static check):** repo-wide grep for column-0
  `import/from transformers|peft|huggingface_hub` in `src/` returns **none** — every
  heavy import is lazy (inside `__init__`/methods/`ensure_model_available`/`_wrap_peft_backbone`).

## Verification (per file)

<a name="src-models-dinov3-py"></a>
### src/models/dinov3.py
**Unchanged; re-verified.** `transformers.AutoModel` and `huggingface_hub` import lazily
inside `__init__` (§11/§12: `import src` clean). `freeze_backbone=False` raises
`NotImplementedError("Full FT lands in Phase 6")`; download failures wrapped in an
actionable `RuntimeError`. CLS embedding = token-0 of a 3-D `last_hidden_state` in both
`forward` and `_cls_embedding_from_output` (used by `get_embeddings`), preserving SINCHI
parity; `get_embeddings` runs under `no_grad()` when in eval. Backbone params frozen.
Clean.

<a name="src-models-dinov3-registry-py"></a>
### src/models/dinov3_registry.py
**Unchanged; re-verified.** `DINOV3_MODEL_IDS`/`DINOV3_DIMS` match §7 exactly (small,
small_plus, base, large, huge_plus → 384/384/768/1024/1280). `resolve_model_id` accepts
size aliases and `org/repo` ids, raises on unknown. `ensure_model_available` imports
`huggingface_hub` lazily, honours `try_to_load_from_cache`, `snapshot_download` with the
§7 `allow_patterns`, and maps Gated/Repo/LocalEntry errors to actionable `RuntimeError`s.
The `local_files_only` completeness guard (`_snapshot_has_required_files`) is a sound
strictness add-on. Clean.

<a name="src-models-classification-init-py"></a>
### src/models/classification/__init__.py
**Unchanged; re-verified.** Re-exports resnet/timm/speciesnet/base classifiers
unconditionally (ResNet baseline surface intact, §12). DINOv3 subpackage gated on
`transformers`+`huggingface_hub` via `importlib.util.find_spec`; unrelated
`ModuleNotFoundError` is re-raised (`e.name not in {...}`). `__getattr__` returns an
actionable `ImportError` for the 4 DINOv3 public names when deps are absent. §11/§12
optional-dep invariant intact. Clean.

<a name="src-models-classification-dinov3-base-init-py"></a>
### src/models/classification/dinov3_base/__init__.py
**Unchanged; re-verified.** Pure re-export of `DINOv3AmazonRainforest`,
`DINOv3BaseInference`, `DINOv3CustomWeights`, `DINOv3Transform`, `_extract_state_dict`,
`extract_checkpoint_meta`, with a matching `__all__`. Names resolve to their defining
modules. Clean.

<a name="src-models-classification-dinov3-base-base-classifier-py"></a>
### src/models/classification/dinov3_base/base_classifier.py
**Unchanged; re-verified against §6 + §8.** `_extract_state_dict` recognises all four
flavours by structural signature: A (`state_dict` w/ `net.*` → strip prefix), B (SINCHI
RAW: `model_state_dict` w/ `backbone.*`+classifier, no PEFT base → remap `backbone.`→
`feature.`), C (SINCHI PEFT: `backbone.base_model.model.*`+lora/cfg.peft → remap+meta),
D (classifier-only: top-level `classifier.weight/bias` + `meta`, no backbone tensors).
`_validate_classifier_only_meta(declared=True)` enforces format/version, required
`model_name`/`embedding_dim`/`num_classes`/`image_size`, `feature_type=='cls'`,
`transform=='dinov3_bicubic_518_imagenet'`, `transform_version==1`, and
`num_classes==classifier.weight.rows` — exactly the keys the stage-2 exporter emits.
Rank/shape and bias-length validation present. §8 API complete:
`results_generation` (img_id/prediction/class_id/confidence/all_confidences),
single/batch classification, `batch_image_classification_with_embeddings`
(save+return), `extract_embeddings`, all under `eval()`+`torch.inference_mode()`. The
`_register_embedding_output` filename-collision guard mirrors the stage-1 §9.1 remedy.
Clean.

<a name="src-models-classification-dinov3-base-amazon-py"></a>
### src/models/classification/dinov3_base/amazon.py
**Unchanged; re-verified.** `DINOv3AmazonRainforest(model_size="base", weights=None,
device="cpu")` resolves the manifest, validates fields/flavor/class-name count, and
`require_published` raises a clear `NotImplementedError` when head_url/sha256 are unset
(placeholder pending Zenodo) while still allowing a local `weights=` path. SHA256 verify
before load; http/file/path sources handled. Delegates to `DINOv3CustomWeights` with
`expected_transform="dinov3_bicubic_518_imagenet"`, `expected_transform_version=1`,
`expected_feature_type="cls"`, `expected_image_size=518`, and `expected_class_names` —
matching the exporter's stamped metadata, so a stage-2 head loads without a
transform-identity gap. 36 CLASS_NAMES, IMAGE_SIZE=518 (§7/§8). Clean.

<a name="src-models-classification-dinov3-base-custom-weights-py"></a>
### src/models/classification/dinov3_base/custom_weights.py
**Unchanged; re-verified.** Full §6 load-time validation: resolves/cross-checks
model_name (requested vs checkpoint), num_classes (meta vs weight rows vs expected),
embedding_dim (meta vs weight cols vs requested), image_size, feature_type, transform,
transform_version, and class-name order. `cls_w.shape[1] == feature.config.hidden_size`
asserted; `strict=False` used only for `classifier_only`. PEFT wrap gates additionally
on `peft` with an actionable `ImportError` (§11 split-import). The `_FALLBACK_DINOV3_*`
maps inside `_registry()` are an intentional §12 import-resilience fallback. Confirms
the export→load round-trip end-to-end. Clean.

<a name="src-models-classification-dinov3-base-transforms-py"></a>
### src/models/classification/dinov3_base/transforms.py
**Unchanged; re-verified against §10.** `DINOv3Transform`: bicubic `Resize((518,518))`,
`ToTensor`, ImageNet `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`, with
RGB convert in `__call__` (PIL or tensor input). Exact SINCHI preprocessing parity; does
not reuse the generic bilinear transform. Clean.

<a name="src-datasets-dinov3-embedding-py"></a>
### src/datasets/dinov3_embedding.py
**Unchanged; re-verified.** `safe_name` matches the stage-1 SINCHI rule (`/`,`\`→`_`).
`build_label_mapping` RAISES `KeyError` on a missing Label column — intentionally distinct
from stage-1 `_label_mapping` which returns `{}` (basis for the REV-004 rejection).
`DINOv3EmbeddingDataset` reads cached pickles, recomputes labels from the CSV via the
injected `label_to_idx` (never trusts the pickle's stored label), warns+skips missing
embeddings, and raises on an empty split. `DINOv3EmbeddingDataModule` remains
unreferenced (grep-confirmed: only its own def + `__all__`) — the validated REV-002
intentional public-API helper, not dead code to remove. Clean.

<a name="src-algorithms-dinov3-linear-py"></a>
### src/algorithms/dinov3_linear.py
**Unchanged since the R2 fix; re-verified.** `export_classifier_only` raises `ValueError`
when `model_name` is falsy (guards a declared-but-unloadable flavour-D dict) and emits the
full §6 authoritative block — format, version, model_name, model_size, embedding_dim,
num_classes, image_size, `transform="dinov3_bicubic_518_imagenet"`, `transform_version=1`,
`feature_type="cls"`, ordered class_names, and detached/cloned `classifier.weight`/`.bias`.
`DINOv3Linear` (image probe) keeps PEFT/full-FT as explicit `NotImplementedError` stubs;
`DINOv3LinearProbe` (cached-embedding probe) validates `class_names` length == num_classes.
Clean.

<a name="scripts-extract-embeddings-py"></a>
### scripts/extract_embeddings.py
**Unchanged since the R2 fix; re-verified against §9.1.** Builds the chosen backbone
(auto-download), applies `DINOv3Transform(518)`, computes CLS embeddings, writes
SINCHI-compatible per-image pickles `{embedding(float32), label(int, -1 if unlabeled),
image_path}`. Collision safety is complete: intra-run `seen_targets` raises on two distinct
images mapping to one pickle (A-then-B), inter-run `_verify_existing_embedding` re-reads the
cached pickle on resume-skip and raises on an `image_path` mismatch (B-then-A), and a
per-split `manifest.csv` (`image_path,pkl_path,label`) is written (header-only on empty
split). Same-image resume skip preserved. Clean.

<a name="scripts-train-dinov3-py"></a>
### scripts/train_dinov3.py
**Unchanged since the R2 fix; re-verified.** Builds `label_to_idx` from the sorted (by
`str`) union of `build_label_mapping(train_csv)` and `build_label_mapping(val_csv)` and
passes the single shared mapping to both datasets, so a val-only class trains with an extra
output instead of tripping the `KeyError`. `inferred_num_classes=len(label_to_idx)`;
`num_classes` override is guarded (`num_classes < inferred → ValueError`).
`_class_names_from_label_mapping` inverts the mapping in index order, so class-name order
matches dataset label indices and the exported checkpoint (§14 drift mitigation). Saves a
flavour-D checkpoint via `export_classifier_only` (which requires `model_name`, always set
to `resolve_model_id(model_size)`). Clean.

<a name="configs-dinov3-linear-yaml"></a>
### configs/dinov3_linear.yaml
**Unchanged; re-verified.** Keys are consistent with the train script: `model_size: base`,
`embeddings_dir`/`train_csv`/`val_csv` present (the required `config[...]` keys),
`num_classes: null` → inferred from the union mapping, `image_size: 518`, plus
epochs/lr/label_smoothing/batch_size/num_workers/output_dir defaults. Coherent. Clean.

## Missed issues

**NEW issues found beyond the prior rounds: NONE.**

Independent hunt this round (beyond the editors' scans):
- **§6 flavour dispatch ordering** — classifier-only is matched before the `model_state_dict`
  branch, but it requires *no* backbone tensors and either a declared `format` or a `meta`
  dict; SINCHI RAW/PEFT ckpts (which carry backbone keys under `model_state_dict`) cannot be
  misrouted into flavour D. PEFT is matched before RAW (`has_peft_base and (has_lora or
  cfg_peft)` first), and a bare `backbone.*` with no classifier raises the Phase-8
  `NotImplementedError`. Dispatch is unambiguous. Not an issue.
- **flavour-D export→load round-trip** — every field `export_classifier_only` emits is
  required/accepted by `_validate_classifier_only_meta` and `DINOv3CustomWeights`/Amazon
  transform checks; no required field is unset on the documented path. Coherent.
- **§12 import invariants** — column-0 grep over `src/` finds no top-level
  `transformers`/`peft`/`huggingface_hub` import; the classification-package guard and
  `__getattr__` stub are intact; ResNet/TIMM/SpeciesNet public surface re-exported
  unconditionally. Confirmed.
- **§10 transform parity / CLS semantics** — bicubic 518 + ImageNet norm (transforms.py) and
  CLS=token-0 of `last_hidden_state` (dinov3.py) unchanged and SINCHI-faithful.
- **Manifest label vs cached label on resume-skip** — re-examined; still benign:
  `DINOv3EmbeddingDataset` recomputes every label from the CSV via the shared `label_to_idx`
  and ignores the pickle's stored label, and `_verify_existing_embedding` already raises on
  any image_path mismatch. No wrong-vector or wrong-label exposure.

## Coverage analysis

- Ledger files: **13** (`SCOPE_LEDGER.json`, `convergence_mode: editing`, no narrowing —
  scope unchanged across all 3 rounds).
- One inquisitor `verified` entry appended to `COVERAGE_LOG.jsonl` per ledger file this
  round, each `report_anchor` resolving to an `<a name="…"></a>` anchor in THIS file
  (`round_03/inquisitor_review.md`).
- `~/.copilot/scripts/scope_check.sh docs/review/audit-fix` → `covered=13/13 uncovered=[]`,
  exit 0. **SCOPE_CHECK=PASS, COVERED=13/13, UNCOVERED=[].**

## Convergence decision

Editing-mode convergence requires ALL of: zero new findings, zero source edits this round,
full coverage, and clean verification. All four hold:
- zero new issues (NEW=0; only prior-round intentional rejections remain, not re-litigated);
- zero edits (git diff 96e4a38..HEAD over ledger files = 0; 0 `fixed` entries in round 3);
- full coverage (13/13, scope_check PASS);
- clean verification (byte-identical source ⇒ round-2 green 56 passed/0 failed stands).

This round CONVERGES.

STATUS: CONVERGED SCOPE_CHECK=PASS COVERED=13/13 UNCOVERED=[]
