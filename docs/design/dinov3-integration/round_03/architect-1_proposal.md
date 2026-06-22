# DINOv3 integration — phased plan (architect-1: phasing/MVP perspective) — round 3

## Executive overview

Convergence round. The round-2 plan stands; the round-3 deltas are narrowly scoped to the inquisitor's binding directives:

1. **Transform-injection** — drop my round-2 `build_dinov3_eval_transform(image_size)` shim and adopt **architect-3's defaulted-kwarg design** (`Custom_Crop(image_size: int = 224)` with DINOv3 configs overriding to 518). This collapses two transform pipelines into one and removes the only fork in the round-2 design (RESOLVED-TOWARD=architect-3).
2. **`huggingface_hub` joins the optional-dep contract** alongside `transformers` and `peft` in `requirements-dinov3.txt`. The default-install surface test now asserts **all three** are absent from `sys.modules`.
3. **Rollback dry-run** — replaced the fixed sibling-path snippet with a tracked `scripts/rollback_dry_run.sh` that uses `mktemp -d` inside `.git/worktrees-tmp/` plus a `trap` cleanup, so concurrent runs in a shared workspace do not collide.
4. **Absent-optional-class surface picked exactly one shape:** when `transformers` is absent, `from src.algorithms import DINOv3Linear` and `from src.models.classification import DINOv3CustomWeights` raise `ImportError("…install with: pip install -r requirements-dinov3.txt")`. The symbols are *present* (a `__getattr__` lazy proxy on the package) and raise on access. Tests now match this exact contract.
5. **Behavior snapshots remain the only regression gate** (no state-dict hash gates anywhere — the round-2 advisory SHA sentinel remains advisory).
6. **PEFT target-module validation tightened** per directive: empty intersection ⇒ `ValueError`; partial intersection ⇒ `logging.warning` listing the unmatched targets; `TaskType.FEATURE_EXTRACTION` retained.
7. **No packaging migration.** Default `requirements.txt` unchanged; CI uses `pip install -r requirements.txt` (and `pip install -r requirements-dinov3.txt` on the DINOv3 matrix leg). No `[project.optional-dependencies]` change.

Phase count, ordering, gates, and rollback DAG are unchanged from round 2. All round-2 disagreements were adjudicated MISTAKEN-BY=architect-1; I re-read the round-2 proposals and confirm. **DISAGREEMENTS=0.**

## Phase index (table)

| # | Phase name | Deliverable | Tests | Est. LOC | Revert unit |
|---|---|---|---|---|---|
| 0 | Test scaffolding + behavior snapshot baseline | `pytest.ini`, `tests/conftest.py`, `tests/fixtures/mini_dataset/`, `tests/fixtures/mock_hf.py`, `tests/regression/test_resnet_baseline.py`, `tests/regression/test_public_surface.py`, `tests/regression/test_default_config_keys.py`, `scripts/run_acceptance_gate0.sh`, `scripts/rollback_dry_run.sh`, `.github/workflows/ci.yml` (if absent) | 6 baseline tests; ResNet-18 + ResNet-50 prediction snapshots | ~270 | PR #0 |
| 1 | CustomWeights `num_layers` bug fix | Patch `resnet_base/custom_weights.py` (+1 defaulted kwarg); failing-test commit precedes fix commit *within* the PR | `tests/inference/test_custom_weights_num_layers.py` (3 tests) | ~30 + 80 test | PR #1 |
| 2 | Mixup utility port | `src/utils/mixup.py` (verbatim, with provenance docstring) | `tests/utils/test_mixup.py` (6 tests) | ~200 | PR #2 |
| 3 | Optional-dependency wiring + `Custom_Crop(image_size=…)` defaulted kwarg | `requirements-dinov3.txt` (`transformers`, `peft`, `huggingface_hub`), `docs/install.md` (install matrix), **edit** `src/datasets/transforms.py` (add `image_size: int = 224` kwarg to `Custom_Crop`, default unchanged), `tests/regression/test_default_install_surface.py` | 3 absent-env tests (one per package), 2 dataset-transform tests (224 default + 518 override), 1 byte-equal-default test | ~110 | PR #3 |
| 4 | DINOv3 model + linear-probe algorithm (MVP core) — mixup wired off-by-default | `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py`, registry wires (lazy `__getattr__` proxy), `configs/config_dinov3_linear.yaml` (`image_size: 518` ⇒ `Custom_Crop` overrides default) | `tests/models/test_dinov3_classifier.py`, `tests/algorithms/test_dinov3_linear_smoke.py`, `tests/datasets/test_dinov3_transform_pipeline.py` | ~410 | PR #4 |
| 5 | Inference loader v1 (MD-Cls `.ckpt` only) + minimal Gate-1 docs | `src/models/classification/dinov3_base/{__init__,base_classifier,custom_weights}.py`, `docs/dinov3_experimental.md` (auth + memory + image-size warnings + scope statement + license) | `tests/inference/test_dinov3_custom_weights_mdckpt.py` | ~300 + 60 docs | PR #5 |
| — | **Gate 1: ship + collect feedback** (Phases 0–5 cover Steps 1, 2, 3, 4, 6-partial, 7-partial, 8) | — | — | — | — |
| 6 | Full fine-tune toggle | Extend `dinov3_linear.py`: dual param-groups, `unfreeze_backbone()`, memory-warning emit | `tests/algorithms/test_dinov3_finetune_grads.py` | ~150 | PR #6 |
| 7 | PEFT (LoRA + PiSSA) with target-module validation (empty=error, partial=warning) + embedded checkpoint metadata | Extend `dinov3_linear.py::_wrap_with_peft` + `on_save_checkpoint`/`on_load_checkpoint` hooks for `peft_config` round-trip | `tests/algorithms/test_dinov3_peft_smoke.py`, `tests/algorithms/test_peft_checkpoint_roundtrip.py`, `tests/algorithms/test_peft_target_modules_validated.py` | ~280 | PR #7 |
| 8 | Inference loader v2 (SINCHI raw + SINCHI PEFT) with fail-loud key categorization + auto-rewrap | Extend `_extract_state_dict` + `_categorize_keys`; consume embedded `peft_config` metadata or accept explicit kwarg | `tests/inference/test_dinov3_custom_weights_sinchi.py`, `tests/inference/test_dinov3_custom_weights_peft.py`, `tests/inference/test_corrupt_ckpt_errors.py` | ~220 | PR #8 |
| — | **Gate 2: ship + collect feedback** | — | — | — | — |
| 9 | Config + docs consolidation (full) | `configs/config.yaml` (commented additions), sibling config files, `docs/usage.md`, `docs/error_taxonomy.md`, `README.md`, `CHANGELOG.md`, license/citation note | `tests/docs/test_sample_configs_parse.py`, `tests/docs/test_readme_snippets.py` | ~200 | PR #9 |

Cumulative net additions ≈ 1,950 lines (code + tests + docs); largest single PR ≈ 410 LOC.

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

Unchanged from round 2 except: **P3's deliverable is now an edit to `Custom_Crop` (defaulted `image_size: int = 224` kwarg), not a new `build_dinov3_eval_transform` shim.** ResNet path stays bit-equal because the default value matches the prior hard-coded value, proved by `tests/datasets/test_resnet_transform_unchanged.py` comparing to the P0 baseline snapshot.

**Ordering rationale (MVP lens, post-round-2):**

1. **Phase 0** — scaffolding (`pytest.ini`, mini-dataset, `MockAutoModel`, behavior snapshots via predictions JSON, public-surface freeze, `scripts/rollback_dry_run.sh`).
2. **Phase 1** — `CustomWeights(num_layers)` bug fix, failing-test-first inside PR.
3. **Phase 2** — Mixup port (leaf utility).
4. **Phase 3** — Optional-dep wiring + `Custom_Crop` defaulted-kwarg edit. Single transform entry point; no separate shim. `transformers`, `peft`, `huggingface_hub` all in `requirements-dinov3.txt`.
5. **Phase 4** — MVP core (DINOv3Classifier + DINOv3Linear). Mixup wired but `use_mixup: False`. PEFT/full-FT guarded with `NotImplementedError`.
6. **Phase 5** — Loader v1 (MD-Cls only) + minimal `docs/dinov3_experimental.md`.
7. **Gate 1** — ship + ≥1 week feedback.
8. **Phase 6** — Full FT.
9. **Phase 7** — PEFT (LoRA + PiSSA) with empty-error / partial-warning target validation.
10. **Phase 8** — Loader v2 (SINCHI raw + PEFT). Hard pre-flight on real ckpt key inspection committed to `docs/design/dinov3-integration/sinchi_key_layouts.md` (tracked-docs path, not artifacts).
11. **Gate 2** — ship + ≥1 week feedback.
12. **Phase 9** — Full docs/config consolidation.

**DAG and rollback chain unchanged from round 2.** Reverse-DAG: P9 → P8 → P7 → P6 → P5 → P4 → P3 → P2 → P1 → P0. P1/P2/P3 are mutually independent prior to P4.

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

Unchanged from round 2. `src/utils/mixup.py` verbatim port with provenance docstring + commit SHA + RNG-source note. Six unit tests under `tests/utils/test_mixup.py` covering shape, label preservation, λ clamp, singleton class, noise-type modes, naive-vs-efficient equivalence. No production callsite until P4.

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

**Phase 4 — MVP core. Mixup wired-but-off-by-default; PEFT and full-FT guarded.**

- **Deliverable:**
  - **NEW** `src/models/dinov3.py` (~90 LOC) — `DINOv3Classifier`. Lazy import of `transformers` and `huggingface_hub` inside `__init__`/factory; raises `ImportError("transformers + huggingface_hub required; install with: pip install -r requirements-dinov3.txt")` if either is absent.
  - **EDIT** `src/models/__init__.py` and `src/algorithms/__init__.py` — install a `__getattr__(name)` lazy proxy at module level. Access to `DINOv3Classifier` / `DINOv3Linear` triggers the import; if `transformers`/`huggingface_hub` missing, raises `ImportError` with the standardized install message. Symbols are *not* removed from `dir(...)` (added to `__all__`) — this is the single contracted surface (resolves A3-N1 by picking the `ImportError`-on-access path).
  - **NEW** `src/algorithms/dinov3_linear.py` (~220 LOC) — `DINOv3Linear` LightningModule:
    - `training_step` calls `noisy_same_class_mixup` only when `self.hparams.use_mixup` is True.
    - `freeze_backbone=False` raises `NotImplementedError("full fine-tune arrives in Phase 6")`.
    - `peft.enable=True` raises `NotImplementedError("PEFT arrives in Phase 7")`.
    - `on_save_checkpoint` writes `checkpoint["hyper_parameters"]["dinov3_metadata"] = {model_name, image_size, normalize_mean, normalize_std, use_mixup}`.
  - **NEW** `configs/config_dinov3_linear.yaml` — `use_mixup: False`, `peft.enable: False`, `freeze_backbone: True`, `image_size: 518`, `model_name: facebook/dinov3-vits16-pretrain-lvd1689m`. The dataset block carries `image_size: 518`, which the dataloader passes to `Custom_Crop(image_size=518)`; the default ResNet path constructs `Custom_Crop()` (kwarg defaults to 224).
- **Entry:** P3 merged (requirements file + `Custom_Crop` defaulted-kwarg edit landed). NO HF token required in CI.
- **Exit:**
  - `from src.algorithms import DINOv3Linear` succeeds with the DINOv3 extras installed.
  - With any of `transformers` / `huggingface_hub` absent: importing `DINOv3Linear` raises `ImportError` with the install instruction; `from src.algorithms import Plain` still works.
  - Smoke training run with `MockAutoModel` produces `.ckpt` containing `hyper_parameters.dinov3_metadata`.
  - Phase-0 behavior snapshot reproduces unchanged.
- **Tests:**
  - `tests/models/test_dinov3_classifier.py`:
    - `test_construct_with_mocked_automodel`
    - `test_encode_extracts_cls_token` — shape `(B, hidden_size)`
    - `test_attribute_naming_matches_md_cls_contract` — `.feature` + `.classifier`
    - `test_transformers_missing_raises_actionable_error` — subprocess, only `requirements.txt` installed, assert `ImportError` whose message names `requirements-dinov3.txt`
    - `test_huggingface_hub_missing_raises_actionable_error` — subprocess, `transformers` installed but `huggingface_hub` hidden (still expect `ImportError` because the proxy checks both)
  - `tests/algorithms/test_dinov3_linear_smoke.py` — 7 tests as round 2 (imports, finetune blocked, peft blocked, mixup on/off, ckpt metadata, frozen grad-flow).
  - `tests/datasets/test_dinov3_transform_pipeline.py`:
    - `test_custom_crop_default_emits_224` — `Custom_Crop()` ⇒ batch shape `(B, 3, 224, 224)`; raw transform bit-equal to P0 baseline serialization
    - `test_custom_crop_override_emits_518` — `Custom_Crop(image_size=518)` ⇒ batch shape `(B, 3, 3, 518, 518)` (per-image `(3, 518, 518)`)
    - `test_dinov3_dataloader_yaml_drives_518` — load `config_dinov3_linear.yaml`, instantiate the dataloader, assert tensor shape
- **Per-phase regression gate:** `pytest tests/models tests/algorithms tests/datasets tests/regression/ -v -m "not gpu and not real_hf"` green.
- **Revert:** single PR.

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

Unchanged from round 2. P6 lifts the `NotImplementedError`, returns two AdamW param groups (`lr_feature`, `lr_classifier`), calls `self.net.unfreeze_backbone()` when `freeze_backbone=False`, and emits a conservative memory warning. Tests cover unfreeze, param-group split, backbone grad flow, frozen-path equivalence with P4 (seeded, deterministic mode), and warn-once.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

**Phase 7 — Lift PEFT guard; runtime target-module validation with two severities; embed `peft_config` in checkpoint.**

- **Deliverable:**
  - **EDIT** `src/algorithms/dinov3_linear.py`:
    - `_wrap_with_peft()` — lazy `from peft import LoraConfig, get_peft_model, TaskType`; `TaskType.FEATURE_EXTRACTION` (per directive); wrap `self.net.feature` only.
    - **NEW** `_validate_peft_targets(target_modules)` (per directive):
      - Build `module_suffix_set = {n.split('.')[-1] for n, _ in self.net.feature.named_modules()}`.
      - Compute `matched = target_modules ∩ module_suffix_set`; `unmatched = target_modules − module_suffix_set`.
      - If `matched == ∅`: raise `ValueError("PEFT target_modules {target_modules} match no modules in {model_name}. Available leaf-module suffixes: {sorted(module_suffix_set)[:30]}")`.
      - If `unmatched ≠ ∅` (partial): `logging.warning("PEFT target_modules contain unmatched entries that will be silently ignored by peft: %s; matched: %s", unmatched, matched)`.
    - `on_save_checkpoint` — extend `dinov3_metadata` with `peft_method`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`, `init_lora_weights`.
    - `on_load_checkpoint` — re-wrap via stored metadata if present.
  - **EDIT** `requirements-dinov3.txt` — already pre-populated in P3 with `peft>=0.10.0,<0.13.0`; no further edits.
- **Tests** (`tests/algorithms/test_peft_target_modules_validated.py`):
  - `test_empty_intersection_raises` — `target_modules=["nonexistent_proj"]` ⇒ `ValueError` whose message includes ≥1 valid candidate suffix.
  - `test_partial_intersection_warns_listing_unmatched` — `target_modules=["q_proj","nonexistent"]` ⇒ wrap succeeds; `caplog.records` contains a WARNING citing `["nonexistent"]`.
  - `test_full_intersection_silent` — fully-matched list ⇒ no warnings; wrap succeeds.
- **PEFT regression test** uses **logits behavior snapshot** (forward on fixed input ⇒ tensor comparison with `atol=1e-5`), **not a state-dict hash** (per directive).
- Rest of phase unchanged from round 2.

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

Unchanged from round 2 except: **real SINCHI key-layout evidence lives at `docs/design/dinov3-integration/sinchi_key_layouts.md`** (tracked design docs path), not `tests/_artifacts/`.

- **P5 Loader v1 (MD-Cls only):** as round 2, with `_categorize_keys` returning `{backbone, classifier, adapter, unknown}` and failing loud on `unknown`. The classification `__init__` uses the same `__getattr__` lazy-proxy pattern as `src/algorithms/__init__.py` for `DINOv3CustomWeights` — access without DINOv3 deps raises `ImportError` with the install instruction.
- **P8 Loader v2 (SINCHI raw + PEFT):** as round 2. Hard pre-flight: `scripts/inspect_sinchi_keys.py` output committed to `docs/design/dinov3-integration/sinchi_key_layouts.md` before any P8 code is written. PEFT-detection regression uses **logits snapshot**, not state-dict hash.

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

Unchanged from round 2. Minimal `docs/dinov3_experimental.md` at P5; full consolidation at P9 (sample configs, `docs/usage.md`, `docs/error_taxonomy.md`, README migration note for P1 bug fix, CHANGELOG, license/citation, DDP out-of-scope, CPU/MPS auto-fallback note).

**Install matrix in `docs/install.md`** (round-3 addition):

| Profile | Command | Includes |
|---|---|---|
| Default (ResNet only) | `pip install -r requirements.txt` | core only — no `transformers`, no `peft`, no `huggingface_hub` |
| DINOv3 train/infer | `pip install -r requirements.txt -r requirements-dinov3.txt` | adds `transformers>=4.40,<5`, `huggingface_hub>=0.20,<1`, `peft>=0.10,<0.13` |

No `[project.optional-dependencies]` migration; no extras_require; no packaging change (per directive).

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

Unchanged from round 2. Three-commit structure inside PR #1 (failing test → patch → flip xfail). Default kwarg `num_layers=50` preserves caller behavior. Tests: ResNet-18 explicit, ResNet-50 default-kwarg, signature compatibility.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

Round-3 hardening (per directive on behavior snapshots):

1. **Behavior snapshot (primary, sole gate).** Predictions JSON for the 10 mini-dataset images; `atol=1e-3` on confidences, exact match on `class_id`. The round-2 advisory `state_dict.sha256` sentinel is **removed entirely** to eliminate any ambiguity about whether it gates merges. Versions/CUDA flag/deterministic-mode flag still recorded as metadata for triage only.
2. **Default-config invariants** (normalized dict equality, no YAML SHA).
3. **Public-surface freeze** — `sorted(dir(...))` for `src.algorithms`, `src.models`, `src.models.classification`. The `__getattr__` lazy proxy means `DINOv3Linear`/`DINOv3CustomWeights` appear in `__all__` (and therefore in `dir(...)`) **from P4 onward**; the freeze snapshot is bumped at P4 with reviewer sign-off and add-only delta.
4. **Byte-equality** restricted to 4 immutable files: `main.py`, `src/algorithms/plain.py`, `src/models/plain_resnet.py`, `src/datasets/custom.py`. (`src/datasets/transforms.py` is *not* in this list because P3 edits it to add the defaulted kwarg.)
5. **Default-install surface freeze (P3).** Subprocess-level: `pip install -r requirements.txt` only; assert **all three** of `transformers`, `peft`, `huggingface_hub` are absent from `sys.modules` after `import src`. Then assert `from src.algorithms import DINOv3Linear` raises `ImportError` with a message naming `requirements-dinov3.txt`. (Adds the `huggingface_hub` check per A1-N1.)

**Behavior-snapshot regen** requires a labelled `regression-baseline-bump` PR with written justification. None of P0–P9 should trigger this.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

Unchanged from round 2 in structure (pyramid, fixtures, CI/local/nightly split, repo-local `tests/_artifacts/`). Two corrections:

- **Real-SINCHI key evidence file** lives at `docs/design/dinov3-integration/sinchi_key_layouts.md` (tracked docs path), not `tests/_artifacts/`. The real SINCHI checkpoint *binaries* still live outside the repo and are referenced via `SINCHI_RAW_CKPT` / `SINCHI_PEFT_CKPT` env vars.
- **CI install commands use `pip install -r` only**, not `pip install -e .[test]`. No packaging migration. Both CI legs (default-only and default+dinov3) run from requirements files.

Per-phase invocation map unchanged.

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

12-item acceptance checklist unchanged from round 2 except:

- Item 10 (lazy-import) now explicitly asserts absence of **all three** optional packages (`transformers`, `peft`, `huggingface_hub`) in the default subprocess environment.
- Item 11 (transform-size proof) uses `Custom_Crop` directly (`Custom_Crop()` ⇒ 224, `Custom_Crop(image_size=518)` ⇒ 518), no `build_dinov3_eval_transform` reference.
- Item 12 (real SINCHI key inspection) output is committed to `docs/design/dinov3-integration/sinchi_key_layouts.md`.

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

Risk register unchanged. **Rollback dry-run operationalized as a committed script** (replaces round-2's absolute-path snippet — A1-N3):

`scripts/rollback_dry_run.sh` (shipped in P0):

```bash
#!/usr/bin/env bash
# Repo-local dry-run rollback. Safe under concurrency on shared workspaces.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKTREE_PARENT="${REPO_ROOT}/.git/worktrees-tmp"
mkdir -p "${WORKTREE_PARENT}"
WORKTREE_DIR="$(mktemp -d "${WORKTREE_PARENT}/rollback.XXXXXXXX")"

cleanup() {
    cd "${REPO_ROOT}"
    git worktree remove --force "${WORKTREE_DIR}" 2>/dev/null || true
    rm -rf "${WORKTREE_DIR}"
}
trap cleanup EXIT INT TERM

TARGET_REF="${1:-HEAD~1}"
git worktree add "${WORKTREE_DIR}" "${TARGET_REF}"
cd "${WORKTREE_DIR}"
pytest tests/regression/ -v
```

- `.git/worktrees-tmp/` is git-internal and naturally excluded from `git status`.
- `mktemp -d` guarantees unique paths under concurrent CI/developer runs.
- `trap` cleanup runs on any exit (success, failure, signal).
- PR template requires running this script and pasting the tail of pytest output.

Reverse-DAG rollback chain unchanged: `P9 → P8 → P7 → P6 → P5 → P4 → P3 → P2 → P1 → P0`.

---

## Rationale (phasing/MVP lens)

Round-3 rationale additions:

- **Why drop the transform shim?** Architect-3's observation is correct: `Custom_Crop` is the *existing* transform; injecting via a defaulted kwarg keeps one code path, requires no new module, and lets the ResNet bit-equality test prove the default value preserves prior behavior. A separate `build_dinov3_eval_transform` would have introduced a parallel path that we then have to keep in sync. Single-path wins on maintainability.
- **Why `__getattr__` lazy proxy for the optional symbols?** It removes the "is the symbol there or not?" ambiguity. The symbol is always *there*; constructing it without the optional deps raises `ImportError` with the install command. One test, one behavior, no environment-conditional `dir()`.
- **Why drop the advisory state-dict SHA sentinel entirely?** Even as an advisory, it tempts engineers to use it as a gate. Removing it eliminates the temptation and forces all regression conversations into behavior-snapshot terms — which is what the inquisitor's directive demands.

---

## Discussion summary

Round 2 raised 7 NEW critiques (3 against this proposal: A1-N1 huggingface_hub gap, A1-N2 transform fork, A1-N3 rollback path collision) plus 9 disagreement claims (all 9 adjudicated MISTAKEN-BY or RESOLVED-TOWARD). All three of *my* round-2 disagreements were MISTAKEN-BY=architect-1 — I re-read the round-2 proposals and confirm: (a) architect-2 did adopt the loader-v1/v2 split, (b) architect-2 did treat full-FT/PEFT as real implementation phases, (c) architect-3 did narrow to `ModuleNotFoundError`. The disputes do not exist in the round-2 record.

Round 3 closes A1-N1 (huggingface_hub added to extras + tested absent), A1-N2 (adopted architect-3's defaulted-kwarg design), A1-N3 (committed `scripts/rollback_dry_run.sh` with `mktemp -d` + `trap`).

## Consensus points

- **From architect-3 (NEW, RESOLVED-TOWARD by inquisitor):** adopted the defaulted-kwarg transform-injection design. `Custom_Crop` gains a single `image_size: int = 224` kwarg; the DINOv3 YAML config sets `image_size: 518` and the dataloader passes it through. No separate transform module. **Credit: architect-3.**
- **From architect-3:** `requirements-dinov3.txt` extras-file policy, reverse-DAG rollback chain, fail-loud key categorization, DDP out-of-scope, pinned `peft` upper bound — all retained.
- **From architect-2:** failing-test-first PR structure, repo-local `tests/_artifacts/`, subprocess-based absent-package testing, `MockAutoModel` fixture, golden-file `results_generation` schema snapshot — all retained.
- **With the inquisitor:** behavior snapshots are the sole regression gate (state-dict SHA fully removed); `huggingface_hub` joins the optional-dep contract; absent-class surface is `ImportError`-on-access via `__getattr__` proxy (one shape, tested exactly); no packaging migration; PEFT target validation is empty-error / partial-warning with `TaskType.FEATURE_EXTRACTION`.

## Disagreements

None — round-2 disputes adjudicated as MISTAKEN-BY=architect-1; transform-injection RESOLVED-TOWARD=architect-3 and adopted.

## Evolution

Round 2 → round 3 deltas:

- **Transform injection:** replaced `src/datasets/transforms.py::build_dinov3_eval_transform(image_size)` shim with a defaulted `image_size: int = 224` kwarg on `Custom_Crop`. ResNet bit-equality preserved by the default value. (Per inquisitor RESOLVED-TOWARD=architect-3; closes A1-N2.)
- **Optional-dependency contract:** added `huggingface_hub>=0.20,<1` to `requirements-dinov3.txt`. Default-install surface test now asserts all three packages (`transformers`, `peft`, `huggingface_hub`) absent from `sys.modules`. New test `test_huggingface_hub_missing_raises_actionable_error`. (Closes A1-N1.)
- **Rollback dry-run:** replaced ad-hoc absolute-path snippet with committed `scripts/rollback_dry_run.sh` that uses `mktemp -d` inside `.git/worktrees-tmp/` and a `trap` cleanup. Script ships in P0. (Closes A1-N3.)
- **Absent-class surface:** standardized on `__getattr__` lazy-proxy + `ImportError` on access for both `src.algorithms.DINOv3Linear` and `src.models.classification.DINOv3CustomWeights`. Symbols present in `__all__` and `dir(...)`; raise on construction with install instruction. (Per inquisitor directive — pick one shape; tests match exactly.)
- **PEFT target validation:** empty intersection now raises `ValueError`; partial intersection emits a `logging.warning` listing unmatched targets. `TaskType.FEATURE_EXTRACTION` retained. New test `test_partial_intersection_warns_listing_unmatched`. (Per directive.)
- **PEFT regression test:** logits behavior snapshot (`atol=1e-5`), not state-dict hash. (Per directive; closes A3-N2 spillover.)
- **State-dict SHA advisory sentinel removed entirely** from regression suite. Behavior snapshot is the sole gate. (Per directive — no state-dict hashes anywhere.)
- **No packaging migration:** CI uses `pip install -r requirements.txt` and `pip install -r requirements-dinov3.txt`, never `pip install -e .[test]`. No `[project.optional-dependencies]` changes.
- **Real SINCHI key evidence path:** moved from `tests/_artifacts/sinchi_keys_real.txt` to tracked-docs `docs/design/dinov3-integration/sinchi_key_layouts.md`. (Inquisitor directive on tracked evidence.)
- **`scripts/rollback_dry_run.sh` added to P0 deliverables** (was: PR-description snippet).
- **Disagreements section** reduced to "None" with rationale.

## NEW This Round

- **`scripts/rollback_dry_run.sh`** committed in P0 with `mktemp -d` worktree + `trap` cleanup (replaces ad-hoc absolute-path snippet).
- **`huggingface_hub` in `requirements-dinov3.txt`** and in the subprocess absent-env test triad.
- **`__getattr__` lazy-proxy pattern** at `src.algorithms.__init__` and `src.models.classification.__init__` as the single contracted absent-class surface (`ImportError` on access with install command in the message).
- **Empty-error / partial-warning split** in `_validate_peft_targets`, with the warning explicitly listing the unmatched target entries.
- **Removal of the advisory `state_dict.sha256` sentinel** — behavior snapshot is now the sole regression gate.

STATUS: NEW=5 DISAGREEMENTS=0
