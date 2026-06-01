# DINOv3 integration — phased plan (architect-1: phasing/MVP perspective) — round 2

## Executive overview

The plan still front-loads the cheapest, highest-leverage work and ships a minimum-viable linear-probe train→load→infer loop before layering full fine-tune, PEFT, and multi-format checkpoint compatibility. Round-1 critiques (47 flaws raised by the inquisitor; 9 specifically against this proposal) forced four structural changes:

1. **A real Phase 0 lands before any code edits** — test harness, hermetic mock-HF fixtures, behavior snapshots (predictions, not just `state_dict` SHA). The bug fix is no longer Phase 0; it is Phase 1 and is preceded by a *failing test artifact* committed in the same PR before the fix (per F-32, F-03).
2. **Optional-dependency policy converged on architect-3's `requirements-dinov3.txt` extras path** (F-04, F-34). Default `requirements.txt` is untouched. The default-install user never imports `transformers`.
3. **Loader split is preserved (Phase 5 / Phase 8) and hardened** with fail-loud key categorization and embedded checkpoint metadata (F-06, F-07, F-33).
4. **Gate-1 ships minimal "experimental" docs** (HF auth, GPU memory, image-size warning) instead of "no docs until last" (F-09, F-39).

The plan now has **10 phases** (was 8). Two new phases: **Phase 0** (scaffolding + behavior snapshot) and **Phase 3** (dependency wiring + transform-injection contract). Largest single PR is still ≤ ~400 LOC. Two pause-and-evaluate gates remain.

## Phase index (table)

| # | Phase name | Deliverable | Tests | Est. LOC | Revert unit |
|---|---|---|---|---|---|
| 0 | Test scaffolding + behavior snapshot baseline | `pytest.ini`, `tests/conftest.py`, `tests/fixtures/mini_dataset/`, `tests/fixtures/mock_hf.py`, `tests/regression/test_resnet_baseline.py`, `tests/regression/test_public_surface.py`, `tests/regression/test_default_config_keys.py`, `scripts/run_acceptance_gate0.sh`, `.github/workflows/ci.yml` (if absent) | 6 baseline tests, ResNet-18 + ResNet-50 prediction snapshots | ~250 | PR #0 |
| 1 | CustomWeights `num_layers` bug fix | Patch `resnet_base/custom_weights.py` (+1 defaulted kwarg); failing-test commit precedes fix commit *within* the PR | `tests/inference/test_custom_weights_num_layers.py` (3 tests) | ~30 + 80 test | PR #1 |
| 2 | Mixup utility port | `src/utils/mixup.py` (verbatim, with provenance docstring) | `tests/utils/test_mixup.py` (6 tests) | ~200 | PR #2 |
| 3 | Optional-dependency wiring + transform-injection contract | `requirements-dinov3.txt`, `docs/install.md` (install matrix), `src/datasets/transforms.py` (`build_dinov3_eval_transform(image_size)` shim), `tests/regression/test_default_install_surface.py` | 2 default-install tests, 1 dataset-transform test | ~120 | PR #3 |
| 4 | DINOv3 model + linear-probe algorithm (MVP core) — mixup wired off-by-default | `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py`, registry wires, `configs/config_dinov3_linear.yaml` | `tests/models/test_dinov3_classifier.py`, `tests/algorithms/test_dinov3_linear_smoke.py`, `tests/datasets/test_dinov3_transform_pipeline.py` | ~420 | PR #4 |
| 5 | Inference loader v1 (MD-Cls `.ckpt` only) + minimal Gate-1 docs | `src/models/classification/dinov3_base/{__init__,base_classifier,custom_weights}.py`, `docs/dinov3_experimental.md` (auth + memory + image-size warnings + scope statement) | `tests/inference/test_dinov3_custom_weights_mdckpt.py` | ~300 + 60 docs | PR #5 |
| — | **Gate 1: ship + collect feedback** (Phases 0–5 cover Steps 1, 2, 3, 4, 6-partial, 7-partial, 8) | — | — | — | — |
| 6 | Full fine-tune toggle | Extend `dinov3_linear.py`: dual param-groups, `unfreeze_backbone()` call, memory-warning emit | `tests/algorithms/test_dinov3_finetune_grads.py` | ~150 | PR #6 |
| 7 | PEFT (LoRA + PiSSA) with target-module validation + embedded checkpoint metadata | Extend `dinov3_linear.py::_wrap_with_peft` + `on_save_checkpoint`/`on_load_checkpoint` hooks for `peft_config` round-trip | `tests/algorithms/test_dinov3_peft_smoke.py`, `tests/algorithms/test_peft_checkpoint_roundtrip.py`, `tests/algorithms/test_peft_target_modules_validated.py` | ~280 | PR #7 |
| 8 | Inference loader v2 (SINCHI raw + SINCHI PEFT) with fail-loud key categorization + auto-rewrap | Extend `_extract_state_dict` + `_categorize_keys`; consume embedded `peft_config` metadata or accept explicit kwarg | `tests/inference/test_dinov3_custom_weights_sinchi.py`, `tests/inference/test_dinov3_custom_weights_peft.py`, `tests/inference/test_corrupt_ckpt_errors.py` | ~220 | PR #8 |
| — | **Gate 2: ship + collect feedback** | — | — | — | — |
| 9 | Config + docs consolidation (full) | `configs/config.yaml` (commented additions), sibling config files, `docs/usage.md`, `docs/error_taxonomy.md`, `README.md`, `CHANGELOG.md`, license/citation note | `tests/docs/test_sample_configs_parse.py`, `tests/docs/test_readme_snippets.py` | ~200 | PR #9 |

Cumulative net additions ≈ 2,000 lines (code + tests + docs); largest single PR ≈ 420 LOC.

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

**Ordering rationale (MVP lens, post-round-1):**

1. **Phase 0 — Scaffolding first.** Round 1 had me starting at the bug fix, but inquisitor F-32 was right: there is no test harness yet, so the bug fix would land with no regression budget. Phase 0 installs `pytest.ini`, a hermetic mini-dataset, a `MockAutoModel` fixture, behavior-snapshot tests (per F-03: prediction JSON not just `state_dict` SHA), and a public-surface freeze. Pure additions — touches no shipped code.
2. **Phase 1 — Bug fix in isolation** with a pre-fix failing test recorded as the first commit inside the same PR (F-20, F-32). One-line surface area, exercised by the Phase-0 harness.
3. **Phase 2 — Mixup port** (leaf utility, no callers).
4. **Phase 3 — Dependency wiring** *before* any module touches `transformers`. Introduces `requirements-dinov3.txt`, documents the three install profiles (default / dinov3-train / dinov3-peft), and ships the `build_dinov3_eval_transform(image_size)` shim in `src/datasets/transforms.py` — the **single transform-injection point** that resolves F-37. ResNet path imports nothing from this module.
5. **Phase 4 — MVP core.** `DINOv3Classifier` + `DINOv3Linear` (linear probe). Mixup is now **wired** in `training_step` but `use_mixup=False` is the default (per F-02). PEFT and full-FT remain `NotImplementedError`-guarded.
6. **Phase 5 — Loader v1 + minimal "experimental" docs.** Closes the MVP loop; ships `docs/dinov3_experimental.md` containing the F-09 warnings (HF auth, GPU memory, image-size, weights_only-trust note).
7. **Gate 1 — ship + ≥1 week feedback.**
8. **Phase 6 — Full FT.** Independent revert point (per F-12 / F-35 — keep the implementation commit here, not folded into earlier phases).
9. **Phase 7 — PEFT.** Adds target-module runtime validation (F-05) and `on_save_checkpoint` hook that embeds `model_name`, `image_size`, `peft_method`, `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`, normalization stats into `checkpoint["hyper_parameters"]["dinov3_metadata"]` (F-06, F-41, F-45).
10. **Phase 8 — Loader v2.** Hard pre-flight gate: real SINCHI `.pt` keys inspected via `torch.load(...).keys()` *before* the remap rule is implemented (F-15, F-29, F-38). Loader categorizes incoming keys into `{backbone, classifier, adapter, unknown}` and fails loud on `unknown` (F-07).
11. **Gate 2 — ship + ≥1 week feedback.**
12. **Phase 9 — Full docs/config consolidation**, license/citation (F-40), error taxonomy (F-46), full README. Minimal gate docs from Phase 5 are superseded.

**Dependency DAG (each arrow = "needs symbols/artifacts from"):**

```
P0 (scaffold) ── independent
   │
   ├──► P1 (bug fix)        — independent of all DINOv3 work
   ├──► P2 (mixup)          — independent
   └──► P3 (deps + transform shim)
                 │
                 └──► P4 (model + linear algorithm; uses P2 mixup + P3 transform)
                            │
                            └──► P5 (loader v1; uses P4 classifier)
                                       │
                                  — Gate 1 —
                                       │
                                       ├──► P6 (full FT; extends P4)
                                       │       │
                                       │       └──► P7 (PEFT; extends P4, P6)
                                       │                │
                                       │                └──► P8 (loader v2; needs P5 + P7 metadata)
                                       │                       │
                                       │                  — Gate 2 —
                                       │                       │
                                       └─────────────────────► P9 (docs)
```

**Independent-revert claim, now honest (per F-22):** P4, P6, P7 all edit `src/algorithms/dinov3_linear.py`. Reverting P6 alone while P7 remains is **not** safe — the DAG above is the **reverse-order rollback chain**. The rollback procedure in the risk section is updated accordingly. What *is* independent: P0, P1, P2, P3 are mutually independent and revertible in any order before P4 lands.

**Generic entry/exit pattern (every phase):**
- **Entry:** prior PRs on `main`; CI green; behavior snapshot from P0 reproduces.
- **Exit:** new tests pass; behavior snapshot still reproduces; public-surface freeze unviolated (additions allowed, removals/renames blocked); CHANGELOG entry; phase-tag commit (`phase-N-merged`) pushed for cheap bisect/rollback.

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

**Phase 2 — Port `noisy_same_class_mixup` + `efficient_batch_same_class_mixup`.**

- **Deliverable:** `src/utils/mixup.py` — verbatim port with provenance docstring naming the source path + commit SHA. RNG source documented (`np.random.beta`, `torch.randn` for noise) per F-40 reproducibility concern.
- **Entry:** P0 merged.
- **Exit:** both functions importable; no production callsite yet (latent until P4).
- **Tests** (`tests/utils/test_mixup.py`):
  - `test_noisy_same_class_mixup_shape_preserved` on `(B=8, D=16)`.
  - `test_noisy_same_class_mixup_labels_unchanged`.
  - `test_noisy_same_class_mixup_lambda_in_range` — monkey-patch `np.random.beta` to capture λ; assert clamp `[0.1, 0.9]`.
  - `test_noisy_same_class_mixup_singleton_class_passthrough`.
  - `test_noise_type_modes` — parametrize `{"gaussian","dropout","both"}` with `torch.manual_seed(0)` + `np.random.seed(0)`.
  - `test_efficient_batch_same_class_mixup_matches_naive` — `atol=1e-5`.
- **Per-phase regression gate:** `pytest tests/utils/ tests/regression/ -v` green.
- **Revert:** single PR; no downstream importers yet.

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

**Phase 4 — MVP core. Mixup wired-but-off-by-default; PEFT and full-FT guarded.**

- **Deliverable:**
  - **NEW** `src/models/dinov3.py` (~90 LOC) — `DINOv3Classifier`. **Lazy import** of `transformers` inside `__init__`; raises `ImportError("transformers>=4.40 required; install with: pip install -r requirements-dinov3.txt")` if absent (F-46 standardized message).
  - **EDIT** `src/models/__init__.py` — append `from .dinov3 import *` **inside a try/except ImportError** that **only catches transformers-related ImportError** (not arbitrary; per F-23). The except branch logs `logging.getLogger(__name__).debug(...)` and stores the failure reason in `src.models._dinov3_import_error` for diagnostic surfacing.
  - **NEW** `src/algorithms/dinov3_linear.py` (~220 LOC) — `DINOv3Linear` LightningModule:
    - `training_step` calls `noisy_same_class_mixup` *only if* `self.hparams.use_mixup` is True (F-02 — explicit wiring with a config key, not silently absent).
    - `freeze_backbone=False` raises `NotImplementedError("full fine-tune arrives in Phase 6")`.
    - `peft.enable=True` raises `NotImplementedError("PEFT arrives in Phase 7")`.
    - `on_save_checkpoint` already writes `checkpoint["hyper_parameters"]["dinov3_metadata"] = {"model_name": ..., "image_size": ..., "normalize_mean": ..., "normalize_std": ..., "use_mixup": ...}` — the metadata schema lands in P4 so P5 inference can read it; PEFT keys arrive in P7 via *additive* extension of the dict.
  - **EDIT** `src/algorithms/__init__.py` — same try/except pattern as models.
  - **NEW** `configs/config_dinov3_linear.yaml` — `use_mixup: False`, `peft.enable: False`, `freeze_backbone: True`, `image_size: 518`, `model_name: facebook/dinov3-vits16-pretrain-lvd1689m`.
- **Entry:** P3 merged (transform shim + extras file exist). **NO HF token required** in CI (F-01).
- **Exit:**
  - `python -c "from src.algorithms import DINOv3Linear"` succeeds with `transformers` installed.
  - With `transformers` **not** installed: `from src.algorithms import Plain` still works; only `DINOv3Linear` is absent.
  - Smoke training run with mocked backbone produces a `.ckpt` containing `hyper_parameters.dinov3_metadata`.
  - Phase-0 regression snapshot reproduces.
- **Tests:**
  - `tests/models/test_dinov3_classifier.py`:
    - `test_construct_with_mocked_automodel` (uses fixture from P0 `MockAutoModel`).
    - `test_encode_extracts_cls_token` shape `(B, hidden_size)`.
    - `test_attribute_naming_matches_md_cls_contract`: `.feature` + `.classifier`.
    - `test_transformers_missing_raises_actionable_error` — subprocess-level: spawn a subprocess with `PYTHONPATH` and import hook that hides `transformers`, assert constructor raises with install-instruction message (F-17 — subprocess, not `sys.modules` trick).
  - `tests/algorithms/test_dinov3_linear_smoke.py`:
    - `test_imports_when_transformers_present`.
    - `test_finetune_path_blocked`.
    - `test_peft_path_blocked`.
    - `test_mixup_wired_when_enabled` — set `use_mixup=True`, spy on `noisy_same_class_mixup`, run one `training_step`, assert called exactly once per batch.
    - `test_mixup_off_by_default` — assert spy never called when `use_mixup=False`.
    - `test_checkpoint_contains_dinov3_metadata` — save ckpt, load raw, assert `hyper_parameters.dinov3_metadata` keys are present.
    - `test_one_step_training_grad_flows_to_classifier_only` (frozen backbone).
  - `tests/datasets/test_dinov3_transform_pipeline.py` (resolves F-08, F-16, F-37):
    - `test_dinov3_dataloader_emits_518` — instantiate `Custom_Crop` with `image_size=518` via the P3 transform shim; assert batch tensor has shape `(B, 3, 518, 518)`.
    - `test_resnet_dataloader_unchanged` — instantiate `Custom_Crop` with the default ResNet config; assert batch tensor still `(B, 3, 224, 224)` and that the transform is bit-equal to a pre-P3 snapshot.
- **Per-phase regression gate:** `pytest tests/models tests/algorithms tests/datasets tests/regression/ -v -m "not gpu and not real_hf"` green.
- **Revert:** single PR.

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

**Phase 6 — Lift the `NotImplementedError`; dual param-groups; emit memory warning.**

- **Deliverable:** `src/algorithms/dinov3_linear.py` edits — `configure_optimizers` returns AdamW with two param groups (`lr_feature`, `lr_classifier`); `setup()` calls `self.net.unfreeze_backbone()` when `freeze_backbone=False`; on first call emit `logging.warning("Full fine-tune of DINOv3 ViT requires ~12GB at 518² fp16, batch_size=4. Consider precision='16-mixed'.")` (resolves F-28 — calibrated, conservative warning instead of citing unverified numbers).
- **Entry:** P5 merged; Gate-1 feedback period (≥1 week) complete; no blocking issue reports.
- **Exit:**
  - `freeze_backbone: False` produces non-zero gradient norms on `self.net.feature.*`.
  - Linear-probe regression: with `freeze_backbone: True`, loss curve matches P4 within `atol=1e-4` under fixed seed (deterministic mode via `torch.use_deterministic_algorithms(True)` in test).
- **Tests** (`tests/algorithms/test_dinov3_finetune_grads.py`):
  - `test_unfreeze_makes_backbone_trainable`.
  - `test_param_groups_split` — exactly 2 groups; correct LRs.
  - `test_backbone_grad_flows` — at least one `feature.*` param has non-zero `.grad`.
  - `test_frozen_path_unchanged` — parametrized: `freeze_backbone=True` matches P4 behavior.
  - `test_memory_warning_emitted_once`.
- **Per-phase regression gate:** P0 snapshot reproduces; P4 linear-probe smoke still passes.
- **Revert:** restores `NotImplementedError`. **No new files.** Safe iff P7 not merged.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

**Phase 7 — Lift PEFT guard; runtime target-module validation; embed `peft_config` in checkpoint.**

- **Deliverable:**
  - **EDIT** `src/algorithms/dinov3_linear.py`:
    - `_wrap_with_peft()` — lazy `from peft import LoraConfig, get_peft_model, TaskType`; `TaskType.FEATURE_EXTRACTION` (verified per inquisitor's external check); wrap `self.net.feature` only (classifier remains plain `nn.Linear`).
    - **NEW** `_validate_peft_targets(target_modules)` (F-05): iterate `self.net.feature.named_modules()`; assert at least one module name ends with one of `target_modules`; if zero matches, raise `ValueError("PEFT target_modules {target_modules} match no modules in {model_name}. Available leaf-module suffixes: {sorted(set(n.split('.')[-1] for n,_ in feature.named_modules()))[:30]}")` (F-46 standardized).
    - `on_save_checkpoint(self, checkpoint)` — extend `dinov3_metadata` with `peft_method`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`, `init_lora_weights` (F-06, F-41).
    - `on_load_checkpoint(self, checkpoint)` — re-wrap via stored metadata if present; emit `logging.warning` if checkpoint advertises PEFT but instance config disables it.
  - **EDIT** `requirements-dinov3.txt` — add `peft>=0.10.0,<0.13.0` (upper bound per F-04 stability).
- **Entry:** P6 merged.
- **Exit:**
  - `peft.enable=True, method=lora` → trainable params `<5%` of total (assert).
  - `method=pissa` → `init_lora_weights="pissa_niter_4"` set; no unrecognized-init warnings captured.
  - Round-trip: save 2-step PEFT ckpt → `DINOv3Linear.load_from_checkpoint(...)` → adapter weights bit-equal; metadata round-tripped.
  - Lazy-import: subprocess test asserts `enable=False` works without `peft` installed; `enable=True` raises standardized `ImportError`.
  - **Target-module validation:** mis-typed target list raises actionable `ValueError`.
- **Tests:**
  - `tests/algorithms/test_dinov3_peft_smoke.py`:
    - `test_lora_wrap_reduces_trainable_count` (<0.1× un-wrapped).
    - `test_pissa_method_accepted` (spy on `LoraConfig`).
    - `test_peft_lazy_import` (subprocess; F-17).
    - `test_classifier_outside_wrap`.
  - `tests/algorithms/test_peft_target_modules_validated.py` (NEW; F-05):
    - `test_invalid_target_raises_actionable_error` — `target_modules=["nonexistent_proj"]` → `ValueError` whose message includes ≥1 valid candidate suffix.
    - `test_default_targets_match_dinov3_vit` — with `MockAutoModel` exposing the SINCHI-default modules, validator passes.
  - `tests/algorithms/test_peft_checkpoint_roundtrip.py`:
    - Train 2 steps → `trainer.save_checkpoint("tests/_artifacts/test_peft.ckpt")` (repo-local; F-14) → reload → assert classifier + LoRA adapter weights bit-equal and `dinov3_metadata.peft_method == "lora"`.
- **Per-phase regression gate:** P0 snapshot reproduces; P6 grad-flow test still passes.
- **Revert:** removes `_wrap_with_peft` body + validation helper + `peft` line in `requirements-dinov3.txt`; reinstates `NotImplementedError`. Safe iff P8 not merged (P8 reads PEFT metadata).

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

**Split across P5 (MD-Cls only) and P8 (SINCHI raw + SINCHI PEFT). Now hardened with fail-loud key categorization (F-07) and metadata-driven auto-rewrap (F-06).**

### Phase 5 — Loader v1: MD-Cls `.ckpt` only

- **Deliverable:**
  - **NEW** `src/models/classification/dinov3_base/__init__.py`.
  - **NEW** `src/models/classification/dinov3_base/base_classifier.py` (~120 LOC) — `DINOv3BaseInference`:
    - `_extract_state_dict(state)` — handles only `{"state_dict": {"net.X": ...}}` flavour; SINCHI flavours route to `_raise_for_sinchi_format(state)` which raises `NotImplementedError("SINCHI flavours land in Phase 8")`.
    - `_categorize_keys(state_dict)` (NEW per F-07) — returns `{"backbone": [...], "classifier": [...], "adapter": [...], "unknown": [...]}`. Raises `ValueError` if `unknown` non-empty.
    - Reads `checkpoint.get("hyper_parameters", {}).get("dinov3_metadata", {})` to retrieve `image_size` and `model_name`; if absent, falls back to constructor kwargs with `logging.warning("checkpoint lacks dinov3_metadata; using kwargs")`.
    - Documents the `weights_only=False` security trade-off in module docstring (F-44).
  - **NEW** `src/models/classification/dinov3_base/custom_weights.py` (~80 LOC) — `DINOv3CustomWeights`.
  - **EDIT** `src/models/classification/__init__.py` — `from .dinov3_base import *` guarded by **specific** `ImportError` (only catches `transformers` missing; per F-23). Re-raises any other ImportError immediately and logs at WARNING level.
  - **NEW** `docs/dinov3_experimental.md` (~60 LOC) shipping with Gate 1 (per F-09):
    - "EXPERIMENTAL — API may change before Phase 9."
    - HF auth instructions (`huggingface-cli login`).
    - GPU memory rule of thumb (vits16: ~3 GB fp16 bs=4; vitb16: ~7 GB).
    - Image-size warning (518 ≠ ResNet 224; cannot mix configs).
    - Checkpoint security note (`weights_only=False`; only load trusted ckpts).
    - License/citation note for DINOv3 weights (F-40).
- **Entry:** P4 merged; at least one MD-Cls `.ckpt` exists in `tests/_artifacts/`.
- **Exit:**
  - `DINOv3CustomWeights(weights=<md_cls_ckpt>, class_names=[...])` loads with `missing == [] and unexpected == []`.
  - `forward()` on `(1, 3, 518, 518)` → `(1, num_cls)`.
  - `results_generation` output schema matches existing `CustomWeights` exactly (golden-file snapshot per architect-2's earlier suggestion, adopted).
- **Tests** (`tests/inference/test_dinov3_custom_weights_mdckpt.py`):
  - `test_load_md_cls_ckpt`.
  - `test_load_uses_embedded_metadata` (image_size + model_name come from ckpt).
  - `test_load_without_metadata_warns_and_uses_kwargs`.
  - `test_categorize_keys_unknown_raises` — synthetic ckpt with `garbage.weight` → `ValueError` listing the unknown key.
  - `test_results_generation_schema` — golden-file snapshot.
  - `test_sinchi_flavour_raises_phase8_message`.
  - `test_pyw_style_import_path`.

### Phase 8 — Loader v2: SINCHI raw + SINCHI PEFT

- **Deliverable:**
  - **EDIT** `base_classifier.py`:
    - Extend `_extract_state_dict` for `{"model_state_dict": {"backbone.X": ..., "classifier.X": ...}}` (raw) and PEFT-wrapped delta keys.
    - Extend `_categorize_keys` to recognize `backbone.*` and adapter (`*.lora_A.weight`, `*.lora_B.weight`) patterns.
    - Accept `peft_config=` kwarg **and** auto-load from `state.get("peft_config")` if present (SINCHI's own convention).
- **Entry — HARD pre-flight gate (F-15, F-29, F-38):**
  - At least one real SINCHI raw `.pt` **and** one real SINCHI PEFT `.pt` checkpoint obtained.
  - `python scripts/inspect_sinchi_keys.py <path>` run; its output (top-level keys + first 20 inner keys) committed to `docs/design/dinov3-integration/sinchi_key_inventory.md`. Remap rule is **written against verified keys**, not the research report's inference.
  - If verified keys diverge from the report (`backbone.*`), update the remap and the unit test fixtures *before* writing implementation code.
- **Exit:**
  - Both SINCHI flavours load. For raw: `unexpected == []`, `missing == []` modulo a documented `hidden_size`-mismatch error if user supplies wrong `model_name`.
  - For PEFT: with auto-rewrap, `missing == []` for adapter keys. Without metadata or kwarg, raises actionable `ValueError("checkpoint contains LoRA adapter keys but no peft_config available; pass peft_config= or use a checkpoint with embedded metadata")` (F-07, F-46).
- **Tests:**
  - `tests/inference/test_dinov3_custom_weights_sinchi.py`:
    - `test_load_sinchi_raw_real_fixture` — runs only if `SINCHI_RAW_CKPT` env var points at a real file (CI-skipped; pass-required for Gate-2 ship).
    - `test_load_sinchi_raw_synthetic` — synthetic `{"model_state_dict": {"backbone.encoder.layer.0.weight": ...}}`; assert remap to `feature.encoder.layer.0.weight`.
    - `test_hidden_size_mismatch_fails_loud` — `model_name` mismatched → `RuntimeError` with the dimension diff in message.
  - `tests/inference/test_dinov3_custom_weights_peft.py`:
    - `test_load_sinchi_peft_with_embedded_config` (auto-rewrap path).
    - `test_load_sinchi_peft_with_explicit_kwarg`.
    - `test_peft_keys_without_config_raises` — assert F-07-style hard error, not silent `missing` list.
  - `tests/inference/test_corrupt_ckpt_errors.py` (F-44):
    - `test_truncated_file_raises_actionable_error`.
    - `test_wrong_state_dict_root_raises_with_key_listing`.

**Revert:** Each sub-phase its own PR. P8 revert restores simpler `_extract_state_dict`. P5 revert removes `dinov3_base/` subpackage entirely. P5 cannot be reverted while P8 is on `main` (revert P8 first).

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

**Split: minimal "experimental" docs at P5 (Gate 1); full consolidation at P9.**

### Minimal Gate-1 docs (delivered in P5)
- `docs/dinov3_experimental.md` (see P5 Deliverable above).

### Full consolidation — Phase 9

- **Deliverable:**
  - **EDIT** `configs/config.yaml` — append the ~25-line **commented** `dinov3:` / `training:` / `label_smoothing` block (research §Step 5). The uncommented section is byte-equal to pre-P9 — proved by `test_default_config_unchanged_in_behavior_keys`.
  - **NEW** `configs/config_dinov3_finetune.yaml`, `configs/config_dinov3_lora.yaml`, `configs/config_dinov3_pissa.yaml`.
  - **EDIT** `docs/usage.md` — full "Fine-tuning with DINOv3" section: capability matrix, config keys, install profile selection (default vs `pip install -r requirements-dinov3.txt`), HF auth, GPU memory, CPU/MPS guidance (F-42), distributed/DDP **out-of-scope** statement (F-43 — explicitly deferred), reproducibility recipe (seeded workers, deterministic transforms; F-45).
  - **NEW** `docs/error_taxonomy.md` (F-46) — table of every standardized error message: HF auth failure, missing deps, corrupt checkpoint, hidden-size mismatch, missing peft_config, no matching LoRA targets, CPU OOM, unknown ckpt keys. Each row: `Error class | Triggered when | Message template | Remedy`.
  - **EDIT** `README.md` — capability bullet; model-zoo example block; **migration note for the P1 bug fix** (F-47) prominent at top.
  - **EDIT** `CHANGELOG.md` — summarise P0–P9.
  - **Supersede** `docs/dinov3_experimental.md` → either delete or replace with a redirect stub to `docs/usage.md#dinov3`.
- **Entry:** P8 merged; Gate-2 ≥1-week feedback collected.
- **Exit:**
  - `python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"` succeeds.
  - Every sample `configs/config_dinov3_*.yaml` parses and required keys present.
  - All README/usage snippets execute as doctests under mocked HF.
- **Tests:**
  - `tests/docs/test_sample_configs_parse.py` — parametrized.
  - `tests/docs/test_default_config_unchanged_in_behavior_keys` — compares normalized `dict` (per F-19; not raw YAML dump) against P0 snapshot.
  - `tests/docs/test_readme_snippets.py` — execute fenced ```python blocks tagged `# doctest`.
- **Revert:** single PR; removes appended docs and sample configs; default `config.yaml` returns to pre-P9.

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

**Phase 1 — Sequenced *after* P0 scaffolding (per F-32). PR contains a failing-test commit before the fix commit (per F-20).**

- **Commit structure inside PR #1:**
  1. Commit A: add `tests/inference/test_custom_weights_num_layers.py::test_resnet18_loads_with_explicit_num_layers` — expected to FAIL. CI annotation: `XFAIL/EXPECTED-FAIL`; the commit message references `git log --oneline -1` SHA as the captured pre-fix artifact.
  2. Commit B: patch `src/models/classification/resnet_base/custom_weights.py` — widen signature to `def __init__(self, weights=None, class_names=None, device="cpu", num_layers=50)`; pass `num_layers=num_layers` (instead of hard-coded `50`).
  3. Commit C: flip the test from `@pytest.mark.xfail` to passing; CI now requires it to pass.
- **Entry:** P0 merged; working tree clean.
- **Exit:**
  - `CustomWeights(weights=<resnet18.ckpt>, class_names=[...], num_layers=18)` loads cleanly.
  - `CustomWeights(weights=<resnet50.ckpt>, class_names=[...])` (no kwarg) still loads cleanly.
  - All existing tests pass; P0 snapshot reproduces.
- **Tests** (3 tests as listed above).
- **Revert:** restores hard-coded `50`. Safe — DINOv3 has its own loader.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

**Strategy revised post-F-03:** the round-1 plan leaned on a brittle `state_dict` SHA. Round 2 layers four defenses, weighted toward *behavior* not *bytes*:

1. **Behavior snapshot (primary, introduced P0).** `tests/fixtures/regression/`:
   - `resnet18_baseline.predictions.json` — class_id + per-class confidence (rounded to 4 decimal places) for the 10 mini-dataset images. **Compared with tolerance** (`atol=1e-3` on confidences, exact-match on class_id).
   - `resnet18_baseline.config.yaml` — frozen YAML copy.
   - `resnet18_baseline.metadata.json` — torch/lightning/transformers versions, CUDA flag, deterministic-mode flag. Mismatch → test issues a `pytest.warns` but does not fail (per F-03 cross-version brittleness).
   - `resnet18_baseline.state_dict.sha256` — kept as a *secondary* sentinel; mismatch is a WARNING, not a failure, unless `STRICT_REGRESSION=1` env var is set.
2. **Default-config invariants (every PR).** `tests/regression/test_default_config_keys.py`:
   - Load `configs/config.yaml`; assert normalized dict (sorted keys, no Munch) equals frozen snapshot. Comments do not affect this (F-19).
3. **Public-surface freeze.** `tests/regression/test_public_surface.py`:
   - Snapshot `sorted(dir(src.algorithms))`, `sorted(dir(src.models))`, `sorted(dir(src.models.classification))` at P0. Each phase may **add**, not remove or rename.
4. **Byte-equality only for files that must never change** (per F-24 — narrow the brittle list):
   - `main.py`, `src/algorithms/plain.py`, `src/models/plain_resnet.py`, `src/datasets/custom.py`. SHA tracked in `tests/regression/critical_files.sha256`. Anything else uses behavior tests.
5. **Default-install surface freeze (P3).** `tests/regression/test_default_install_surface.py` runs in a subprocess with **only** `requirements.txt` installed (CI uses a separate `pip install -r requirements.txt` venv): `from src.algorithms import Plain` succeeds; `from src.algorithms import DINOv3Linear` raises `ImportError`; no `transformers` in `sys.modules` after `import src`.

**Per-phase regression gate:** Phase N's PR may not merge if `pytest tests/regression/` reports any failure under default mode.

**Hash regeneration policy (F-03):** behavior-snapshot regen requires a labelled `regression-baseline-bump` PR with reviewer sign-off and a written justification. None of P0–P9 should require this; bumps are an alarm bell.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Test pyramid (must-pass vs nice-to-have, per F-36)

| Layer | What | Must-pass in CI? | Where | Wall-time |
|---|---|---|---|---|
| Unit (must) | Pure-Python (mixup math, key-remap, key-categorization) | YES | `tests/utils/`, `tests/inference/` | <30 s |
| Integration (must) | One-step `training_step` with `MockAutoModel`; one-shot `load_state_dict` w/ synthetic ckpts | YES | `tests/algorithms/`, `tests/inference/`, `tests/models/`, `tests/datasets/` | <2 min |
| Regression (must) | Behavior snapshots, default-config, public surface, critical-file bytes, default-install subprocess | YES | `tests/regression/` | <3 min |
| Smoke-GPU (nice) | 1-epoch real training | NO (nightly only) | `tests/smoke/`, `@pytest.mark.gpu` | local |
| Smoke-HF (nice) | Real HF download + inference | NO (nightly only) | `@pytest.mark.real_hf` | local |
| Real-SINCHI gate (must before P8 ship) | Load real `.pt` files | YES at Gate 2 only | `tests/inference/test_*sinchi*::*real_fixture` | local |
| Acceptance (manual) | 10-item checklist | manual at gates | `scripts/run_acceptance.sh` | varies |

Coverage thresholds are **not** a merge gate (F-18, F-36) — only the must-pass layers gate merge. Coverage reports are generated but advisory.

### Fixtures
- `tests/fixtures/mini_dataset/` — 6 images × 2 classes; cropped CSVs pre-generated.
- `tests/fixtures/ckpts/`:
  - `resnet18_mini.ckpt`, `resnet50_mini.ckpt` (P0)
  - `dinov3_linear_mini.ckpt` (P4 → P5)
  - `synthetic_sinchi_raw.pt`, `synthetic_sinchi_peft.pt` (P8 in-test via `torch.save`)
  - **Real SINCHI** ckpt paths supplied via `SINCHI_RAW_CKPT` / `SINCHI_PEFT_CKPT` env vars at Gate 2 (F-15)
- `tests/conftest.py` — `MockAutoModel` returning deterministic embeddings, `pytest --basetemp=tests/_artifacts` configured.
- **Repo-local temp policy (F-14):** `pytest.ini` sets `tmp_path_retention_policy = none` and `--basetemp=tests/_artifacts`. **Never** `/tmp`. `tests/_artifacts/` gitignored.

### Per-phase test invocation map

| Phase | Command |
|---|---|
| 0 | `pytest tests/regression/ tests/fixtures/ -v` |
| 1 | `pytest tests/inference/test_custom_weights_num_layers.py tests/regression/ -v` |
| 2 | `pytest tests/utils/ tests/regression/ -v` |
| 3 | `pytest tests/regression/test_default_install_surface.py tests/datasets/ tests/regression/ -v` |
| 4 | `pytest tests/models tests/algorithms tests/datasets tests/regression/ -v -m "not gpu and not real_hf"` |
| 5 | `pytest tests/inference/test_dinov3_custom_weights_mdckpt.py tests/regression/ -v` |
| 6 | `pytest tests/algorithms/test_dinov3_finetune_grads.py tests/regression/ -v` |
| 7 | `pytest tests/algorithms/test_dinov3_peft_*.py tests/regression/ -v` |
| 8 | `pytest tests/inference/test_dinov3_custom_weights_sinchi.py tests/inference/test_dinov3_custom_weights_peft.py tests/inference/test_corrupt_ckpt_errors.py tests/regression/ -v` + **real-SINCHI fixture** |
| 9 | `pytest tests/docs/ tests/regression/ -v` |
| Pre-Gate ship | `bash scripts/run_acceptance.sh` (manual GPU) |

### CI vs local vs nightly

- **CI default (no GPU, no HF token, no peft):** must-pass layers only. <6 min. Two install matrices (per F-26, staged): from P3 onward, CI runs `requirements.txt`-only and `requirements.txt + requirements-dinov3.txt`.
- **Local (full):** entire pyramid.
- **Nightly GPU runner:** `-m "gpu or real_hf"`.

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

`scripts/run_acceptance.sh` chains commands; fails on first non-zero exit. Reduced variant `scripts/run_acceptance_gate1.sh` ships in P5.

| # | Research check | Concrete command | Pass criterion | First satisfied | CI / local / GPU / real-artifact |
|---|---|---|---|---|---|
| 1 | Existing ResNet path bit-identical | `pytest tests/regression/ -v` | All green; behavior snapshot matches with `atol=1e-3` | P0 | CI |
| 2 | Linear-probe trains 1 epoch | `python main.py --config configs/config_dinov3_linear.yaml` (epochs=1) | `.ckpt` exists; finite loss; metadata embedded | P4 | local GPU |
| 3 | Full FT produces backbone grads | `pytest tests/algorithms/test_dinov3_finetune_grads.py::test_backbone_grad_flows` | `.grad` non-zero | P6 | CI (mocked) + local GPU |
| 4 | Mixup toggle changes loss curve | `python scripts/compare_loss_curves.py --with-mixup --without-mixup` | Curves differ; no shape errors | P4 (wired off) + P9 (documented enable) | local GPU |
| 5 | LoRA + PiSSA train | `python main.py --config configs/config_dinov3_lora.yaml`; `..._pissa.yaml` | Train; trainable params <5% | P7 | local GPU |
| 6 | Inference — MD-Cls ckpt | `python scripts/infer_demo.py --ckpt <md_cls.ckpt>` | No missing/unexpected | P5 | CI (mocked) + local |
| 7 | Inference — SINCHI raw | `python scripts/infer_demo.py --ckpt <sinchi_raw.pt>` | Remap applied; loads | P8 | **real-artifact gate** |
| 8 | Inference — SINCHI PEFT | `python scripts/infer_demo.py --ckpt <sinchi_peft.pt>` | Auto-rewrap from embedded config or explicit kwarg | P8 | **real-artifact gate** |
| 9 | CustomWeights regression | `pytest tests/inference/test_custom_weights_num_layers.py` | Both ResNet-18 and -50 load | P1 | CI |
| 10 | Lazy-import (subprocess) | `tests/regression/test_default_install_surface.py` | Default-install user has no `transformers`; constructing DINOv3 in default env raises actionable ImportError | P3 (default surface) + P4 (transformers) + P7 (peft) | CI |
| **+11** | Transform-size proof (NEW; F-08, F-30, F-37) | `pytest tests/datasets/test_dinov3_transform_pipeline.py` | DINOv3 path emits 518; ResNet path emits 224 byte-equal to pre-P3 | P4 | CI |
| **+12** | Real SINCHI key inspection (NEW; F-15, F-29, F-38) | `python scripts/inspect_sinchi_keys.py $SINCHI_RAW_CKPT $SINCHI_PEFT_CKPT` | Output committed to `sinchi_key_inventory.md`; reviewer sign-off recorded | P8 pre-flight | real-artifact gate |

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

### Risk register

| ID | Risk | Phase exposed | L | I | Mitigation |
|---|---|---|---|---|---|
| R1 | HF gating: tests/CI without token cannot fetch | P4 | High | Med | `MockAutoModel`; `@pytest.mark.real_hf` auto-skipped; docs in P5/P9 |
| R2 | GPU OOM at 518² ViT-B fp32 | P4 | Med | High | Docs (P5 + P9) recommend `precision="16-mixed"` + small batch; P6 emits memory warning; smoke uses tiny mocked backbone |
| R3 | `transformers` version drift | P4 | Low | Med | Pin `>=4.40.0,<5` in `requirements-dinov3.txt` |
| R4 | `peft` optional dep absent breaks unrelated imports | P7 | Low | High | Lazy import inside `_wrap_with_peft`; subprocess test `test_peft_lazy_import`; default-install surface test in CI |
| R5 | SINCHI checkpoint inner keys inferred | P8 | Med-High | Med | **Hard pre-flight gate**: real ckpts inspected; inventory committed; tests written against verified keys |
| R6 | Lightning ckpt-format drift | P4, P7 | Low | Med | Pinned versions; regression suite catches behavior drift |
| R7 | `weights_only=False` security | P5, P8 | Low | Med | Standardized doc warning in `docs/dinov3_experimental.md` (P5) and `docs/usage.md` (P9); negative test in `test_corrupt_ckpt_errors.py` |
| R8 | P1 bug fix changes behavior for caller relying on hard-coded `50` | P1 | Very Low | Low | Default kwarg = `50`; signature test |
| R9 | Surface-diff regression | P4 | Low | Low | Public-surface freeze; additions allowed |
| R10 | Embedding-dim mismatch on wrong `model_name` | P5, P8 | Med | Low | Fail-loud `RuntimeError` with dim diff in message |
| R11 | Gate-1 feedback invalidates P6–P8 design | Gate 1 | Med | Med | Built-in by design |
| **R12 (NEW; F-05)** | PEFT target_modules don't match any module name | P7 | Med | High | Runtime `_validate_peft_targets` raises actionable `ValueError` listing valid candidates |
| **R13 (NEW; F-42)** | DINOv3 on CPU/MPS — unsupported `precision="16-mixed"`, slow | P4 | Med | Low | Docs note (P5 + P9); algorithm auto-falls-back to fp32 with WARNING when CUDA unavailable |
| **R14 (NEW; F-40)** | DINOv3 weight license/citation obligations | P5 | Med | Low | License note in `docs/dinov3_experimental.md` (P5); full citation in `docs/usage.md` (P9) |
| **R15 (NEW; F-43)** | DDP/multi-GPU untested | All training phases | Low | Low | Explicitly **out-of-scope**; documented in P9 |
| **R16 (NEW; F-47)** | P1 bug fix discoverability — silent migration | P1 | Low | Med | Prominent CHANGELOG entry at P1 merge; README migration note at top in P9 |

### Per-phase rollback procedure

Every phase = one squash-merged PR + a `phase-N-merged` annotated tag.

```bash
git log --oneline --merges --grep="Phase N" -1
git revert -m 1 <merge-sha>
git push origin main
pytest tests/regression/ -v
```

**Phase dependency rollback chain (honest, per F-22 — not "independent"):**

```
P9 → P8 → P7 → P6 → P5 → P4 → P3 → P2 → P1 → P0
```

To revert P_k, revert P_{k+1}..P_n first **in reverse order**. The only exceptions: P1, P2, P3 are mutually independent (none import another) and may be reverted in any order *before* P4 is merged.

### Operationalized rollback dry-run (F-25)

For each phase, the PR description must include a paste of:
```bash
# Dry-run rollback in a scratch worktree
git worktree add /shared/home/aahernandez/MegaDetector-Classifier-rollback-test HEAD~1
cd /shared/home/aahernandez/MegaDetector-Classifier-rollback-test
pytest tests/regression/ -v
cd - && git worktree remove /shared/home/aahernandez/MegaDetector-Classifier-rollback-test
```
(Absolute repo-local path; no `/tmp`; per environment rules.)

### Gate-level rollback

Gate-1 fundamental flaw → revert P5 → P4 → P3 → P2 → P1 → P0 (six commands). Repo returns to pre-DINOv3 state. Because no symbols pre-existing in P0 are renamed/removed at any phase, no downstream code breaks during rollback.

---

## Rationale (phasing/MVP lens)

- **Why P0 *before* P1 now?** F-32 was right: bug fix without a regression harness is unfalsifiable. Adding the harness as P0 costs ~250 LOC of pure test infrastructure; every subsequent phase amortizes that cost.
- **Why a separate P3 (deps + transform shim)?** F-04 / F-34 forced a single dependency policy decision. Doing it in its own phase means the transform-injection contract (resolving F-37) is decoupled from any model code — the shim is testable in isolation and gives P4 a clean import target.
- **Why split inference loader P5/P8?** Same MVP reasoning as round 1, now reinforced by F-33: MD-Cls is the only checkpoint flavour we fully control (write in P4, read in P5). SINCHI flavours block on real-binary pre-flight (F-15, F-29, F-38) — making P5 wait would stall the MVP unnecessarily.
- **Why mixup wired-but-off in P4 (not deferred)?** F-02 was right that the round-1 wording ("port now, wire later") hid the seam. Now `use_mixup: bool = False` is in the config schema from day one; enabling is a config flip, not a code change.
- **Why minimal docs at P5, full at P9?** F-09 / F-39: Gate-1 users hit HF auth, GPU memory, and image-size issues *before* any docs would exist. The minimal `docs/dinov3_experimental.md` blunts that. P9 then ships the polished surface once we've shipped.
- **Why is target-module validation worth its own test file?** F-05 + inquisitor's HF source check showed the SINCHI list is plausible but not universal across DINOv3 variants. Runtime `named_modules()` validation makes the failure mode visible and recoverable rather than silently producing a no-op LoRA wrap.
- **Why embed `peft_config` in the checkpoint (F-06, F-41)?** Without it, inference depends on users carrying around sidecar JSON — a UX cliff. Embedding into Lightning's `hyper_parameters` is two lines of code and turns the loader from "fragile" to "auto-configuring".

---

## Discussion summary

The inquisitor raised 9 flaws directly at this proposal (F-01..F-09) and 8 cross-architect divergences touching phasing (F-32..F-39), plus 8 missing perspectives (F-40..F-47). I incorporated **all 9 direct flaws** and **all 8 divergences**. Of the 8 missing perspectives, I incorporated 6 explicitly (F-40 license, F-41 embedded metadata, F-42 CPU/MPS, F-44 weights_only security, F-45 reproducibility, F-46 error taxonomy, F-47 migration notes) and explicitly scoped-out 1 (F-43 distributed training — deferred, documented in P9). I did not push back on any inquisitor critique.

Against the other architects:
- **architect-2 (testing):** better on "tests fail first" workflow (F-20), `tests/_artifacts/` repo-local temp path (F-14), subprocess-based lazy-import testing (F-17), golden-file output snapshots — all adopted. Worse on inference phasing (loader-last bundles too much; F-11) and on pre-implementing PEFT/full-FT branches in earlier phases (F-12, F-13) — I rejected.
- **architect-3 (risk/backwards-compat):** better on `requirements-dinov3.txt` extras policy (F-21, F-34), reverse-DAG rollback (F-22), explicit dependency-rollback policy, fail-loud loader checks (F-33) — all adopted. Worse on the `try/except: pass` import-guard pattern (F-23 — I narrow it to specific ImportError catch) and Phase-7 loader-after-PEFT ordering (F-27 — I keep my split P5/P8).

## Consensus points

- **From architect-2:** adopted "tests fail first" pre-fix failing-test commit (P1 commit structure); repo-local `tests/_artifacts/` with pytest `--basetemp` (F-14); subprocess-based absent-package testing (F-17); `MockAutoModel` fixture pattern; golden-file `results_generation` schema snapshot (P5).
- **From architect-3:** adopted `requirements-dinov3.txt` extras file with documented install matrix (F-04, F-34); reverse-DAG rollback chain (F-22); fail-loud key categorization in loader (F-33); explicit out-of-scope statement for DDP (F-43); pinned upper bound on `peft` version.
- **With both architects-2 and -3:** ResNet path must remain bit-identical (default config); `main.py` is not touched at any phase; lazy imports for heavy deps; mixup is its own phase before the algorithm.
- **With the inquisitor:** behavior snapshots > byte hashes; target-module runtime validation is mandatory; PEFT metadata embedded in checkpoints; real SINCHI ckpt inspection is a hard pre-flight gate, not a "nice to have"; minimal docs ship with Gate 1.

## Disagreements

- **With architect-2 (F-11):** I keep the inference loader split (P5 MD-Cls only at Gate 1, P8 SINCHI at Gate 2) rather than bundling all loader work into a final phase. Reasoning: bundling delays the MVP train→load→infer loop until after PEFT lands, and couples MD-Cls (which we can fully test today) with SINCHI (which requires real ckpts we may not have). Splitting costs one extra PR and one extra `__init__.py` edit; bundling costs the MVP gate.
- **With architect-2 (F-12, F-13):** I keep full-FT (P6) and PEFT (P7) as separate implementation phases, not "test-only" verifications of branches pre-coded in P4. Reasoning: the scope ledger names `full-finetune-phase` and `peft-phase` as separate items; folding their implementation into P4 violates the independent-revert claim *for those specific units* (even acknowledging the dependency chain) and inflates P4 well past the 500-LOC ceiling.
- **With architect-3 (F-23):** I do **not** use `try/except ImportError: pass` as a broad guard for the DINOv3 import in `src/models/classification/__init__.py`. I catch only the specific `transformers`-related ImportError, log at WARNING level, store the failure reason in a module attribute for diagnostics, and re-raise anything else. Broad `pass` masks real bugs (this was the inquisitor's F-23 point as well).

## Evolution

- **P0 added as scaffolding phase** (was: scaffolding embedded in P0 bug fix) — triggered by F-32, F-03.
- **P0 behavior snapshot is now predictions JSON, not `state_dict` SHA** (SHA demoted to advisory sentinel) — F-03.
- **P1 bug fix now requires pre-fix failing test as first commit inside PR** — F-20.
- **P3 added as dedicated deps + transform-shim phase** (was: `transformers` appended to `requirements.txt` in P2) — F-04, F-34, F-37.
- **Dependency policy switched from `requirements.txt` edit to `requirements-dinov3.txt` extras** — F-04, F-21, F-34.
- **`src/datasets/transforms.py` shim added in P3** as single transform-injection point with explicit ResNet-unchanged regression — F-08, F-16, F-30, F-37.
- **P4 mixup is now wired in `training_step` (off-by-default) instead of "ported but not called"** — F-02.
- **P4 entry criterion no longer requires HF credentials in CI** — F-01.
- **P4 lazy-import error message standardized** with explicit install instruction — F-46.
- **P4 ships `on_save_checkpoint` writing `dinov3_metadata`** (model_name, image_size, normalization) — F-06, F-41, F-45.
- **P5 ships minimal `docs/dinov3_experimental.md`** with auth/memory/image-size/security/license warnings — F-09, F-39, F-40, F-44.
- **P5 loader includes `_categorize_keys` with fail-loud unknown-key handling** — F-07.
- **P5 loader auto-loads `image_size`/`model_name` from embedded metadata** — F-06, F-41.
- **P5 import guard in `classification/__init__.py` narrowed** (only catches transformers ImportError, logs, stores diagnostic) — F-23.
- **P6 emits memory warning with conservative, attributed numbers** — F-28.
- **P7 adds runtime `_validate_peft_targets` with actionable errors** — F-05, F-46.
- **P7 embeds `peft_config` in checkpoint via extended `dinov3_metadata`** — F-06, F-41.
- **P7 pins `peft>=0.10.0,<0.13.0`** (upper bound) — F-04.
- **P8 hard pre-flight gate: real SINCHI ckpt inspection committed to `sinchi_key_inventory.md` before code** — F-15, F-29, F-38.
- **P8 adds `test_corrupt_ckpt_errors.py` for `weights_only=False` security failure modes** — F-44.
- **P9 adds `docs/error_taxonomy.md`** — F-46.
- **P9 adds CPU/MPS guidance + DDP out-of-scope statement** — F-42, F-43.
- **P9 adds prominent README migration note for P1 bug fix** — F-47.
- **Regression strategy revised:** behavior snapshots primary, byte equality only for 4 critical files, default-install subprocess test — F-03, F-24.
- **Regression hash comparison uses normalized dict, not `yaml.safe_dump`** — F-19.
- **Rollback claim corrected from "independent" to reverse-DAG chain** — F-22.
- **Rollback dry-run operationalized with concrete `git worktree` commands** (repo-local, no `/tmp`) — F-25.
- **Acceptance items 11 (transform proof) and 12 (real SINCHI key inspection) added** — F-08, F-15, F-29, F-30, F-37, F-38.
- **Acceptance items labelled CI / local / GPU / real-artifact** for clarity — F-36.
- **CI install matrix staged: introduced at P3, not P0** — F-26.
- **Coverage thresholds explicitly NOT a merge gate** — F-18, F-36.
- **`MockAutoModel` fixture moved to `tests/conftest.py` and shared** — F-01.
- **All temp paths repo-local (`tests/_artifacts/`); never `/tmp`** — F-14.
- **Subprocess used for true absent-package testing**, not `sys.modules[...] = None` trick — F-17.

## NEW This Round

- Dedicated **P0 scaffolding phase** with predictions-based behavior snapshot.
- Dedicated **P3 dependency + transform-shim phase**.
- **`requirements-dinov3.txt` extras policy** with documented install matrix (adopted from architect-3).
- **`src/datasets/transforms.py::build_dinov3_eval_transform(image_size)` injection point** — single contract resolving the transform-size question.
- **Mixup wired-but-off in P4** (config key `use_mixup: False`) instead of "latent port".
- **`on_save_checkpoint` / `on_load_checkpoint` hooks embedding `dinov3_metadata`** (model_name, image_size, normalization, peft_method, lora_r/alpha/dropout, target_modules).
- **`_validate_peft_targets`** runtime named-module inspection with actionable error listing valid candidate suffixes.
- **`_categorize_keys`** loader helper that classifies keys into backbone/classifier/adapter/unknown and fails loud on unknown.
- **Hard pre-flight gate for P8:** `scripts/inspect_sinchi_keys.py` output committed to `sinchi_key_inventory.md` before any P8 code is written.
- **`tests/regression/test_default_install_surface.py`** runs in a subprocess with only `requirements.txt` installed; proves default users see no `transformers` import.
- **`docs/dinov3_experimental.md`** minimal Gate-1 doc with HF auth, GPU memory, image size, weights_only security, license.
- **`docs/error_taxonomy.md`** standardized error message catalogue.
- **`tests/inference/test_corrupt_ckpt_errors.py`** negative tests for `weights_only=False` failure modes.
- **`tests/datasets/test_dinov3_transform_pipeline.py`** asserts DINOv3 path emits 518 and ResNet path emits byte-equal 224.
- **`tests/algorithms/test_peft_target_modules_validated.py`** asserts mis-typed target list raises actionable error.
- **Phase-tag annotated commits** (`phase-N-merged`) for cheap bisect and rollback identification.
- **Reverse-DAG rollback chain** documented honestly; per-phase rollback dry-run paste mandatory in PR description.
- **CPU/MPS auto-fallback to fp32 with WARNING** when `precision="16-mixed"` is requested on non-CUDA.
- **Conservative GPU memory numbers** (per-model RoT in docs; no unverified citations).
- **Subprocess-based absent-package testing** for `transformers` and `peft` (replaces `sys.modules` trick).
- **Predictions-JSON behavior snapshot with `atol=1e-3`** instead of brittle `state_dict` SHA.
- **Public-surface freeze** as a separate test enforcing add-only API evolution.
- **Critical-files byte-equality** restricted to 4 named files (`main.py`, `plain.py`, `plain_resnet.py`, `custom.py`).
- **Pre-fix failing-test commit** as the first commit inside the P1 bug-fix PR.
- **Acceptance checklist extended to 12 items** with explicit CI / local-GPU / real-artifact tagging.
- **License/citation note** explicit at P5 (minimal) and P9 (full).
- **DDP/multi-GPU explicitly scoped out** in P9 docs.
- **CHANGELOG migration note for P1 bug fix** at top of README in P9.

STATUS: NEW=25 DISAGREEMENTS=3
