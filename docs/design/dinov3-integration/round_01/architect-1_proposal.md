# DINOv3 integration — phased plan (architect-1: phasing/MVP perspective)

## Executive overview

The research report specifies a strictly-additive, ~750-line integration spread across 10 steps. The fastest path to user value is to **front-load the cheapest, most-leverage change (the `CustomWeights` bug fix)**, then deliver a **minimum-viable linear-probe pipeline** (train + load + infer) before layering the more complex modes (full fine-tune → LoRA/PiSSA → SINCHI-checkpoint compatibility). The work is decomposed into **8 phases**, each a single reviewable PR (≤ ~500 LOC of new code), with two explicit *pause-and-evaluate gates* where the team ships to users and waits for real-world feedback before continuing. Architecturally risky work — the new model + algorithm — lands in Phase 2 so that everything downstream (inference, PEFT, multi-format loading) only refines a known-good core.

Key ordering principles:
1. **Risk-front-loaded:** the most uncertain integration (HF `AutoModel` + Lightning marriage) ships early, in Phase 2, when only ~3 files exist downstream of it.
2. **Dependency-respecting:** each phase only needs symbols introduced in prior phases.
3. **MVP at Gate 1:** after Phase 3, a user can train a DINOv3 linear probe on demo data and load the resulting `.ckpt` through `DINOv3CustomWeights`. That alone is shippable.
4. **Single-PR sizing:** every phase has a defined revert unit (one squash-merge PR).

## Phase index

| # | Phase name | Deliverable | Tests | Est. LOC | Revert unit |
|---|---|---|---|---|---|
| 0 | CustomWeights bug fix + regression baseline | Patch `resnet_base/custom_weights.py`; add regression snapshot fixture | `tests/inference/test_custom_weights_num_layers.py`, `tests/regression/test_resnet_baseline.py` | ~80 | PR #0 (squash) |
| 1 | Mixup utility port | `src/utils/mixup.py` | `tests/utils/test_mixup.py` | ~200 | PR #1 |
| 2 | DINOv3 model + linear-probe algorithm (MVP core) | `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py`, registry wires, `requirements.txt` (+`transformers`), sample config | `tests/models/test_dinov3_classifier.py`, `tests/algorithms/test_dinov3_linear_smoke.py` | ~380 | PR #2 |
| 3 | Inference loader (MD-Cls `.ckpt` flavour only) | `src/models/classification/dinov3_base/{__init__,base_classifier,custom_weights}.py`, registry wire | `tests/inference/test_dinov3_custom_weights_mdckpt.py` | ~280 | PR #3 |
| — | **Gate 1: ship + collect feedback** | (Phases 0–3 cover Steps 1, 2, 3, 4, 6 partial, 7 partial, 8) | — | — | — |
| 4 | Full fine-tune toggle | Extend `dinov3_linear.py` with dual param-groups + `unfreeze` path | `tests/algorithms/test_dinov3_finetune_grads.py` | ~120 | PR #4 |
| 5 | PEFT (LoRA + PiSSA) | Extend `dinov3_linear.py` `_wrap_with_peft`; add `peft` to `requirements.txt` (optional) | `tests/algorithms/test_dinov3_peft_smoke.py`, `tests/algorithms/test_peft_checkpoint_roundtrip.py` | ~200 | PR #5 |
| 6 | Multi-format inference loader (SINCHI raw + SINCHI PEFT) | Extend `_extract_state_dict` autodetect; add PEFT re-wrap path | `tests/inference/test_dinov3_custom_weights_sinchi.py`, `tests/inference/test_dinov3_custom_weights_peft.py` | ~150 | PR #6 |
| — | **Gate 2: ship + collect feedback** | (Phases 4–6 close Steps 5 advanced toggles + Step 7 multi-format) | — | — | — |
| 7 | Config + docs consolidation | `configs/config.yaml` commented block, `configs/config_dinov3_linear.yaml`, `docs/usage.md` section, `README.md` touch-ups | `tests/docs/test_sample_configs_parse.py` | ~150 | PR #7 |

Cumulative net additions ≈ 1,560 lines (code + tests + docs); largest single PR ≈ 380 LOC.

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

**Ordering rationale (MVP lens):**

1. **Phase 0 first** because the `CustomWeights` `num_layers=50` hard-code is an *existing* bug that already blocks current users training with the default `num_layers: 18` in `configs/config.yaml`. Fixing it now is an immediate, zero-risk, user-visible win that does not depend on any DINOv3 work and lets us *establish the regression baseline fixture* the next 7 phases will rely on.
2. **Phase 1 (mixup port)** is a pure leaf utility — no callers yet, easy to unit-test, trivially revertible. Front-loading it removes a dependency from Phase 2's surface area.
3. **Phase 2** is the **architecturally riskiest** slice: it introduces `transformers`, HF `AutoModel`, the `.feature`/`.classifier` shape contract for a ViT backbone, and the Lightning↔raw-PyTorch translation. We deliberately deliver this as the *third* PR, before any inference or PEFT complexity, so it can be exercised in isolation. The mixup port already exists by now but is **not yet wired** into the algorithm in this phase — we keep mixup off-by-default so Phase 2 only proves "linear probe trains end-to-end on demo data."
4. **Phase 3** adds the minimal inference loader (MD-Cls `.ckpt` only) so users can do something with the checkpoints from Phase 2. This closes the **end-to-end MVP loop** (train → save → load → predict) using only the most homogenous checkpoint flavour we control.
5. **Gate 1** ships Phases 0–3. The team waits for ≥1 week of real-world feedback. Real-world signals we hope to catch: HF auth gating UX, GPU OOM on user GPUs, transform-resize behavior on non-square crops, checkpoint path conventions.
6. **Phases 4–5** layer the advanced training toggles (full fine-tune, then LoRA/PiSSA) — both are deltas on the single algorithm file from Phase 2, so they are small.
7. **Phase 6** generalizes the inference loader to SINCHI's two checkpoint shapes. We deliberately *do not do this before Gate 1*: real SINCHI checkpoint binary samples may not be available to us at planning time, and rather than block the MVP on that risk, we ship MD-Cls-only loading first.
8. **Gate 2** ships Phases 4–6 — every code capability now exists.
9. **Phase 7** is the docs/config consolidation pass — done **last** so docs describe what actually shipped, not what was planned.

**Dependency DAG (each arrow = "needs symbols from"):**

```
P0 (bug fix)  ── independent
P1 (mixup)    ── independent
P2 (model+algo) ── needs P1 (imports noisy_same_class_mixup; mixup OFF by default)
P3 (loader v1)  ── needs P2 (DINOv3Classifier)
                            │
                       — Gate 1 —
                            │
P4 (full FT)    ── needs P2 (extends DINOv3Linear)
P5 (PEFT)       ── needs P2, P4 (uses unfreeze path; param-group selection)
P6 (loader v2)  ── needs P3, P5 (PEFT re-wrap for adapter ckpts)
                            │
                       — Gate 2 —
                            │
P7 (docs)       ── needs P0..P6 (describes shipped surface)
```

**Generic entry/exit criteria pattern (applies to every phase):**

- **Entry:** previous phases' PRs merged on `main`; CI green on `main`; regression baseline snapshot (from Phase 0) reproduces bit-identically before phase work starts.
- **Exit:** phase PR rebased on `main`; new tests pass; regression baseline still bit-identical (or, for Phase 4+, identical on the default ResNet config path); reviewers confirm the LOC budget and single-revert-unit constraints are met; CHANGELOG.md entry added.

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

**Phase 1 — Port `noisy_same_class_mixup` + `efficient_batch_same_class_mixup`.**

- **Deliverable (files):**
  - **NEW** `src/utils/mixup.py` — verbatim port of both functions from `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py`, with a header docstring naming the source path and commit SHA.
  - (No `__init__.py` change — users import explicitly: `from src.utils.mixup import noisy_same_class_mixup`.)
- **Entry criteria:**
  - Phase 0 merged (regression baseline exists).
  - `src/utils/` confirmed to be a package (verify `__init__.py` exists).
- **Exit criteria:**
  - Both functions importable: `python -c "from src.utils.mixup import noisy_same_class_mixup, efficient_batch_same_class_mixup"`.
  - Unit tests pass.
  - No production callsite yet (the port is *latent*; Phase 2 wires it).
- **Tests** (`tests/utils/test_mixup.py`):
  - `test_noisy_same_class_mixup_shape_preserved`: assert output `(emb, y)` has same shape as input for synthetic `(B=8, D=16)` embeddings and integer labels.
  - `test_noisy_same_class_mixup_labels_unchanged`: assert `y_out == y_in` element-wise (per SINCHI semantic: labels never change).
  - `test_noisy_same_class_mixup_lambda_in_range`: monkey-patch `torch.distributions.Beta` to capture sampled λ; assert all draws clamped to `[0.1, 0.9]`.
  - `test_noisy_same_class_mixup_singleton_class_passthrough`: a sample whose label has no partner returns unchanged (no NaN, no crash).
  - `test_noise_type_modes`: parametrize over `{"gaussian","dropout","both"}`; assert output differs from no-noise baseline and from each other under a fixed `torch.manual_seed(0)`.
  - `test_efficient_batch_same_class_mixup_matches_naive`: assert vectorized variant matches the naive loop to within `atol=1e-5` on a 32-sample fixture.
  - Invocation: `pytest tests/utils/test_mixup.py -v`.
- **Revert:** Single PR; revert = `git revert <merge SHA>`. Has no downstream importers yet, so revert is purely local.

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

**Phase 2 — The MVP core.** Adds `DINOv3Classifier` + `DINOv3Linear` (linear-probe mode only; full fine-tune and PEFT switches *exist as config keys* but are inert/guarded — see exit criteria).

- **Deliverable (files):**
  - **NEW** `src/models/dinov3.py` (~90 LOC) — verbatim per research §Step 2: `DINOv3Classifier` with `.feature` (HF `AutoModel`) and `.classifier` (`nn.Linear`) split, `freeze_backbone()`/`unfreeze_backbone()`, `encode()`, `forward()`.
  - **EDIT** `src/models/__init__.py` — append `from .dinov3 import *`.
  - **NEW** `src/algorithms/dinov3_linear.py` (~180 LOC) — `DINOv3Linear` LightningModule per research §Step 4, but **PEFT branch raises `NotImplementedError` if enabled** ("LoRA support arrives in Phase 5") and **`freeze_backbone=False` raises `NotImplementedError`** ("full fine-tune arrives in Phase 4"). This keeps Phase 2's surface narrow and its PR reviewable.
  - **EDIT** `src/algorithms/__init__.py` — append `from .dinov3_linear import *`.
  - **EDIT** `requirements.txt` — append `transformers>=4.40.0` (no `peft` yet; deferred to Phase 5).
  - **NEW** `configs/config_dinov3_linear.yaml` — minimal working config (linear probe, mixup off, peft off) for the demo dataset.
- **Entry criteria:**
  - Phase 1 merged; `src/utils/mixup` importable.
  - HuggingFace credentials available in CI secret or skip-marker in place.
- **Exit criteria:**
  - `python -c "from src.models import DINOv3Classifier; from src.algorithms import DINOv3Linear"` succeeds.
  - Smoke training run: `python main.py --config configs/config_dinov3_linear.yaml` runs for 1 epoch on the demo `Custom_Crop` dataset, produces a `.ckpt` under `weights/Crop/DINOv3Linear/`.
  - Regression baseline from Phase 0 still bit-identical on the default ResNet config.
  - Lazy-import guard: in an env without `transformers`, `from src.models import classification` (the inference-zoo umbrella) still imports without error.
- **Tests:**
  - `tests/models/test_dinov3_classifier.py`:
    - `test_construct_with_mocked_automodel`: monkey-patch `transformers.AutoModel.from_pretrained` to return a stub with `.config.hidden_size=64` and a deterministic forward; assert `DINOv3Classifier(num_cls=3)` builds, `.classifier` is `nn.Linear(64,3)`, frozen-by-default state of `feature.parameters()` is `requires_grad=False`.
    - `test_encode_extracts_cls_token`: feed a stubbed output with `last_hidden_state` shape `(2, 197, 64)`; assert `encode()` returns shape `(2, 64)`.
    - `test_attribute_naming_matches_md_cls_contract`: `hasattr(model, "feature") and hasattr(model, "classifier")`.
  - `tests/algorithms/test_dinov3_linear_smoke.py`:
    - `test_imports_lazily`: with `transformers` unavailable (use `importlib.util.find_spec` mock), `from src.algorithms import DINOv3Linear` still succeeds; only construction raises `ImportError` with the install-instruction message.
    - `test_finetune_path_blocked`: building with `freeze_backbone=False` raises `NotImplementedError`.
    - `test_peft_path_blocked`: building with `dinov3.peft.enable=True` raises `NotImplementedError`.
    - `test_one_step_training` (GPU-marked, skipped if no GPU + no HF token): build with mocked tiny backbone, run a single `training_step` on a synthetic 4-sample batch, assert loss is finite and gradient flows into `classifier` only.
  - Invocation: `pytest tests/models tests/algorithms -v -m "not gpu"` for CI; full suite with `-m gpu` locally.
- **Revert:** Single PR. Revert cleanly removes 3 new files + 3 single-line `__init__` edits + 1-line requirements edit + 1 sample config. No other code depends on it yet (Phase 3 not merged).

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

**Phase 4 — Lift the `NotImplementedError` for `freeze_backbone=False`.**

- **Deliverable (files):**
  - **EDIT** `src/algorithms/dinov3_linear.py` — remove the Phase-2 guard; introduce dual param-groups in `configure_optimizers` (one for `self.net.feature.parameters()`, one for `self.net.classifier.parameters()`) so users can give the backbone a smaller LR (`lr_feature`) than the head (`lr_classifier`), matching the convention from `Plain.configure_optimizers`. Also call `self.net.unfreeze_backbone()` when `freeze_backbone=False`.
- **Entry criteria:**
  - Phase 3 merged; Gate 1 feedback period (≥1 week) completed; no blocking issue reports from Gate 1.
- **Exit criteria:**
  - `freeze_backbone: False` config produces non-zero gradient norms on `self.net.feature.*` params during the first `training_step`.
  - Linear-probe regression: with `freeze_backbone: True`, training loss curve over 1 epoch matches Phase 2's curve under fixed seed (within `atol=1e-4`).
- **Tests** (`tests/algorithms/test_dinov3_finetune_grads.py`):
  - `test_unfreeze_makes_backbone_trainable`: after construction with `freeze_backbone=False`, `sum(p.requires_grad for p in net.feature.parameters()) > 0`.
  - `test_param_groups_split`: assert `configure_optimizers()` returns an `AdamW` (or `SGD` per chosen pattern) with exactly two param groups whose LRs match `lr_feature` and `lr_classifier`.
  - `test_backbone_grad_flows`: one `training_step` on synthetic batch with mocked tiny backbone; assert at least one `net.feature.*` parameter has non-`None`, non-zero `.grad`.
  - `test_frozen_path_still_works`: parametrized check that `freeze_backbone=True` path remains identical to Phase 2 behavior.
- **Revert:** Single PR; revert restores the `NotImplementedError` guard. No new files. Safe — Phase 5 has not yet been merged.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

**Phase 5 — Lift the PEFT `NotImplementedError`; add LoRA + PiSSA wrap.**

- **Deliverable (files):**
  - **EDIT** `src/algorithms/dinov3_linear.py` — implement `_wrap_with_peft()` per research §Step 4 (lazy `from peft import LoraConfig, get_peft_model, TaskType`; wrap `self.net.feature` only; `TaskType.FEATURE_EXTRACTION`; default `target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]`; PiSSA via `init_lora_weights="pissa_niter_4"` when `method=="pissa"`).
  - **EDIT** `requirements.txt` — append `peft>=0.10.0  # optional, only for LoRA / PiSSA`.
- **Entry criteria:**
  - Phase 4 merged.
- **Exit criteria:**
  - With `dinov3.peft.enable: True, method: lora`, the trainable-parameter count is `< 5%` of total (assert in test).
  - With `method: pissa`, the wrap succeeds with no warnings about unrecognized `init_lora_weights`.
  - Saved Lightning `.ckpt` from a 1-epoch PEFT run can be re-loaded by the same `DINOv3Linear.load_from_checkpoint(...)` (LoRA adapter weights survive the round-trip).
  - Lazy-import: env without `peft` installed → constructing `DINOv3Linear` with `peft.enable=False` still succeeds; with `enable=True` it raises `ImportError` with install instruction.
- **Tests:**
  - `tests/algorithms/test_dinov3_peft_smoke.py`:
    - `test_lora_wrap_reduces_trainable_count`: with mocked tiny backbone, assert wrapped trainable count is `< 0.1 ×` un-wrapped count.
    - `test_pissa_method_accepted`: build with `method="pissa"`, assert `init_lora_weights="pissa_niter_4"` is set on the constructed `LoraConfig` (capture via spy).
    - `test_peft_lazy_import`: simulate `peft` missing → `enable=False` works; `enable=True` raises `ImportError("peft is required...")`.
    - `test_classifier_outside_wrap`: assert `self.net.classifier` is still a plain `nn.Linear` (not PEFT-wrapped) after `_wrap_with_peft`.
  - `tests/algorithms/test_peft_checkpoint_roundtrip.py`:
    - Train 2 steps with PEFT enabled → call `trainer.save_checkpoint("/tmp_local/test.ckpt")` → `DINOv3Linear.load_from_checkpoint("/tmp_local/test.ckpt")` → assert classifier weights and LoRA adapter weights are bit-equal to the saved tensors (use repo-local tmp dir, **never `/tmp`** per environment rules).
- **Revert:** Single PR; revert removes `_wrap_with_peft` body and re-instates the `NotImplementedError`. Phase 4 capabilities unaffected. Note: removing `peft>=0.10.0` from `requirements.txt` is part of the revert and is safe since it's optional.

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

This scope item is **split across Phase 3 and Phase 6** to keep PRs small and to let the MVP ship without depending on SINCHI checkpoint binaries that may not be on hand.

### Phase 3 — Loader v1: MD-Cls `.ckpt` only

- **Deliverable (files):**
  - **NEW** `src/models/classification/dinov3_base/__init__.py` (2 lines, re-export both classes).
  - **NEW** `src/models/classification/dinov3_base/base_classifier.py` (~110 LOC) — `DINOv3BaseInference` per research §Step 7, but `_extract_state_dict` handles **only** the `"state_dict"` + `net.` prefix flavour. SINCHI flavours raise `NotImplementedError("SINCHI checkpoint flavours supported in Phase 6")`.
  - **NEW** `src/models/classification/dinov3_base/custom_weights.py` (~70 LOC) — `DINOv3CustomWeights` per research §Step 7.
  - **EDIT** `src/models/classification/__init__.py` — append `from .dinov3_base import *`.
- **Entry criteria:**
  - Phase 2 merged; at least one MD-Cls `DINOv3Linear` `.ckpt` exists under `weights/Crop/DINOv3Linear/` (produced by Phase 2 smoke run).
- **Exit criteria:**
  - `DINOv3CustomWeights(weights=<md_cls_ckpt>, class_names=[...])` loads with zero missing/unexpected keys (assert).
  - `forward()` on a `(1, 3, 518, 518)` tensor returns logits of shape `(1, num_cls)`.
  - `results_generation` output dict has exactly the existing 5 keys (`img_id, prediction, class_id, confidence, all_confidences`) — schema parity with `CustomWeights`.
- **Tests** (`tests/inference/test_dinov3_custom_weights_mdckpt.py`):
  - `test_load_md_cls_ckpt`: build, load Phase-2 fixture, assert `missing == [] and unexpected == []`.
  - `test_results_generation_schema`: assert exact key set matches existing `CustomWeights` reference.
  - `test_sinchi_flavour_raises`: pass a fixture with `model_state_dict` key → raises `NotImplementedError`.
  - `test_pyw_style_import_path`: `from src.models import classification as pw_classification; pw_classification.DINOv3CustomWeights` resolves.

### Phase 6 — Loader v2: SINCHI raw + SINCHI PEFT

- **Deliverable (files):**
  - **EDIT** `src/models/classification/dinov3_base/base_classifier.py` — extend `_extract_state_dict` to handle `"model_state_dict"` + `backbone.` → `feature.` remap, plus PEFT-wrapped delta keys; accept `peft_config=` kwarg that re-wraps the backbone before `load_state_dict(strict=False)`.
- **Entry criteria:**
  - Phase 5 merged.
  - **Pre-flight (MANDATORY):** at least one real SINCHI raw `.pt` and one real SINCHI PEFT `.pt` checkpoint have been obtained and their inner key layouts verified by inspection (`torch.load(...).keys()`). This is the *single biggest risk* in the plan because the report's `backbone.*` claim is *inferred*, not verified (see research report §Confidence). If keys differ, the remap rule is adjusted before code is written.
- **Exit criteria:**
  - Both SINCHI flavours load with `strict=False`; for raw, `unexpected == []` and `missing` is empty (modulo `embedding_dim` mismatch warnings if the user passes the wrong `model_name`).
  - For PEFT, `missing` is empty after the re-wrap; adapter delta keys present in `state` are consumed.
- **Tests:**
  - `tests/inference/test_dinov3_custom_weights_sinchi.py`:
    - `test_load_sinchi_raw`: pass a SINCHI fixture; assert key remap applied (spy on `_extract_state_dict` output keys) and `load_state_dict` strict succeeds.
    - `test_load_sinchi_raw_with_synthetic_keys`: build a synthetic `{"model_state_dict": {"backbone.encoder.layer.0.weight": tensor, "classifier.weight": tensor, "classifier.bias": tensor}}` checkpoint; assert remap to `feature.encoder.layer.0.weight`.
  - `tests/inference/test_dinov3_custom_weights_peft.py`:
    - `test_load_sinchi_peft_with_config`: pass `peft_config={"r":8,"lora_alpha":8,"target_modules":[...]}` along with a synthetic PEFT-flavoured fixture; assert backbone is re-wrapped (`isinstance(net.feature, PeftModel)`) and adapter weights are loaded.
    - `test_peft_without_config_warns`: loading PEFT ckpt without `peft_config` produces a clear error or fall-back with non-empty `missing` list (whichever behavior we choose — documented in test).

- **Revert (both sub-phases):** Each is its own PR. Phase 3 revert removes the entire `dinov3_base/` subpackage; Phase 6 revert restores only the simpler `_extract_state_dict` body. Neither revert affects training (Phase 2/4/5).

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

**Phase 7 — Consolidate configs and docs *after* all capabilities ship.**

- **Deliverable (files):**
  - **EDIT** `configs/config.yaml` — append the ~25-line commented `dinov3:` / `training:` / `label_smoothing` block (research §Step 5). All commented → existing default run is bit-identical (regression-protected).
  - **EDIT** `configs/config_dinov3_linear.yaml` — the Phase-2 minimal config is now enriched with documented full-FT, PEFT, and mixup options (commented variants). Possibly add sibling files `configs/config_dinov3_finetune.yaml` and `configs/config_dinov3_lora.yaml` for clarity.
  - **EDIT** `docs/usage.md` — append "Fine-tuning with DINOv3" section per research §Step 9: capability matrix, config key tables, HF auth note, GPU memory note, inference snippet.
  - **EDIT** `README.md` — capability bullet + model-zoo example block per research §Step 9.
  - **NEW** `CHANGELOG.md` entry summarising Phases 0–7 (or append to existing).
- **Entry criteria:**
  - Phase 6 merged; Gate 2 ≥1-week feedback collected.
- **Exit criteria:**
  - `python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"` succeeds → default block parses.
  - Every sample `configs/config_dinov3_*.yaml` parses and produces a `Munch` object whose required keys are present.
  - `README.md` and `docs/usage.md` accurately describe shipped behaviour (verify by running each documented snippet end-to-end on demo data).
- **Tests** (`tests/docs/test_sample_configs_parse.py`):
  - Parametrized over every `configs/*.yaml`: `test_yaml_parses`, `test_required_keys_present` (assert `algorithm`, `model_name`, `dataset_name`, `num_classes`, `num_epochs`).
  - `test_default_config_unchanged_in_behavior_keys`: load `configs/config.yaml`, assert the uncommented section matches a frozen snapshot (regression).
- **Revert:** Single PR; revert removes the appended docs sections and sample configs. Default `config.yaml` returns to pre-Phase-7 state, but every Phase 0–6 capability remains available via direct code use.

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

**Phase 0 — Ships first. Stand-alone, blocks nothing, fixes a real user-facing latent bug.**

- **Deliverable (files):**
  - **EDIT** `src/models/classification/resnet_base/custom_weights.py` — widen constructor signature from `def __init__(self, weights=None, class_names=None, device="cpu")` to `def __init__(self, weights=None, class_names=None, device="cpu", num_layers=50)`, and pass `num_layers=num_layers` (instead of hard-coded `50`) into `super().__init__(...)`. Default preserves existing behavior; positional callers unaffected.
  - **NEW** `tests/fixtures/regression/` — directory holding two snapshot fixtures (see "Regression protection" §). For Phase 0, create one fixture: SHA-256 of the `state_dict` of a freshly-trained ResNet-18 demo run, plus the YAML used.
- **Entry criteria:**
  - Working tree clean; CI green on `main`.
  - One ResNet-18 demo checkpoint and one ResNet-50 demo checkpoint exist (or are produced as part of the PR's test fixtures).
- **Exit criteria:**
  - `CustomWeights(weights=<resnet18.ckpt>, class_names=[...], num_layers=18)` loads with zero missing/unexpected keys.
  - `CustomWeights(weights=<resnet50.ckpt>, class_names=[...])` (no `num_layers` kwarg) still loads with zero missing/unexpected keys — backward compatibility verified.
  - All existing tests pass.
- **Tests** (`tests/inference/test_custom_weights_num_layers.py`):
  - `test_resnet18_loads_with_explicit_num_layers`: produce/load a minimal ResNet-18 checkpoint; assert `missing == [] and unexpected == []`.
  - `test_resnet50_default_still_loads`: produce/load a minimal ResNet-50 checkpoint without passing `num_layers`; assert successful load.
  - `test_default_signature_unchanged_at_call_site`: call `CustomWeights(weights=p, class_names=cn, device="cpu")` (positional + kwargs as currently used in `docs/usage.md`) — assert no `TypeError`.
- **Revert:** Single PR; revert restores the hard-coded `num_layers=50`. Safe because no later phase depends on this fix structurally (DINOv3 has its own loader).

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

**Strategy:** every phase must prove that the *default ResNet pipeline* (`configs/config.yaml` unchanged) and the *existing inference path* (`PlainResNetInference`, `CustomWeights`) remain bit-identical. We use a three-layer defense:

1. **Snapshot fixtures (introduced in Phase 0).** Stored under `tests/fixtures/regression/`:
   - `resnet18_baseline.ckpt.sha256` — SHA-256 of the `state_dict` produced by `python main.py` with `configs/config.yaml` + `seed_everything(42)` + 1 epoch on the demo dataset.
   - `resnet18_baseline.config.yaml` — frozen copy of the YAML.
   - `resnet18_baseline.predictions.json` — first 10 demo-image predictions (`img_id → class_id`).
2. **Invariants enforced in CI (`tests/regression/test_resnet_baseline.py`)** — run *every PR*:
   - `test_baseline_state_dict_hash_unchanged`: train 1 epoch with the frozen YAML + seed; SHA-256 of the resulting `state_dict` (sorted-key, `torch.save`-encoded) equals the stored hash.
   - `test_baseline_predictions_unchanged`: run inference on the 10 demo images with the resulting checkpoint; assert per-image `class_id` matches the stored JSON.
   - `test_default_config_yaml_unchanged`: parse `configs/config.yaml`; assert the *un-commented* section equals a frozen `dict` snapshot. (New commented blocks in Phase 7 must not affect this assertion — comments are stripped by YAML loader anyway.)
   - `test_plain_imports_unchanged`: `from src.algorithms import Plain; from src.models import PlainResNetClassifier; from src.models import classification as pw; pw.CustomWeights` — every symbol that pre-existed must still resolve.
3. **Public-surface freeze.** In `tests/regression/test_public_surface.py`, snapshot the `dir()` of `src.algorithms`, `src.models`, `src.datasets`, `src.models.classification` at Phase-0 completion. Each subsequent phase may **add** to these surfaces but must not **remove or rename** anything.

**Per-phase regression gate:** Phase N's PR may not merge if `pytest tests/regression/` reports any failure.

**Hash regeneration policy:** the snapshot may be regenerated **only** with explicit reviewer sign-off (a labelled `regression-baseline-bump` PR), and only in response to an intentional, documented behaviour change. None of Phases 0–7 should require such a bump.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Test pyramid

| Layer | What | Where | Runs |
|---|---|---|---|
| Unit | Pure-Python functions (mixup math, key-remap logic, state-dict hashing) | `tests/utils/`, `tests/inference/test_*_extract_state_dict.py` | CI on every PR; <30 s total |
| Integration | One-step `training_step` with mocked tiny HF backbone; one-shot `load_state_dict` with synthetic ckpts | `tests/algorithms/`, `tests/inference/` | CI on every PR; <2 min total |
| Regression | Bit-identical baseline (state_dict SHA, default-config snapshot, public surface) | `tests/regression/` | CI on every PR; <3 min |
| Smoke (GPU) | One-epoch real training run on demo dataset for each shipped path | `tests/smoke/`, marked `@pytest.mark.gpu` and `@pytest.mark.hf` | Local + nightly CI on a GPU runner |
| Acceptance | Full 10-item validation checklist (see next §) | `scripts/run_acceptance.sh` | Manual before each Gate ship |

### Fixtures

- `tests/fixtures/mini_dataset/` — 6 images × 2 classes, with pre-generated `*_annotations_cropped.csv` files. Used by every integration test that needs a `Custom_Crop` instance.
- `tests/fixtures/ckpts/`:
  - `resnet18_mini.ckpt`, `resnet50_mini.ckpt` (Phase 0)
  - `dinov3_linear_mini.ckpt` (Phase 2 produces; Phase 3 consumes)
  - `synthetic_sinchi_raw.pt`, `synthetic_sinchi_peft.pt` (Phase 6; constructed in-test via `torch.save({"model_state_dict": {...}}, ...)`)
- `tests/fixtures/mock_hf/` — a `MockAutoModel` class returning deterministic embeddings; used everywhere the real HF backbone is too heavy. Implementation lives in `tests/conftest.py` as a fixture.
- All temporary file writes go to `tests/tmp/` (gitignored), **never** `/tmp` (environment rules forbid `/tmp` / `mktemp`).

### Per-phase test invocation map

| Phase | Test command in CI |
|---|---|
| 0 | `pytest tests/inference/test_custom_weights_num_layers.py tests/regression/ -v` |
| 1 | `pytest tests/utils/test_mixup.py tests/regression/ -v` |
| 2 | `pytest tests/models tests/algorithms tests/regression/ -v -m "not gpu and not hf"` |
| 3 | `pytest tests/inference/test_dinov3_custom_weights_mdckpt.py tests/regression/ -v` |
| 4 | `pytest tests/algorithms/test_dinov3_finetune_grads.py tests/regression/ -v` |
| 5 | `pytest tests/algorithms/test_dinov3_peft_smoke.py tests/algorithms/test_peft_checkpoint_roundtrip.py tests/regression/ -v` |
| 6 | `pytest tests/inference/test_dinov3_custom_weights_sinchi.py tests/inference/test_dinov3_custom_weights_peft.py tests/regression/ -v` |
| 7 | `pytest tests/docs/test_sample_configs_parse.py tests/regression/ -v` |
| **Pre-Gate ship** | `bash scripts/run_acceptance.sh` (manual, GPU machine) |

### CI vs local

- **CI (no GPU, no HF token):** layers 1–3 plus dependency-skipped variants of layer 4 (smoke tests auto-skip when `torch.cuda.is_available()` is False or `HF_TOKEN` is unset). Target wall-time: < 6 min.
- **Local (GPU, HF token):** full pyramid including smoke + acceptance.

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

Map of research-report §8 checklist items to executable commands. `scripts/run_acceptance.sh` runs each in order and fails on the first non-zero exit.

| # | Research check | Concrete command | Pass criterion | Phase first satisfied |
|---|---|---|---|---|
| 1 | Existing ResNet path bit-identical | `pytest tests/regression/test_resnet_baseline.py -v` | All assertions green | P0 |
| 2 | New linear-probe path trains 1 epoch | `python main.py --config configs/config_dinov3_linear.yaml` (epochs=1) | `weights/Crop/DINOv3Linear/*.ckpt` exists, training loss finite | P2 |
| 3 | Full fine-tune produces backbone grads | `pytest tests/algorithms/test_dinov3_finetune_grads.py::test_backbone_grad_flows -v` | Backbone param `.grad` is non-zero | P4 |
| 4 | Mixup toggle changes loss curve | `python scripts/compare_loss_curves.py --with-mixup --without-mixup` (1 epoch each, fixed seed) | Curves differ; no shape errors | P5 (mixup gets wired-into-default once it's safe to enable — Phase 2 keeps it off, Phase 5 documents enabling) |
| 5 | LoRA + PiSSA train | `python main.py --config configs/config_dinov3_lora.yaml` then `configs/config_dinov3_pissa.yaml` | Both train; trainable-param count is `< 5%` of total (verify via `scripts/count_trainable.py`) | P5 |
| 6 | Inference loading — MD-Cls ckpt | `python scripts/infer_demo.py --ckpt weights/Crop/DINOv3Linear/<...>.ckpt` | No missing/unexpected keys | P3 |
| 7 | Inference loading — SINCHI raw | `python scripts/infer_demo.py --ckpt <sinchi_raw.pt>` | Loads after `backbone.→feature.` remap | P6 |
| 8 | Inference loading — SINCHI PEFT | `python scripts/infer_demo.py --ckpt <sinchi_peft.pt> --peft-config '{"r":8,...}'` | Loads with `strict=False`; no missing on adapter keys | P6 |
| 9 | CustomWeights regression | `pytest tests/inference/test_custom_weights_num_layers.py -v` | Both ResNet-18 and ResNet-50 load | P0 |
| 10 | Lazy-import in env without transformers | `pip uninstall -y transformers peft && python -c "from src.models import classification"` then re-install | Import succeeds; construction raises with install instruction | P2 (transformers) + P5 (peft) |

`scripts/run_acceptance.sh` is delivered in Phase 7 (it depends on every prior capability). A reduced variant `scripts/run_acceptance_gate1.sh` shipping in Phase 3 covers items {1, 2, 6, 9, 10-partial}, sufficient for the Gate-1 ship.

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

### Risk register

| ID | Risk | Phase first exposed | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| R1 | HuggingFace gating: DINOv3 models require `huggingface-cli login`; tests/CI without token cannot fetch | P2 | High | Med | Tests use `MockAutoModel` (no real fetch); CI marks real-HF tests as `@pytest.mark.hf` and auto-skips; docs in P7 instruct users to `huggingface-cli login` |
| R2 | GPU OOM at 518×518 ViT-B in FP32 on user hardware | P2 | Med | High | Document `precision="16-mixed"` + small `batch_size` in `docs/usage.md` (P7) and the sample config; smoke test deliberately uses tiny mocked backbone |
| R3 | `transformers` version incompatibility with newer PyTorch | P2 | Low | Med | Pin `transformers>=4.40.0` (matches SINCHI); CI runs against the pinned version; revert P2 PR if breaks emerge |
| R4 | `peft` optional dep absent breaks unrelated imports | P5 | Low | High | Lazy-import inside `_wrap_with_peft` body; covered by `test_peft_lazy_import`; if regression-protection test 4 fails on a fresh env, P5 PR is reverted |
| R5 | SINCHI checkpoint inner key layout is *inferred* (`backbone.*`), not verified — remap rule may be wrong | P6 | **Med-High** | Med | **Hard pre-flight gate on P6:** must obtain real SINCHI checkpoints and verify keys via `torch.load(...).keys()` before code is written. If keys differ from inference, update remap rule and update the unit test fixtures to match reality |
| R6 | Lightning ckpt-format drift (PyTorch Lightning version pin) | P2, P5 | Low | Med | Existing `requirements.txt` pins `lightning`; any version bump must re-run regression suite |
| R7 | `weights_only=False` in `torch.load` is a security risk if loading untrusted ckpts | P3, P6 | Low | Med | Document that users should only load checkpoints from trusted sources; matches existing `resnet_base/base_classifier.py` behavior |
| R8 | Bug fix in P0 silently changes behavior for an existing caller that *relied* on the hard-coded `50` | P0 | Very Low | Low | Default kwarg value remains `50`; `test_default_signature_unchanged_at_call_site` proves positional/kwarg call sites unaffected |
| R9 | Hidden coupling: an existing test inadvertently depends on `from src.algorithms import *` only yielding `Plain` | P2 | Low | Low | `test_public_surface` allows additions; if a real test fails on the surface diff, treat as a P2 blocker |
| R10 | Embedding-dim mismatch when user changes `dinov3.model_name` but loads a ckpt trained with a different backbone | P3, P6 | Med | Low | `DINOv3Classifier.__init__` infers `hidden_size`; loader raises a clear shape-mismatch error rather than silent corruption — covered by an explicit negative test in P3 |
| R11 | Real-world feedback at Gate 1 invalidates Phase 4–6 design assumptions | Gate 1 | Med | Med | Built-in: that's the *point* of the gate. The phasing is intentionally late-binding past Gate 1 |

### Per-phase rollback procedure

Every phase is one squash-merged PR. Rollback recipe is uniform:

```bash
# Identify the merge commit
git log --oneline --merges --grep="Phase N" -1
# Revert
git revert -m 1 <merge-sha>
git push origin main
# Verify regression
pytest tests/regression/ -v
```

Phase-specific notes:
- **P0**: revert restores hard-coded `num_layers=50`. Snapshot fixtures created in P0 are *not* removed by the revert (they live in `tests/fixtures/` and are still valid baselines). If reverted, subsequent phases must add a `tests/fixtures/regression/` directory of their own — flag this in the revert PR description.
- **P2**: largest blast radius. Reverting cleanly removes all DINOv3 training capability but leaves P0 + P1. P3+ must also be reverted (in reverse order: P7→P6→P5→P4→P3→P2) if P2 itself is reverted.
- **P5**: revert removes only the PEFT wrap implementation; `peft` line in `requirements.txt` is removed; users who manually installed `peft` are unaffected.
- **P6**: revert returns inference loader to MD-Cls-only flavour; users who loaded SINCHI checkpoints in production must pin their MD-Cls version prior to revert.
- **P7**: revert removes docs/sample configs but no capability is lost; the *uncommented* portion of `configs/config.yaml` is unchanged by P7 in the first place, so reverting is safe.

### Gate-level rollback

If Gate-1 feedback reveals a fundamental design flaw, **revert in order: P3 → P2 → P1 → P0**. The repo returns to pre-DINOv3 state in four commands. Because no symbols exported from existing modules are renamed or removed at any phase, no third-party code that depends on `MegaDetector-Classifier` breaks during rollback.

---

## Rationale (phasing/MVP lens)

- **Why P0 first, not P2?** The bug fix has zero dependencies, ships immediate value (ResNet-18 users can finally load checkpoints), and *establishes the regression infrastructure* the rest of the plan stakes itself on. Doing it later is wasteful because Phases 1–7 each have to assert regression — they need P0's machinery anyway.
- **Why split inference loader across P3 and P6?** The MD-Cls flavour is the only one where we control both endpoints (write in P2, read in P3). It is testable with zero external artifacts. The SINCHI flavours depend on real binary fixtures whose key layouts we have not yet verified (R5) — making P3 wait for them blocks the MVP unnecessarily. Splitting also keeps each PR under ~280 LOC.
- **Why mixup before model, when mixup has no caller until P2?** Three reasons: (a) it's a leaf utility easy to land first and forget; (b) putting it in P2 would push P2 over the 500-LOC ceiling; (c) reverting P1 alone is free since nothing imports it until P2.
- **Why two pause-and-evaluate gates rather than one?** Gate 1 (post-P3) ships the *capability* (train + load DINOv3 linear probes). Gate 2 (post-P6) ships the *coverage* (all training modes + all checkpoint flavours). Real-world signals from each are different: Gate 1 surfaces UX/perf issues; Gate 2 surfaces compatibility issues. Conflating them would mean shipping LoRA before knowing whether linear-probe even works for users.
- **Why are docs last (P7)?** Docs that promise features that aren't yet shipped are worse than no docs — they invite bug reports. P7 ships docs that describe *exactly* what already works.
- **Why no `main.py` edits at any phase?** The registry pattern (`algorithms.__dict__[name]`) was designed precisely for this. Touching `main.py` would expand blast radius for every phase and would violate the "pure additive" invariant of the research report.

### Pre-empted objections

- **To architect-2 (testing):** the test pyramid above is deliberately conservative on what runs in CI (no real HF fetches, no GPU dependency) so the dev loop stays fast. The smoke and acceptance layers exist for the manual gates. If architect-2 wants golden-file output snapshots for `results_generation`, I'd accept that as a P3 addition (~20 extra LOC, low cost).
- **To architect-3 (risk):** R5 (SINCHI key layout) is the only risk I flag as *blocking* — and it blocks only P6, not the MVP. Everything else is mitigated by lazy imports, gated tests, and the per-phase regression suite. If architect-3 wants a formal "release-gate checklist" for each Gate, that fits naturally as a `docs/release_gates.md` artifact owned by P7.

## New ledger items proposed

None. All 12 existing `scope_items` are addressed above. If the lead/inquisitor discovers a missing in-scope item during review (e.g., "embedding-cache phase" or "W&B integration"), I would propose deferring it to a post-Phase-7 follow-up rather than expanding the current 8-phase plan — both are explicitly out-of-scope in research report §7 and adding them would violate the MVP-first ordering principle.

## Discussion summary

*Round 1: to be populated after debate.*

## Consensus points

*Round 1: to be populated after debate.*

## Disagreements

*Round 1: to be populated after debate.*

## Evolution

*Round 1: n/a.*

## NEW This Round

*Round 1: n/a.*

STATUS: NEW=0 DISAGREEMENTS=0
