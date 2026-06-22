# DINOv3 integration — phased plan (architect-2: testing/verification, round 2)

> **Perspective unchanged.** A phase does not exist until it is verifiable. Test
> plans precede code. Round 2 rewires the test plan against the inquisitor's 11
> direct critiques (F-10..F-20) and 6 cross-architect findings that hit test
> strategy / fixtures / acceptance (F-32, F-33, F-35, F-36, F-37, F-38, F-44,
> F-45, F-46).

## Executive overview (changes from round 1)

Three structural changes to round 1:

1. **Phase 0 was too big** (F-10). It is split into **Phase 0a** (pytest harness
   + 6 baseline smoke tests, no coverage gates, no new CI job) and **Phase 0b**
   (coverage thresholds + the optional-deps absent-env matrix). Phase 0a is the
   only thing that must land before any other phase. Phase 0b can slip.
2. **Inference loader was bundled too late** (F-11, F-33). Adopting
   architect-1's split: **loader-v1** (MD-Cls `.ckpt` only) lands immediately
   after the linear-probe algorithm; **loader-v2** (SINCHI raw + SINCHI PEFT)
   lands after the PEFT phase and is gated on a **real-checkpoint key
   inspection** artifact (F-15, F-38).
3. **Full-FT and PEFT are real implementation phases**, not test-only verifications
   (F-12, F-13, F-35). The Phase-3/4 skeleton ships with `freeze_backbone=False`
   and `peft.enable=True` guarded by `raise NotImplementedError(...)`; later
   phases lift those guards. Each guard removal is its own commit with its own
   failing-test-first proof.

Renumbering: **0a, 0b, 1, 2, 3, 4, 5 (loader-v1), 6 (full-FT), 7 (PEFT), 8 (loader-v2), 9 (config+docs)**.
Total 10 phases. Phase 0a is the only hard prerequisite for "first user-visible value."

## Phase index (table)

| # | Phase | Touches | New tests | Gate to next phase |
|---|---|---|---|---|
| 0a | Pytest harness + baseline smoke | `pyproject.toml` (pytest ini block only — no extras), `tests/conftest.py`, `tests/fixtures/tiny_dataset/`, `tests/baseline/` | 6 baseline | All 6 pass on stock `main`; no CI job change yet |
| 0b | Coverage gates + optional-deps absent-env CI matrix | `.github/workflows/tests.yml` (new), `pyproject.toml` `[project.optional-dependencies]` | (none — gates only) | CI matrix runs `requirements.txt`-only env + dinov3-extras env, both green |
| 1 | `CustomWeights` bug fix | `src/models/classification/resnet_base/custom_weights.py` | 4 regression (failing-test-first, see §bug-fix) | Phase 0a green; pre-fix CI artifact (red) committed to PR description |
| 2 | Mixup utility port | `src/utils/mixup.py` | 9 unit | `pytest tests/unit/test_mixup.py` green; **no `/tmp` writes** |
| 3 | DINOv3 model | `src/models/dinov3.py`, `src/models/__init__.py` | 7 unit (mocked HF) + 1 transform integration | Stub-backbone shape contract verified; 518-vs-224 transform contract test green (see F-16, F-37) |
| 4 | DINOv3Linear (linear-probe MVP) | `src/algorithms/dinov3_linear.py`, `src/algorithms/__init__.py`, `configs/config_dinov3_linear.yaml` | 4 unit + 2 integration | 1-epoch CPU fit produces `.ckpt` under `tests/_artifacts/`; `freeze_backbone=False` and `peft.enable=True` raise `NotImplementedError` |
| 5 | Inference loader v1 (MD-Cls `.ckpt` only) | `src/models/classification/dinov3_base/{__init__,base_classifier,custom_weights}.py`, `src/models/classification/__init__.py` | 5 unit + 1 integration + 2 fail-loud negative | MD-Cls ckpt from Phase 4 loads `strict=True`; SINCHI flavors raise `NotImplementedError` |
| 6 | Full fine-tune toggle (lift guard) | `src/algorithms/dinov3_linear.py` | 3 unit (grad flow, dual param groups) | Backbone grads non-zero with `freeze_backbone: False`; linear-probe (Phase 4) seed-fixed loss unchanged |
| 7 | PEFT (LoRA + PiSSA, lift guard) | `src/algorithms/dinov3_linear.py`, `requirements-dinov3.txt` adds `peft` | 5 unit + 2 integration (incl. ckpt-roundtrip in-phase per F-33/architect-3 R6.3) | Target-modules validated against `named_modules()` at runtime; LoRA trainable-param ratio < 5% |
| 8 | Inference loader v2 (SINCHI raw + SINCHI PEFT) | `src/models/classification/dinov3_base/base_classifier.py` (extend `_extract_state_dict`) | 6 unit + 2 integration with **real** SINCHI key inspection fixture | Gated on a `tests/_artifacts/sinchi_keys_real.txt` artifact produced from a real SINCHI checkpoint inspected by a human reviewer (F-15, F-38) |
| 9 | Config + docs | `configs/config.yaml`, `docs/usage.md`, `docs/dinov3.md`, `README.md`, `requirements.txt` (comment-only) | 2 config-validation + 1 doc-snippet doctest | `Munch`-loaded dict equality (not yaml.safe_dump hash) per F-19 |

All phases must keep Phase 0a's 6 baseline tests green. Phases 0b and 9 add gates
but never modify previously-shipped behavior.

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

**Ordering rationale (testing-first, updated).**

- **Phase 0a first** because nothing else is falsifiable without it. Cut to the
  minimum bootstrap (F-10): 6 baseline tests, no coverage threshold, no new CI
  job, no extras_require. Provable in < 1 day.
- **Phase 0b separate** because coverage gates and CI workflow changes are
  quality-of-life, not DINOv3 prerequisites. The team can ship the bug fix and
  the linear-probe MVP without Phase 0b; Phase 0b can land any time after 0a
  and before 9.
- **Phase 1 (bug fix) second** (F-32). Land minimal regression scaffolding first
  (Phase 0a), then bug fix. The PR for Phase 1 MUST include a CI artifact
  (build log) showing the new test red on the pre-fix commit, attached to the
  PR body. Operationalized as: PR description includes a link to the failing
  GitHub Actions run on the parent SHA, OR a `git notes` annotation with the
  raw `pytest` red output. This satisfies F-20.
- **Phase 2 (mixup)** — pure utility, no callers, highest test-density-per-LOC.
- **Phase 3 (model) before Phase 4 (algorithm)** so model contract is verified
  in isolation under the stub HF backbone.
- **Phase 4 (linear-probe MVP)** is the **Gate 1** ship target (adopting
  architect-1's MVP framing). Full-FT and PEFT guarded by
  `NotImplementedError` (F-12, F-13). The guards are themselves tested:
  `tests/unit/test_dinov3_linear_algorithm.py::test_finetune_path_raises_until_phase_6` and
  `::test_peft_path_raises_until_phase_7`.
- **Phase 5 (loader-v1)** immediately after Phase 4 (F-11, F-33). MD-Cls
  `.ckpt` flavor only — depends only on Phase 4 artifacts; closes the
  train→save→load→infer MVP loop with zero external fixtures.
- **Phase 6 (full-FT) and Phase 7 (PEFT)** sequential after the MVP loop
  closes. Each lifts a guard added in Phase 4. Failing-test-first applies
  to each: the test for the new branch lives in the PR that ADDS the new
  branch, demonstrably red on the parent SHA.
- **Phase 8 (loader-v2)** after PEFT. Hard-gated on a real SINCHI key
  inspection (F-15, F-38) — see §inference-loader-phase.
- **Phase 9 (config + docs)** last so docs reflect as-built.

**Entry/exit criteria (compact table).**

| Phase | Entry | Exit |
|---|---|---|
| 0a | `git status` clean | `pytest tests/baseline -q` exits 0 |
| 0b | 0a green | CI workflow file present; both env-matrix legs green on `main` |
| 1 | 0a green; failing-test artifact captured on parent SHA | 4 regression tests green; Phase 0a still green |
| 2 | 1 green | `pytest tests/unit/test_mixup.py -q` green |
| 3 | 2 green | Model unit tests green under stub HF; transform test asserts `(B,3,518,518)` reaches `DINOv3Classifier.forward` while `Plain` still receives `(B,3,224,224)` |
| 4 | 3 green | `tests/integration/test_dinov3_linear_cli.py` writes `.ckpt` under `tests/_artifacts/` (NOT `/tmp`); both guards raise |
| 5 | 4 green | MD-Cls ckpt loads `strict=True` (modulo declared buffers); SINCHI-flavor input raises `NotImplementedError` |
| 6 | 5 green; pre-fix red artifact for full-FT branch | Backbone grad non-zero; linear-probe loss-curve seed-stable |
| 7 | 6 green; `peft` installable; pre-fix red artifact for PEFT branch | Target-modules validated at runtime; LoRA trainable-ratio < 5%; in-phase ckpt-roundtrip (per architect-3 R6.3) |
| 8 | 7 green; `tests/_artifacts/sinchi_keys_real.txt` committed, signed-off by reviewer | All three flavors load; fail-loud assertions cover missing-classifier-key, hidden-size-mismatch, unknown-envelope |
| 9 | 8 green | Default config dict-equal to pre-phase (F-19); doctest snippet executes under stub HF |

**Dependency graph (updated).**

```
0a ──┬──► 1  (bug fix, parallel-safe)
     ├──► 2  (mixup, parallel-safe)
     └──► 3 ──► 4 ──┬──► 5 (loader-v1)  ── (Gate 1 MVP ship: 0a+1+2+3+4+5)
                   ├──► 6 (full-FT)
                   │      └──► 7 (PEFT) ──► 8 (loader-v2) ──► 9 (docs)
                   └──► 0b (CI extras matrix; can land any time after 4)
```

Phases 1, 2, 0b are parallel-safe after their predecessors.

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

### What lands
- **NEW** `src/utils/mixup.py` — verbatim port of `noisy_same_class_mixup` +
  `efficient_batch_same_class_mixup` from `Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py`.
  Header docstring records source path + commit SHA (provenance).
- No `__init__.py` edit (matches existing repo pattern; agreed with architect-3 R1.1).

### Test plan (load-bearing)

**File:** `tests/unit/test_mixup.py`. All tests CPU-only, zero `/tmp` writes,
explicit seeding (`np.random.seed(0)` AND `torch.manual_seed(0)` AND `random.seed(0)`
via the autouse `seed_all` fixture from Phase 0a).

| Test name | Assertion |
|---|---|
| `test_noisy_same_class_mixup_preserves_output_shape` | `out.shape == emb.shape` |
| `test_noisy_same_class_mixup_preserves_labels_unchanged` | `torch.equal(out_labels, in_labels)` |
| `test_noisy_same_class_mixup_pairs_only_within_class` | Per-sample partner shares label. **Determinism fix (F-45):** monkey-patch `torch.randint` to return a recorded sequence rather than relying on a single seed; the test reads the recorded partner index and asserts `labels[i] == labels[partner_i]`. Avoids cross-numpy/torch RNG drift. |
| `test_noisy_same_class_mixup_singleton_class_passthrough` | Single-member class → output equals input |
| `test_noisy_same_class_mixup_lambda_clamped_to_open_unit_interval` | **Reworked (F-45):** instead of back-solving λ from output, monkey-patch `np.random.beta` to a stub that returns 1000 controlled values spanning `[-0.5, 1.5]`; assert all values reaching the mix line are clipped to `[0.1, 0.9]` by inspecting captured call args |
| `test_noisy_same_class_mixup_gaussian_noise_seeded_value` | Seeded, exact via `pytest.approx(abs=1e-6)`; golden constant pinned to the `numpy>=1.24,<2.0` / `torch>=2.0,<2.3` band declared in `pyproject.toml` test extras. Regenerate-on-version-bump policy documented in `tests/README.md`. |
| `test_noisy_same_class_mixup_dropout_fraction_within_tolerance` | `noise_type="dropout"`, `dropout_prob=0.5`; assert fraction of exact zeros ∈ `[0.4, 0.6]` over `(B=64, D=128)`. Statistical, robust across seeds. |
| `test_noisy_same_class_mixup_both_combines_modes` | `noise_type="both"` differs from `dropout`-only AND from `gaussian`-only under the same seed |
| `test_efficient_batch_same_class_mixup_no_op_when_no_same_class_pair` | `labels=[0,1,2,3]` → permutation impossible → output equals input |

### Invocation
```bash
pytest tests/unit/test_mixup.py -v
```
**No coverage gate in Phase 2** (F-10 / F-18 separation). The `--cov-fail-under=100`
gate moves to Phase 0b where it belongs.

### Artifact / fixture root
Any temporary writes use `tmp_path_factory` configured by `pyproject.toml`'s
`tmp_path_retention_policy = "failed"` AND `[tool.pytest.ini_options].basetemp = "tests/_pytest_tmp"`.
**Repo-local; no `/tmp`** (F-14).

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

Split into **Phase 3 (model)** and **Phase 4 (algorithm, linear-probe MVP)**.

### Phase 3 — DINOv3 model

**What lands:** `src/models/dinov3.py` (~90 LOC, per research §Step 2) +
1-line addition to `src/models/__init__.py`.

**Lazy-import contract (mandatory):** `transformers` is imported INSIDE
`DINOv3Classifier.__init__`, mirroring `speciesnet_tf.py`. Module top imports
are only `torch` + `torch.nn`. This contract is tested in Phase 0a
(`test_top_level_imports_succeed_without_transformers`).

#### Stub-backbone fixture (single primitive for Phases 3–8)
`tests/conftest.py::mock_hf_automodel` monkey-patches `transformers.AutoModel.from_pretrained`
to return a `StubBackbone` exposing:
- `.config.hidden_size = 64`
- `forward(pixel_values=T(B,3,H,W)) → SimpleNamespace(last_hidden_state=T(B, N_tokens, 64))`
  where `N_tokens = 1 + (H//16)*(W//16)` so the stub produces the correct
  token count for both 224 (197 tokens) AND 518 (1+32×32=1025 tokens).

A second fixture `mock_hf_automodel_with_named_linears` adds child modules named
exactly `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` (note:
NOT `gate_proj` — verifying that the runtime target-module validator from F-05
fires when configured targets do not match the model's actual modules).

#### Test file: `tests/unit/test_dinov3_model.py`

| Test name | Assertion |
|---|---|
| `test_dinov3_classifier_infers_hidden_size_from_backbone_config` | `model.embedding_dim == 64` |
| `test_dinov3_classifier_raises_when_hidden_size_missing` | `del config.hidden_size`; `pytest.raises(ValueError, match="hidden_size")` |
| `test_dinov3_classifier_freeze_backbone_disables_grad` | Default → `feature.*` `requires_grad=False`, `classifier.*` `requires_grad=True` |
| `test_dinov3_classifier_unfreeze_backbone_enables_grad` | After `unfreeze_backbone()`, `feature.*` `requires_grad=True` |
| `test_dinov3_classifier_forward_returns_logits_shape_B_x_numcls` | Asserts on **both** `(2,3,224,224)` and `(2,3,518,518)` input — covers F-16 |
| `test_dinov3_classifier_encode_returns_cls_token` | `encode(x).shape == (B, 64)` |
| `test_dinov3_classifier_classifier_head_init_deterministic_under_seed` | Seeded, `pytest.approx(abs=1e-6)` |

#### Test file: `tests/integration/test_dinov3_dataset_transform.py` (NEW per F-16, F-37)

| Test name | Assertion |
|---|---|
| `test_dinov3_dataset_emits_518_when_image_size_518_configured` | Construct `Custom_Crop` with `image_size=518` (the transform-injection point chosen below); assert the first batch yields tensor shape `(B,3,518,518)`. |
| `test_resnet_dataset_emits_224_when_default_config_used` | Construct `Custom_Crop` with the unmodified `configs/config.yaml`; assert first batch yields `(B,3,224,224)`. **Regression guard** that proves ResNet path's transform is untouched. |

**Transform-injection decision (F-37, was deferred by architect-3):** add a
single optional `image_size: int = 224` kwarg to `Custom_Crop`'s constructor,
threaded through to `Classification_Inference_Transform`. Default preserves
ResNet behavior bit-identically; `configs/config_dinov3_linear.yaml` sets
`image_size: 518`. This is a 2-line additive change to `src/datasets/custom.py`
landing in **Phase 3** (so the integration test above is meaningful). Guarded by
the Phase 0a `test_imports_unchanged.py::test_default_config_yaml_round_trips`
regression test (which is dict-equality, not yaml.safe_dump hash, per F-19).

### Phase 4 — DINOv3Linear (linear-probe MVP)

**What lands:**
- `src/algorithms/dinov3_linear.py` (~180 LOC). **`freeze_backbone=False` raises
  `NotImplementedError("full fine-tune lands in Phase 6")`** and
  **`peft.enable=True` raises `NotImplementedError("PEFT lands in Phase 7")`**
  (F-12, F-13). Each guard is its own LOC and has its own test (below).
- 1-line `src/algorithms/__init__.py` addition.
- `configs/config_dinov3_linear.yaml` (minimal: linear probe, mixup off, peft
  off, `image_size: 518`).
- **No `requirements.txt` edit yet** (deferred to Phase 0b's optional-deps
  policy — see F-34 / architect-3's extras approach, adopted).

#### Test file: `tests/unit/test_dinov3_linear_algorithm.py`

| Test name | Assertion |
|---|---|
| `test_dinov3_linear_init_via_models_registry` | `isinstance(learner.net, DINOv3Classifier)` |
| `test_dinov3_linear_training_step_returns_finite_loss` | `loss.item()` finite, > 0 |
| `test_dinov3_linear_configure_optimizers_returns_adamw_and_cosine` | `AdamW` + `CosineAnnealingLR`, `T_max == num_epochs` |
| `test_dinov3_linear_frozen_only_trains_classifier_params` | `sum(p.requires_grad for p in learner.parameters()) == 2` |
| `test_finetune_path_raises_until_phase_6` | `freeze_backbone=False` → `NotImplementedError` |
| `test_peft_path_raises_until_phase_7` | `peft.enable=True` → `NotImplementedError` |

#### Test file: `tests/integration/test_dinov3_linear_cli.py`

| Test name | Assertion |
|---|---|
| `test_dinov3_linear_cli_writes_checkpoint_after_one_epoch` | `pl.Trainer(max_epochs=1, accelerator="cpu", default_root_dir=str(tmp_path_factory.mktemp("dlcli")))` writes ckpt; `tmp_path_factory`'s root is `tests/_pytest_tmp/` per Phase 0a basetemp config (F-14) |
| `test_dinov3_linear_cli_checkpoint_loads_strict_into_fresh_model` | `learner2.load_state_dict(ckpt["state_dict"], strict=True)` |
| `test_dinov3_linear_save_hyperparameters_does_not_pickle_automodel` | Following architect-3 R4.6: ckpt unpicklable in a subprocess where `transformers` is uninstalled. **Real absent-package test (F-17):** spawn `subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": ...})` against a venv created on-the-fly by `uv venv` containing only `requirements.txt` deps. See §test-strategy. |

### Artifact produced
`tests/_artifacts/dinov3_linear_md_ckpt.ckpt` — committed via `git lfs` (size ~5
KB with the stub backbone), consumed by Phase 5.

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

### What lands (revised per F-12)
Real implementation deltas, not test-only:
- **EDIT** `src/algorithms/dinov3_linear.py`: remove the
  `NotImplementedError` for `freeze_backbone=False`; introduce dual param-groups
  in `configure_optimizers` (`lr_feature` vs `lr_classifier`); call
  `self.net.unfreeze_backbone()` in `setup()`.
- **EDIT** `src/algorithms/dinov3_linear.py`: emit `on_train_start` log with
  trainable-parameter count + memory-band warning (architect-3 R5.1, adopted).

### Failing-test-first proof
The PR that adds Phase 6 must include a CI artifact showing
`test_unfreeze_backbone_includes_backbone_params_in_trainable_set` and
`test_finetune_path_raises_until_phase_6` BOTH red on the parent SHA (the first
because the branch doesn't exist, the second because the guard is being
removed). After the patch, the guard test is **renamed** to
`test_finetune_path_works_with_freeze_backbone_false` and asserts the new
behavior. The renaming is itself in the same commit so the test history is
auditable.

### Test plan
**File:** `tests/unit/test_dinov3_full_finetune.py`

| Test name | Assertion |
|---|---|
| `test_unfreeze_backbone_includes_backbone_params_in_trainable_set` | `len([p for p in learner.parameters() if p.requires_grad]) > 2` |
| `test_full_finetune_backbone_grad_is_nonzero_after_backward` | After `loss.backward()`, some `net.feature.*` param has `grad.abs().sum() > 0` |
| `test_configure_optimizers_returns_two_param_groups_with_distinct_lrs` | `len(opt.param_groups) == 2` AND `opt.param_groups[0]['lr'] != opt.param_groups[1]['lr']` |

### Linear-probe regression (per architect-1 / architect-3 invariant)
`tests/integration/test_dinov3_linear_loss_curve_seed_stable.py::test_phase4_linear_loss_unchanged_by_phase6`
— with `freeze_backbone: True`, 1-epoch CPU loss curve is identical (`atol=1e-4`)
to the snapshot saved by Phase 4. Guards against Phase 6 accidentally perturbing
the frozen path.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

### What lands (revised per F-13, F-05, F-46)
- **EDIT** `src/algorithms/dinov3_linear.py`: implement `_wrap_with_peft()` per
  research §Step 4. Lazy `from peft import LoraConfig, get_peft_model, TaskType`
  inside the method, NOT at module top. Guard removal is the failing-test-first
  proof (analogous to Phase 6).
- **NEW** `src/algorithms/_peft_targets.py` (~15 LOC): a single function
  `validate_target_modules(model: nn.Module, requested: list[str]) -> list[str]`
  that walks `model.named_modules()`, returns the intersection, and raises
  `ValueError("None of {requested} match any module in the backbone. "
  "Available modules ending in '_proj': {found_proj_names}. "
  "See docs/dinov3.md#peft-target-modules.")` if the intersection is empty.
  Addresses F-05 (industry comparison) and F-46 (error taxonomy).
- **NEW** Lightning hook `DINOv3Linear.on_save_checkpoint(self, checkpoint)`
  embeds `checkpoint["dinov3_peft_config"] = self._peft_config_dict` so
  loader-v2 (Phase 8) can auto-re-wrap without external JSON (F-06, architect-3
  R6.6, adopted).
- **NEW** `requirements-dinov3.txt` gains `peft>=0.10.0,<0.14.0` (architect-3's
  pin, adopted per F-21 dependency policy).

### Test plan
**File:** `tests/unit/test_dinov3_peft.py`

| Test name | Assertion |
|---|---|
| `test_peft_lora_wrap_reduces_trainable_param_count_by_at_least_95pct` | `trainable_after / trainable_before < 0.05` (with `mock_hf_automodel_with_named_linears`) |
| `test_peft_pissa_method_sets_init_lora_weights_to_pissa_niter_4` | Spy on `LoraConfig` ctor; assert `init_lora_weights == "pissa_niter_4"` |
| `test_peft_classifier_head_remains_outside_peft_wrap` | `isinstance(learner.net.classifier, nn.Linear)` AND no `base_model` attr on classifier |
| `test_validate_target_modules_raises_when_no_match` | Configure `target_modules=["nonexistent"]`; assert ValueError with `match="None of"` |
| `test_validate_target_modules_returns_intersection_when_partial_match` | Stub lacks `gate_proj`; configure all 7; assert returned list is the 6 that match AND a warning is emitted (`pytest.warns(UserWarning, match="gate_proj")`) |

**File:** `tests/integration/test_dinov3_peft_ckpt_roundtrip.py` (per F-33,
architect-3 R6.3 — in-phase, not deferred to loader)

| Test name | Assertion |
|---|---|
| `test_peft_lora_checkpoint_roundtrip_preserves_adapter_weights` | Train 1 step → save Lightning `.ckpt` → load via `DINOv3Linear.load_from_checkpoint` → `torch.equal` on every `lora_A`/`lora_B` tensor |
| `test_peft_checkpoint_embeds_peft_config_metadata` | After `trainer.save_checkpoint(...)`, `torch.load(p, weights_only=False)["dinov3_peft_config"]` is a dict with keys `{r, lora_alpha, target_modules, init_lora_weights, task_type}` |

**Real absent-package test for `peft`** (F-17):
`tests/integration/test_peft_absent_environment.py::test_peft_enable_false_does_not_require_peft`
uses `uv venv` to construct a clean env with only `requirements.txt`, runs a
subprocess `python -c "from src.algorithms import DINOv3Linear; ..."` with
`peft.enable=False`, asserts exit 0 and that `peft` is not in `sys.modules`
inside the subprocess (captured via the script printing `json.dumps(list(sys.modules.keys()))`).

### Artifact produced
`tests/_artifacts/dinov3_linear_peft_md_ckpt.ckpt` — embedded `dinov3_peft_config`
verified; consumed by Phase 8.

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

**Split** into Phase 5 (loader-v1: MD-Cls only) and Phase 8 (loader-v2: SINCHI
formats) per F-11, F-33.

### Phase 5 — loader-v1 (MD-Cls `.ckpt` only)

**What lands:**
- `src/models/classification/dinov3_base/{__init__.py, base_classifier.py, custom_weights.py}`.
- `_extract_state_dict` handles ONLY the `{"state_dict": ..., "net."-prefix}` shape.
  SINCHI flavors raise `NotImplementedError("SINCHI checkpoint flavors land in Phase 8")`.
- 1-line guarded re-export in `src/models/classification/__init__.py`. **Guard
  caveat (F-23):** `try: from .dinov3_base import * except ImportError as e:
  if "transformers" not in str(e) and "peft" not in str(e): raise`. Catches
  ONLY the optional-dep ImportErrors; bubbles any real import bug. Documented
  in the file's header comment.

**Tests** (`tests/unit/test_dinov3_custom_weights_state_dict_extract.py` +
`tests/integration/test_dinov3_custom_weights_v1.py`):

| Test name | Assertion |
|---|---|
| `test_extract_state_dict_strips_net_prefix_for_md_cls_ckpt` | Pure unit on `_extract_state_dict` |
| `test_extract_state_dict_accepts_bare_tensor_dict` | Fallback |
| `test_extract_state_dict_raises_on_unknown_envelope_with_actionable_message` | Error message includes `list(ckpt.keys())[:10]` (architect-3 R7.3, adopted; F-46) |
| `test_extract_state_dict_raises_on_sinchi_flavor_input` | `model_state_dict` envelope → `NotImplementedError` with phase pointer |
| `test_extract_state_dict_raises_on_corrupt_checkpoint` | 16-byte garbage file → `RuntimeError` with path + "is this a torch checkpoint?" (architect-3 R7.2, F-44) |
| `test_custom_weights_loads_md_cls_ckpt_and_forwards` | Loads Phase 4 artifact; `strict=True` minus declared buffers; forward `(1,3,518,518) → (1, num_cls)` |
| `test_custom_weights_raises_on_missing_classifier_key` | Construct ckpt missing `classifier.weight`; assert `ValueError(match="classifier")` (F-46, architect-3 R7.6 — fail loud, don't trust `strict=False`) |
| `test_custom_weights_raises_on_hidden_size_mismatch` | `state["classifier.weight"].shape[1] != net.embedding_dim` → `ValueError` (architect-3 R7.7, F-46) |
| `test_custom_weights_results_generation_schema_unchanged` | Returns dict with exactly `{"img_id","prediction","class_id","confidence","all_confidences"}` |

**Test for guarded `__init__.py`** (`tests/regression/test_classification_init_guard.py`):
- `test_classification_init_loads_when_transformers_absent`: subprocess in
  `uv venv`-built clean env; assert `from src.models import classification`
  succeeds AND `classification.CustomWeights` resolves AND
  `hasattr(classification, "DINOv3CustomWeights") == False`.
- `test_classification_init_propagates_non_dep_importerror`: inject a syntax
  error into a copy of `dinov3_base/base_classifier.py` (via `monkeypatch.syspath_prepend`
  on a fixture-shadow dir); assert the import RAISES (not silently swallowed).

### Phase 8 — loader-v2 (SINCHI raw + SINCHI PEFT)

**Hard gate (F-15, F-38):** the PR may NOT merge until
`tests/_artifacts/sinchi_keys_real.txt` is committed. This file is a plain-text
dump of `list(torch.load(real_sinchi_ckpt, weights_only=False).keys())` AND
`list(torch.load(real_sinchi_ckpt)['model_state_dict'].keys())[:50]` for ONE
real SINCHI raw `.pt` and ONE real SINCHI PEFT `.pt`. The file's header includes
the checkpoint provenance (source path, SHA-256, date inspected, reviewer
initials). Synthetic-only tests are insufficient (F-15).

**What lands:**
- **EDIT** `src/models/classification/dinov3_base/base_classifier.py`: extend
  `_extract_state_dict` to handle `{"model_state_dict": {"backbone.*": ...}}` →
  remap to `feature.*`; accept `peft_config=` kwarg AND auto-read embedded
  `dinov3_peft_config` from MD-Cls's own PEFT ckpts (Phase 7 artifact).

**Tests** add to existing files:

| Test name | Assertion |
|---|---|
| `test_extract_state_dict_remaps_backbone_to_feature_for_sinchi_raw` | Real-keys fixture: load `sinchi_keys_real.txt`, construct an in-memory dict whose keys MATCH the real SINCHI key list exactly (zero-tensors for values); assert remap behavior. **Not synthetic key invention** — uses recorded real keys. |
| `test_extract_state_dict_preserves_peft_adapter_keys_in_sinchi_peft` | Same approach with real PEFT key list |
| `test_custom_weights_loads_sinchi_raw_pt` | Integration |
| `test_custom_weights_loads_sinchi_peft_pt_with_peft_config_kwarg` | Integration |
| `test_custom_weights_loads_md_cls_peft_ckpt_auto_rewraps` | Uses Phase 7 artifact (embedded `dinov3_peft_config`); no external JSON required (F-06) |
| `test_custom_weights_raises_when_lora_keys_present_but_no_peft_config` | Per architect-3 R7.4 / F-46: fail loud, do not silently load wrong-shape backbone |

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

### What lands
- **EDIT** `configs/config.yaml`: append commented `dinov3:` / `training:` /
  `label_smoothing` block (research §Step 5). Comments only — default
  `yaml.safe_load` output unchanged.
- **NEW** `configs/config_dinov3_linear.yaml`: now-enriched with documented
  full-FT / PEFT / mixup options as commented variants.
- **NEW** `configs/config_dinov3_finetune.yaml`, `configs/config_dinov3_lora.yaml`
  per architect-1 (adopted for discoverability).
- **NEW** `docs/dinov3.md` (architect-3 naming, adopted): HF auth, GPU memory,
  PEFT defaults, security note for `weights_only=False` (F-44), trusted-source
  policy.
- **EDIT** `docs/usage.md`: append "Fine-tuning with DINOv3" section.
- **EDIT** `README.md`: append one capability bullet (no reorder per architect-3 R8.2).
- **EDIT** `requirements.txt`: **comment-only** addition `# For DINOv3: pip install -r requirements-dinov3.txt` per architect-3 (adopted; F-34).
- **NEW** `CHANGELOG.md` entry summarizing Phases 0a–9 including the
  `CustomWeights` bug fix as a behavioral note (F-47).
- **Minimal Gate-1 docs already shipped in Phase 5** (per F-09, F-39): a one-page
  `docs/dinov3-experimental.md` lands in Phase 5 with HF auth + GPU memory
  caveats so MVP users aren't surprised.

### Test plan

**File:** `tests/unit/test_config_yaml.py`

| Test name | Assertion |
|---|---|
| `test_default_config_yaml_loads_to_same_dict_as_pre_phase` | **Reworked per F-19:** load `configs/config.yaml`, compare resulting dict to a pre-phase snapshot dict (committed as `tests/_artifacts/config_default_pre_phase9.json`). Direct `dict == dict`, NOT a yaml.safe_dump SHA. |
| `test_config_dinov3_linear_yaml_loads_and_has_required_keys` | All keys consumed by `DINOv3Linear.__init__` present |
| `test_all_sample_configs_parse_to_munch` | Parametrized over `configs/config_dinov3_*.yaml` |

**File:** `tests/unit/test_doc_examples.py`

| Test name | Assertion |
|---|---|
| `test_readme_dinov3_snippet_executes_under_mocked_hf` | Extract `<!-- doctest:dinov3 -->` block; exec under `mock_hf_automodel`; complete without raise |
| `test_dinov3_md_install_command_uses_extras_file` | grep `docs/dinov3.md` for `pip install -r requirements-dinov3.txt`; assert present (F-31 / architect-3 R8.5 adopted) |

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

### What lands
Patch `src/models/classification/resnet_base/custom_weights.py`:
- Add `num_layers: int = 50` (trailing kwarg) to `__init__`.
- Forward `num_layers=num_layers` (not literal `50`) to `super().__init__`.

### Failing-test-first operationalization (F-20)
The Phase 1 PR description MUST include either:
1. A link to a CI run on the parent SHA showing
   `tests/regression/test_custom_weights_bug_fix.py::test_custom_weights_accepts_num_layers_18_kwarg_without_typeerror`
   exiting with FAIL, OR
2. A `git notes` annotation (created via `git notes --ref=refs/notes/precheck add -F pretest_output.txt $PARENT_SHA`)
   containing the raw `pytest` output.

The PR template (added in Phase 0a as `.github/pull_request_template.md`)
contains a checkbox `[ ] Pre-fix failing-test artifact attached (link or git
notes ref)` — unchecked = block merge.

### Test plan

**File:** `tests/regression/test_custom_weights_bug_fix.py`

| Test name | Assertion |
|---|---|
| `test_custom_weights_default_num_layers_is_50_for_backward_compat` | `inspect.signature(CustomWeights).parameters["num_layers"].default == 50` |
| `test_custom_weights_accepts_num_layers_18_kwarg_without_typeerror` | Construct with `num_layers=18`; no `TypeError` (RED pre-fix) |
| `test_custom_weights_resnet18_checkpoint_loads_with_num_layers_18` | Uses `tests/fixtures/regression/resnet18_synth.ckpt` — synthesized at test-time from `PlainResNetClassifier(num_cls=2, num_layers=18)` + `torch.save` to `tmp_path_factory.mktemp("ckpt")` (no committed binaries; no `/tmp`) |
| `test_custom_weights_resnet50_default_unchanged` | Synthesized ResNet-50 ckpt loads with default `num_layers`; **regression on existing callers** |

### Existing tests
Phase 0a baseline must remain green.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

Round-1 plan was over-reliant on a single `state_dict` SHA-256 (F-03). Round 2
combines three orthogonal layers (synthesizing architect-3's 12-invariant table
with architect-1's prediction-snapshot proposal and my behavioral approach):

### Layer 1 — Import / surface invariants (cheap, deterministic)

| # | Invariant | Test |
|---|---|---|
| I1 | `python -c "import src"` succeeds in env with `requirements.txt` only (no transformers/peft/huggingface_hub) | `test_default_env_imports_clean` (subprocess in `uv venv` env) |
| I2 | `algorithms.__dict__['Plain']` resolves to the same class object | `test_plain_algorithm_class_identity` |
| I3 | `models.__dict__['PlainResNetClassifier']` resolves to the same class object | `test_plain_resnet_class_identity` |
| I4 | `from src.models import classification; classification.CustomWeights` resolves | `test_classification_customweights_resolves` |
| I5 | `python main.py --help` exit 0; key substrings (`--config`, `--gpus`, `--logger-type`) present | `test_main_cli_help_unchanged` |
| I6 | `yaml.safe_load(open("configs/config.yaml"))` dict equals committed pre-phase snapshot | `test_default_config_yaml_unchanged` (F-19 fix: dict equality, NOT yaml.safe_dump hash) |

### Layer 2 — Behavior snapshots (robust to version drift, per F-03)

`tests/baseline/test_plain_resnet_behavior_snapshot.py`:
- Seed `pl.seed_everything(42)` + `torch.use_deterministic_algorithms(True, warn_only=True)` + worker seeding via `DataLoader(worker_init_fn=lambda i: np.random.seed(42+i))` (F-45 reproducibility).
- Run 1 epoch with `pl.Trainer(max_epochs=1, accelerator="cpu", deterministic=True)` on the 4-image fixture.
- `test_plain_resnet_predictions_unchanged`: assert the per-image
  `argmax(logits)` and top-1 confidence (rounded to 4 decimals) on the 4-image
  fixture match a committed JSON snapshot. **Behavior-level, not byte-level.**
- A second test `test_plain_resnet_train_loss_within_tolerance` asserts the 1-epoch
  training loss is within `atol=1e-3` of the committed value.

Snapshot regeneration policy (codified in `tests/README.md`): regenerate only on
a labelled `regression-baseline-bump` PR; header line of the JSON snapshot
records `torch.__version__`, `pytorch_lightning.__version__`,
`numpy.__version__`, and the test-running Python version.

### Layer 3 — Byte-equality on files that should NEVER change (architect-3 partial-adopt, F-24 fix)

Limited to files that the plan **categorically** does not touch:
- `src/algorithms/plain.py`
- `src/models/plain_resnet.py`
- `src/datasets/custom.py` **except** the 2-line Phase 3 `image_size` kwarg
  addition — invariant updated in Phase 3 commit, but the file is otherwise
  byte-frozen
- `main.py`

NOT byte-equality on `configs/config.yaml` (Phase 9 comment-only append) or
`requirements.txt` (Phase 9 comment-only append) — those use dict-equality /
runtime-dep-set-equality respectively (F-24).

### Layer 4 — Optional-dep absent-env CI matrix (Phase 0b, F-17)

`.github/workflows/tests.yml` runs two legs:
1. `uv venv .venv-min && uv pip install -r requirements.txt -e .[test]` → run
   `pytest -m "not gpu and not slow and not requires_dinov3_deps"`.
2. `uv venv .venv-full && uv pip install -r requirements.txt -r requirements-dinov3.txt -e .[test]` →
   run `pytest -m "not gpu and not slow"`.

Leg 1 is the load-bearing one: it proves the lazy-import contract holds for
real users running `pip install -r requirements.txt`.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Test pyramid (clarified per F-36)

| Tier | Must-pass before merge? | Count | Wall-time budget |
|---|---|---|---|
| **Unit** (pure functions, mocked deps) | YES | ~35 | < 30 s |
| **Component** (single class with stub HF) | YES | ~12 | < 60 s |
| **Integration** (Lightning `fast_dev_run` on 4-image fixture) | YES | ~6 | < 3 min |
| **Regression / baseline** (Layer 1+2+3, prediction snapshots, surface) | YES | ~12 | < 30 s |
| **Slow** (1-epoch real Lightning training) | NO (nightly) | 2 | ~3 min |
| **GPU** (real DINOv3 forward) | NO (manual on dev box) | 3 | ~5 min |
| **Real-SINCHI-keys** (against `sinchi_keys_real.txt`, Phase 8 only) | YES (Phase 8 only) | 2 | < 5 s |
| **Coverage gates** (Phase 0b infrastructure, NOT a DINOv3 gate) | optional/nice-to-have per F-18 | n/a | n/a |

**CI wall-time target: < 6 minutes per env-matrix leg.**

### Fixture strategy (revised)

- **Hermetic by default.** No network, no GPU, no HF auth in default CI.
- **Stub HF backbone** is the central primitive (`mock_hf_automodel`,
  `mock_hf_automodel_with_named_linears`).
- **Tiny dataset** (4 PNGs, 32×32, ~5 KB committed) under `tests/fixtures/tiny_dataset/`.
- **Synthetic checkpoints** are OK for **shape-contract** tests only. Anything
  asserting "this is the SINCHI flavor" uses the real-key fixture
  `tests/_artifacts/sinchi_keys_real.txt` (F-15, F-38). The fixture itself is
  human-readable (no binary checkpoint committed) — it records key lists, not
  weights.
- **Synthesized-on-demand ResNet-18/50 ckpts** for Phase 1: generated in the test
  body via `PlainResNetClassifier(...).state_dict()` + `torch.save` to
  `tmp_path_factory.mktemp("...")` whose basetemp is `tests/_pytest_tmp/`. **No
  committed binaries, no `/tmp`** (F-14).

### Temp / artifact root (F-14)

`pyproject.toml` (Phase 0a):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --basetemp=tests/_pytest_tmp"
tmp_path_retention_policy = "failed"
markers = [
  "gpu: requires CUDA; deselected in CI",
  "slow: > 10 s; deselected in default CI",
  "requires_dinov3_deps: needs transformers/peft; deselected in min env",
  "requires_real_sinchi_keys: needs tests/_artifacts/sinchi_keys_real.txt; Phase 8 only",
]
```
`tests/_pytest_tmp/` is `.gitignored`. Every `tmp_path` resolves there. No
test uses `/tmp`, `mktemp`, or shell `>`.

### Absent-package tests (F-17 — strengthened)

For each optional dep (`transformers`, `peft`), the absent-package test does
NOT use `monkeypatch.setitem(sys.modules, "transformers", None)`. Instead:

```python
def test_default_env_imports_clean(tmp_path):
    venv_dir = tmp_path / "venv-min"
    subprocess.check_call(["uv", "venv", str(venv_dir)])
    py = venv_dir / "bin" / "python"
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-e", "."])
    # Now the real ImportError test:
    result = subprocess.run(
        [str(py), "-c", "import src; from src.models import classification; print('ok')"],
        capture_output=True, text=True, env={"PYTHONNOUSERSITE": "1"}
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
```

This satisfies F-17 by creating a genuinely absent-package environment, not a
`sys.modules` poisoning. Runs in Phase 0b's CI matrix.

### Determinism rules (F-45 — expanded)

Autouse `seed_all` fixture seeds `numpy`, `torch`, `random`. Tests that exercise
Lightning's DataLoader pass `worker_init_fn` so worker processes seed
identically. Tests asserting numerical values use `pytest.approx(abs=1e-6)` or
`torch.equal`. **No bare `==` on floats.** No test asserts on a value derived
from non-deterministic SDPA/CUDA kernels (CPU-only in default CI guarantees
this).

### Test naming convention

`test_<verb>_<subject>_<expectation>`. Enforced by review checklist in
`tests/README.md`.

### Coverage targets (Phase 0b only, not gating DINOv3 per F-18)

Phase 0b's CI step runs `pytest --cov=src --cov-report=term` but does NOT fail
on threshold. Targets are advisory:
- `src/utils/mixup.py`: 100%
- `src/models/dinov3.py`: ≥ 90%
- `src/algorithms/dinov3_linear.py`: ≥ 85%
- `src/models/classification/dinov3_base/`: ≥ 90%
The `--cov-fail-under=...` flag is **NOT** part of any merge gate. Adding it
would conflate "DINOv3 ships" with "we adopt coverage thresholds repo-wide"
which is a separate decision per F-18.

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

The research report §8's 10-item checklist mapped to concrete commands. **Every
item is either CI-automatable or has a documented manual GPU command;
nothing remains "user verifies visually."**

| § | Validation item | Maps to | Mode |
|---|---|---|---|
| 1 | ResNet path unchanged | `tests/baseline/test_plain_resnet_behavior_snapshot.py::test_plain_resnet_predictions_unchanged` | CI CPU |
| 2a | Linear-probe trains 1 epoch (mocked) | `tests/integration/test_dinov3_linear_cli.py::test_dinov3_linear_cli_writes_checkpoint_after_one_epoch` | CI CPU |
| 2b | Linear-probe trains 1 epoch (real DINOv3-ViT-S) | `tests/gpu/test_real_dinov3_linear_one_epoch.py::test_real_dinov3_vits_one_epoch_writes_ckpt` (marked `@pytest.mark.gpu`) | local GPU + HF login |
| 3 | Full FT backbone grads non-zero | `tests/unit/test_dinov3_full_finetune.py::test_full_finetune_backbone_grad_is_nonzero_after_backward` | CI CPU |
| 4 | Mixup toggle differs from baseline | `tests/integration/test_dinov3_linear_mixup_changes_loss.py::test_mixup_enabled_yields_different_loss_than_disabled_under_seed` | CI CPU |
| 5a | LoRA / PiSSA paths function (mocked) | `tests/unit/test_dinov3_peft.py::test_peft_lora_wrap_reduces_trainable_param_count_by_at_least_95pct` + `::test_peft_pissa_method_sets_init_lora_weights_to_pissa_niter_4` | CI CPU |
| 5b | LoRA / PiSSA paths function (real) | `tests/gpu/test_real_dinov3_peft_lora.py` | local GPU |
| 6 | Load MD-Cls ckpt | `tests/integration/test_dinov3_custom_weights_v1.py::test_custom_weights_loads_md_cls_ckpt_and_forwards` | CI CPU |
| 7 | Load SINCHI raw ckpt | `tests/integration/test_dinov3_custom_weights_v2.py::test_custom_weights_loads_sinchi_raw_pt` (real-keys fixture) | CI CPU |
| 8 | Load SINCHI PEFT ckpt | same file, `::test_custom_weights_loads_sinchi_peft_pt_with_peft_config_kwarg` | CI CPU |
| 9 | `CustomWeights` ResNet-18/50 regression | `tests/regression/test_custom_weights_bug_fix.py::*` | CI CPU |
| 10 | Lazy-import (real absent env) | `tests/regression/test_default_env_imports_clean.py` (subprocess venv per F-17) | CI CPU (matrix leg 1) |

Plus addressing F-37 (transform) and F-44 (security):
| §+ | Validation item | Maps to | Mode |
|---|---|---|---|
| 11 | DINOv3 dataloader emits 518; ResNet dataloader emits 224 | `tests/integration/test_dinov3_dataset_transform.py::*` | CI CPU |
| 12 | Corrupt-checkpoint load raises actionable error | `tests/unit/test_dinov3_custom_weights_state_dict_extract.py::test_extract_state_dict_raises_on_corrupt_checkpoint` | CI CPU |

### Acceptance gate ("done")

```bash
# CI gate — both env-matrix legs:
pytest -m "not gpu and not slow"
# manual GPU gate on dev workstation (Phase 8+):
pytest -m gpu tests/gpu/
# Phase 8 hard prereq:
test -f tests/_artifacts/sinchi_keys_real.txt
```

All three commands must exit 0 with a signed-off `sinchi_keys_real.txt`.

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

### Risk register (round-2 additions in **bold**)

| Risk | Phase | Mitigation | Rollback |
|---|---|---|---|
| `transformers` import-time side-effects pollute `Plain` path | 3, 5 | Lazy import; Phase 0a + 0b matrix tests | Revert Phase 3 |
| `peft` version incompatible | 7 | Pin `peft>=0.10,<0.14` per architect-3 R6.1 | Loosen pin |
| HF auth gating blocks CI | 3–8 | All CI uses stubs; real-weight tests `@pytest.mark.gpu` only | n/a |
| Stub vs real DINOv3 forward shape divergence | 3 | `tests/gpu/test_real_dinov3_forward_shape.py` smoke on dev box | Adjust `encode()` fallback |
| GPU OOM at 518×518 | 4–7 | Sample configs default to ViT-S + `batch_size=4` + `precision="16-mixed"` (architect-3 R8.4 adopted); doc memory table | Configuration concern |
| `CustomWeights` bug-fix changes positional ordering | 1 | Trailing kwarg with default `50` | Revert single commit |
| Lightning ckpt-hash drift across torch versions | Layer-2 baseline | **Behavior snapshot (predictions), not bytes (F-03)** | Regenerate snapshot via labelled PR |
| `_extract_state_dict` mis-detects future ckpt shape | 5, 8 | Fail-loud on unknown envelope (F-46) | Add a branch + test |
| New `dinov3_base` wildcard shadows existing symbol | 5 | Explicit `__all__`; Phase 0a regression test | Rename |
| Mixup determinism across `numpy` versions | 2 | Monkey-patch `np.random.beta` for determinism-critical tests; version-banded golden constants | Pin numpy minor |
| **SINCHI key layout differs from research report** (F-15, F-38) | 8 | Hard gate on `sinchi_keys_real.txt` artifact | Adjust remap; the artifact IS the spec |
| **Loader silently consumes corrupt ckpt** (F-44) | 5 | Wrap `torch.load` in try/except; re-raise as `RuntimeError` with path + suggestion; explicit negative test | n/a (always fails loud) |
| **PEFT target-modules don't match HF DINOv3ViT** (F-05) | 7 | Runtime `validate_target_modules` walks `named_modules()`; raises if intersection empty | Fix target list |
| **PEFT ckpt missing `peft_config` → silent wrong load** (F-46) | 8 | Detect LoRA-style keys without peft_config → raise (architect-3 R7.4 adopted) | n/a |
| **DINOv3 dataloader transform-size regression** (F-37) | 3, 5 | Phase 3 transform integration test asserts 518; Phase 0a regression test asserts 224 for default | Revert Phase 3 transform kwarg |
| **Worker-process RNG drift** (F-45) | all integration | `DataLoader(worker_init_fn=...)` seeded; deterministic algorithms enabled on CPU | n/a |
| **Reviewer skips failing-test-first proof** (F-20) | 1, 6, 7 | PR template checkbox; reviewer signs off explicitly; git notes ref optional second proof | Reject PR |
| **Tests write to `/tmp`** (F-14) | all | `pyproject.toml` sets `basetemp=tests/_pytest_tmp`; CI grep `tests/` for raw `/tmp` literal as a lint | n/a |

### Rollback procedure

Each phase = one squash-merged PR. Per architect-3 R6: reverts run in **reverse
DAG order**: 9 → 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0b → 0a. Phase 1 can be
reverted independently (no DINOv3 dependency). Phases 4 + 6 + 7 share
`src/algorithms/dinov3_linear.py`; reverting 7 alone leaves 4 + 6 intact;
reverting 6 requires also reverting 7. Documented in `docs/release_gates.md`
(architect-1 suggestion, adopted).

After each revert, run:
```bash
pytest -m "not gpu and not slow"
```
to confirm the repo lands green.

### Specific failure-mode preempts

- **Phase 3 stub-vs-real divergence** → fallback `encode()` to `pooler_output`;
  GPU test covers (`tests/gpu/test_real_dinov3_cls_extraction_shape.py`).
- **Phase 8 SINCHI keys differ from `sinchi_keys_real.txt` snapshot at a later
  date** → re-run inspection, update the artifact (recorded in a labelled
  `sinchi-keys-bump` PR), regenerate remap rule. The artifact is the source of
  truth.

---

## Rationale (testing lens)

Three meta-decisions, evolved from round 1:

1. **Phase 0 was split** (F-10). Phase 0a (pytest + 6 baseline tests) is a
   1-day cost; Phase 0b (CI matrix, coverage gates) is detachable and not on the
   DINOv3 critical path. The team can ship MVP (Phase 5) without 0b.
2. **Loader split** (F-11, F-33). MD-Cls loader after linear-probe closes the
   train→load→infer MVP loop with zero external fixtures. SINCHI loader after
   PEFT, gated on a real-key artifact.
3. **Full-FT and PEFT are real phases with real implementations** (F-12, F-13,
   F-35). The Phase 4 skeleton has `NotImplementedError` guards; later phases
   lift them. Each guard removal has its own failing-test-first proof. This
   reconciles the testing-first philosophy with the ledger's demand for
   independent phase boundaries.

---

## Discussion summary

Round 1 produced 47 NEW critiques. 11 hit architect-2 directly; 6 cross-architect
findings (F-32..F-39) and 8 missing-perspective items (F-40..F-47) overlapped my
scope. Round 2 addresses each as documented in §Evolution and §NEW This Round
below.

## Consensus points

Adopted from **architect-1**:
- **Loader split into v1/v2** (F-33). MD-Cls early, SINCHI late. Replaces my
  round-1 "all loader work in Phase 7."
- **MVP framing at Gate 1** post-loader-v1. My phases 4 + 5 are the joint MVP.
- **PR template / failing-test-first workflow operationalization** (F-20).
- **`docs/release_gates.md`** for documenting phase rollback DAG.
- **Sample configs split** (`config_dinov3_finetune.yaml`, `config_dinov3_lora.yaml`)
  for discoverability.

Adopted from **architect-3**:
- **12-invariant table approach** (Layer 1 of regression) — better than my
  round-1 single-hash approach (F-03).
- **`requirements-dinov3.txt` / extras approach** (F-21, F-34) — default
  `pip install -r requirements.txt` resolves to the same set as before.
- **Lazy try/except guard on `classification/__init__.py`** — adopted WITH
  refinement per F-23: catch only optional-dep ImportErrors, bubble real bugs;
  add `test_classification_init_propagates_non_dep_importerror` to enforce.
- **In-phase PEFT ckpt-roundtrip test** (R6.3) — round-1 I deferred to loader;
  this is moved into Phase 7.
- **Fail-loud loader assertions** (R7.4, R7.6, R7.7) — adopted; F-46 error
  taxonomy.
- **Embed `peft_config` in MD-Cls ckpts** via `on_save_checkpoint` (R6.6) —
  adopted; F-06.
- **Sample config defaults to ViT-S + batch_size=4 + fp16** (R8.4) — adopted; F-28.
- **Reverse-DAG rollback** (R6) — adopted; F-22.
- **`speciesnet_base`-style guarded import** as the proven pattern.

From the **inquisitor's industry comparison**:
- **Runtime validation of PEFT `target_modules` against `named_modules()`** (F-05) —
  new `validate_target_modules` helper added in Phase 7.

## Disagreements

- **Architect-1, "HF credentials in CI"** (F-01): I disagree. CI must be
  hermetic by default. Real-HF tests are `@pytest.mark.gpu` and `@pytest.mark.hf`,
  run only on dev workstations. Architect-1's own round-1 prose actually
  agrees with hermetic CI elsewhere — the "credentials available in CI secret"
  line is inconsistent with the rest of the proposal; I treat it as a misstatement.
- **Architect-3, byte-equality on `configs/config.yaml` and
  `requirements.txt`** (F-24): I disagree with byte-equality on these specific
  files because Phase 9 legitimately appends commented lines / a single
  comment line. Use dict-equality on YAML and runtime-dep-set-equality on
  requirements. Byte-equality is preserved ONLY on `plain.py`, `plain_resnet.py`,
  `custom.py` (with the explicit Phase 3 `image_size` exception documented),
  `main.py`.
- **Architect-3, "transform decision deferred"** (F-30, F-37): I disagree with
  deferring. Round-2 commits to the concrete injection point — a 2-line
  additive `image_size` kwarg in `Custom_Crop` landing in Phase 3, with
  regression tests in both directions.

## Evolution

| Round 1 claim | Round 2 fix | Driving critique |
|---|---|---|
| Phase 0 bundles pytest + coverage + CI in one step | Split into 0a (minimal) and 0b (coverage + CI matrix) | F-10 |
| Phase 7 is the only inference-loader phase | Split into Phase 5 (loader-v1, MD-Cls) and Phase 8 (loader-v2, SINCHI) | F-11, F-33 |
| Full-FT phase is "mostly test-only" | Full-FT is a real implementation phase that lifts a `NotImplementedError` guard added in Phase 4 | F-12, F-35 |
| PEFT phase is "already coded in Phase 4 skeleton, just verifies it" | PEFT phase is real implementation that lifts a guard; `_wrap_with_peft` body lands in Phase 7 | F-13, F-35 |
| Tests use `tmp_path` without specifying location | `pyproject.toml` sets `basetemp=tests/_pytest_tmp` (gitignored); CI lint forbids raw `/tmp` literals | F-14 |
| Synthetic-only SINCHI checkpoints sufficient | Phase 8 hard-gated on `tests/_artifacts/sinchi_keys_real.txt` from real ckpt inspection | F-15, F-38 |
| Forward test uses 224×224 only | Forward test parametrized over (224, 518); new integration test asserts dataloader emits the correct size for each config | F-16, F-37 |
| `sys.modules["transformers"]=None` for lazy-import test | Subprocess in `uv venv` clean env, real absent package | F-17 |
| `pytest-cov` thresholds part of acceptance gate | Coverage gates move to Phase 0b (optional infra); NOT a DINOv3 acceptance criterion | F-18 |
| `yaml.safe_dump(...) == <hash>` for config | Dict equality against committed `config_default_pre_phase9.json` | F-19 |
| "Tests fail first" stated but not operationalized | PR template checkbox + CI artifact link OR `git notes` annotation requirement | F-20 |
| Single-hash regression on `state_dict` | Three-layer regression: import/surface invariants + behavior snapshots (predictions) + selective byte-equality | F-03 (cross-arch) |
| No coverage of corrupt-ckpt / security-error paths | New `test_extract_state_dict_raises_on_corrupt_checkpoint`; `docs/dinov3.md` security note on `weights_only=False` | F-44 |
| Worker-process RNG not addressed | `DataLoader(worker_init_fn=...)`; pinned env recorded in snapshot headers | F-45 |
| No standardized error messages | New `validate_target_modules` + fail-loud loader assertions form an error taxonomy; documented in `docs/dinov3.md` | F-46 |
| No migration note for `CustomWeights` change | `CHANGELOG.md` entry shipped in Phase 9 explicitly covers behavioral note | F-47 |

## NEW This Round

Items added to my proposal in round 2 that were not in round 1:

1. **Phase 0a / Phase 0b split** with detachable CI matrix.
2. **Phase 5 (loader-v1)** as a dedicated phase, completing MVP at Gate 1.
3. **Phase 8 (loader-v2) real-key fixture gate** (`sinchi_keys_real.txt`).
4. **`tests/_artifacts/sinchi_keys_real.txt`** — human-readable artifact, not a
   binary checkpoint, recording real SINCHI key layouts as the spec for the
   remap rule.
5. **`validate_target_modules` helper** in `src/algorithms/_peft_targets.py`
   with runtime `named_modules()` walk + actionable error message.
6. **`on_save_checkpoint` hook embedding `dinov3_peft_config`** so loader-v2
   auto-re-wraps without external JSON.
7. **`image_size` kwarg on `Custom_Crop`** (2-line additive, Phase 3) +
   bidirectional dataloader-emits-correct-size regression test.
8. **`uv venv`-based absent-package subprocess tests** replacing
   `sys.modules` poisoning.
9. **`pyproject.toml` `basetemp=tests/_pytest_tmp` + CI lint forbidding `/tmp`**.
10. **PR template + failing-test-first workflow** operationalized with
    `git notes` or CI-log link.
11. **Three-layer regression** (surface invariants + behavior snapshots +
    selective byte-equality) replacing the single-hash approach.
12. **Behavior snapshots** of `Plain` predictions (argmax + rounded confidence)
    instead of `state_dict` SHA.
13. **Fail-loud loader assertions**: missing classifier key, hidden-size
    mismatch, LoRA keys without `peft_config`, corrupt-checkpoint detection —
    each with a negative test.
14. **`docs/dinov3-experimental.md`** shipped in Phase 5 (not waiting for Phase 9)
    with HF auth + GPU memory caveats for MVP users.
15. **Coverage gates explicitly NOT part of DINOv3 acceptance** (separated per F-18).
16. **Worker-process RNG seeding** via `DataLoader(worker_init_fn=...)` and
    `deterministic=True` in `pl.Trainer` for snapshot tests.
17. **Snapshot files carry version metadata** (torch / lightning / numpy /
    python) in their headers for regen-on-bump auditability.

STATUS: NEW=17 DISAGREEMENTS=3
