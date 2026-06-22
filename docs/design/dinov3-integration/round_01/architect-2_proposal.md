# DINOv3 integration — phased plan (architect-2: testing/verification perspective)

> **Perspective.** A phase does not exist until it is verifiable. Every phase below declares its tests, fixtures, assertions and pytest invocation BEFORE listing implementation work. The test plan is the spine; the code is the consequence.

## Executive overview

The research report at `research/what-are-the-steps-necessary-to-allow-the-repo-on-.md` already specifies WHAT to build (a 10-step, strictly-additive integration). My job is to sequence the work so that:

1. **Regression protection is in place before the first code edit lands.** The target repo currently ships **zero tests** (no `tests/`, no `pytest.ini`, no `pyproject.toml`, only `.github/workflows/deploy-docs.yml` for mkdocs). The single behavior-altering change in the whole plan — Step 8 widening `CustomWeights(num_layers=…)` — is a backward-compatibility edit on a class that has **no test covering it today**. Any phase that runs before regression snapshots are committed is flying blind.
2. **Hermetic fixtures replace network and HF auth.** DINOv3 weights are gated on HuggingFace Hub and large (300 MB – 27 GB). No CI/test can pull them. The `transformers.AutoModel.from_pretrained` calls must be **monkey-patched in tests** to a tiny stub backbone (a 3-layer `nn.Module` exposing `.config.hidden_size` and returning a `BaseModelOutput`-like object with `last_hidden_state`). Production code untouched.
3. **Every item in the research report's §8 Validation checklist (10 items) becomes either a pytest or a documented manual GPU command** — none remain as "the user checks visually".
4. **Determinism gates.** The mixup port uses `np.random.beta` (not `torch.distributions`); the classifier head uses `nn.init.normal_(std=0.01)`. Both must be seeded in tests with explicit assertions on numerical output.

The dependency graph below collapses the report's 10 steps into 9 phases ordered so that each phase's deliverables are testable in isolation and unlock the next phase's tests.

## Phase index (table)

| # | Phase | Touches | New tests (count) | Gate to next phase |
|---|---|---|---|---|
| 0 | Test scaffolding + regression snapshot | `pyproject.toml`, `tests/conftest.py`, `tests/fixtures/` | 6 baseline / smoke tests | All 6 pass on stock `main` |
| 1 | `CustomWeights` bug fix | `src/models/classification/resnet_base/custom_weights.py` | 3 unit + 1 regression | All Phase 0 tests still pass |
| 2 | Mixup utility port | `src/utils/mixup.py` | 9 unit tests (pure-function) | 100% line coverage on `mixup.py` |
| 3 | DINOv3 model | `src/models/dinov3.py`, `src/models/__init__.py` | 5 unit (mocked AutoModel) | Model instantiates + forward shape correct under mock |
| 4 | DINOv3Linear algorithm (linear-probe) | `src/algorithms/dinov3_linear.py`, `src/algorithms/__init__.py` | 4 unit + 2 integration (CLI on 4-image CSV) | 1-epoch fit + `.ckpt` written |
| 5 | Full fine-tune toggle | (same `dinov3_linear.py`) | 2 unit (grad flow) | Backbone grads non-zero with `freeze_backbone: False` |
| 6 | PEFT (LoRA/PiSSA) | (same `dinov3_linear.py`) | 4 unit + 1 ckpt-roundtrip | LoRA wrap reduces trainable params; checkpoint reload bit-identical |
| 7 | Inference loader | `src/models/classification/dinov3_base/**`, `src/models/classification/__init__.py` | 6 unit + 3 ckpt-flavor roundtrips | Three checkpoint shapes load with `strict=False` and zero unexpected non-LoRA keys |
| 8 | Config + docs | `configs/config.yaml`, `configs/config_dinov3_linear.yaml`, `docs/usage.md`, `README.md`, `requirements.txt` | 2 config-validation tests + 1 doc-snippet eval | YAML loads under `Munch`; doc example runs as a doctest under mocked HF |

All phases must keep Phase 0's 6 baseline tests green. The Phase 0 baseline is the **regression budget**: any change breaking those snapshots is rolled back.

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

**Ordering rationale (testing-first).**

- **Phase 0 first** because there are no tests in the repo. Without a tested baseline of the `Plain` ResNet pipeline, "preserving existing behavior" is unfalsifiable. Phase 0 produces (a) a `pytest` harness, (b) a 4-image hermetic dataset, (c) a "compile + 1 train step + 1 eval step" snapshot for `Plain`, (d) hash of the resulting `state_dict`. Every later phase must keep those four assertions green.
- **Phase 1 (bug fix) before any DINOv3 work** because it's the only edit to *existing* code. Sequencing it first means the regression suite is exercised on a one-line surface area before being asked to defend against larger additions. Also: a failed Phase 1 is a fast, cheap rollback.
- **Phase 2 (mixup) before Phase 3 (model)** because `mixup.py` is a pure function — testable with `torch` only, no HF, no Lightning. Highest test-density-per-LOC in the entire change. Acts as a smoke test that the new `tests/` infrastructure can carry pytorch-shaped assertions.
- **Phase 3 (model) before Phase 4 (algorithm)** so the algorithm has a real (mocked-HF) `DINOv3Classifier` to instantiate. Decouples `nn.Module` shape testing from Lightning loop testing.
- **Phase 4 (linear probe) before Phase 5/6** because linear probe is the simplest training mode and the basis on which fine-tune (5) and PEFT (6) are toggles. Tests added in 4 are reused with parametrized flags in 5 and 6.
- **Phase 7 (inference) last among code phases** because it consumes checkpoints emitted by Phases 4–6. Sequencing it last means its 3 checkpoint-flavor roundtrip tests use real artifacts produced by earlier-phase tests (the integration test in Phase 4 writes a `.ckpt` to `tests/_artifacts/` that Phase 7 then loads — this is a free end-to-end check that costs zero extra fixtures).
- **Phase 8 (config + docs) last overall** so the docs document the as-built API, not the as-designed API.

**Entry/exit criteria per phase.**

| Phase | Entry | Exit |
|---|---|---|
| 0 | `git status` clean on `main` | `pytest tests/baseline -q` exits 0; `tests/_artifacts/plain_state_dict.sha256` committed |
| 1 | Phase 0 green | New tests `test_*_custom_weights_*` pass; Phase 0 still green |
| 2 | Phase 1 green | `pytest tests/unit/test_mixup.py --cov=src/utils/mixup --cov-fail-under=100` exits 0 |
| 3 | Phase 2 green | `pytest tests/unit/test_dinov3_model.py -q` exits 0 with mocked HF |
| 4 | Phase 3 green | `pytest tests/integration/test_dinov3_linear_cli.py -q` exits 0; 1 `.ckpt` written under `tests/_artifacts/` |
| 5 | Phase 4 green | Grad-flow assertion (`test_unfreeze_backbone_yields_nonzero_grad`) passes |
| 6 | Phase 5 green | LoRA trainable-param ratio asserted < 5%; PiSSA init applied without warning capture |
| 7 | Phase 6 green | All three flavors (MD-Cls `.ckpt`, SINCHI raw `.pt`, SINCHI PEFT `.pt`) load via `DINOv3CustomWeights` |
| 8 | Phase 7 green | YAML round-trips through `Munch`; README/doc snippets execute as doctests |

**Deliverables per phase** are listed in each phase section below.

**Dependency graph.**

```
Phase 0 ──┬─► Phase 1 ──► (independent)
          ├─► Phase 2 ──► Phase 3 ──► Phase 4 ──┬─► Phase 5
          │                                     ├─► Phase 6 ──► Phase 7 ──► Phase 8
          │                                     └─► (writes .ckpt fixture consumed by 7)
```

Phases 1 and 2 can be developed in parallel by different agents after Phase 0 lands. Phases 5 and 6 can be developed in parallel after Phase 4. This ordering is *partial*, not strictly linear, which matters for the implementation team's throughput.

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

### What lands
- **NEW** `src/utils/mixup.py` — verbatim port of `noisy_same_class_mixup` and `efficient_batch_same_class_mixup` from `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py` (lines 1–171; see report footnotes [^sinchimixup], [^sinchimixupeff]). Add module-level docstring with provenance line: `# Ported from SINCHI_DINOV3 commit <sha> utils/mixup.py`.
- No edits to `src/utils/__init__.py` required (callers import explicitly per existing repo pattern).

### Test plan (load-bearing)

**File:** `tests/unit/test_mixup.py` (NEW)

| Test name | Assertion | Why it matters |
|---|---|---|
| `test_noisy_same_class_mixup_preserves_output_shape` | `out.shape == emb.shape` for `(B=8, D=16)` input | Shape contract — any future re-implementation cannot regress this |
| `test_noisy_same_class_mixup_preserves_labels_unchanged` | `torch.equal(out_labels, in_labels)` | SINCHI invariant — labels MUST NOT shift; mixup is same-class |
| `test_noisy_same_class_mixup_pairs_only_within_class` | For each `i`, the selected partner index satisfies `labels[i] == labels[partner]`; assert by monkey-patching `torch.randint` to a deterministic stub and reading back the selected indices via a hooked `embeddings.__getitem__` (or via reproducing the function's loop in the test) | The function's defining property |
| `test_noisy_same_class_mixup_singleton_class_is_identity_when_no_noise` | With `labels=[0,1]` (each class size 1), `noise_type="none"` (or `noise_std=0` + `dropout_prob=0`, `mixup_alpha=0`), `out` equals `0.5*emb + 0.5*emb == emb` | Covers the `len(same_class_idxs) == 1` branch (lines 56–58) |
| `test_noisy_same_class_mixup_lambda_clamped_to_open_unit_interval` | Seed `np.random.seed(0)`; sample λ via the public function 1000 times by capturing the mixed output and back-solving; assert `0.1 <= λ <= 0.9` always | Guards the `max(0.1, min(0.9, lam))` clamp (lines 71–72) |
| `test_noisy_same_class_mixup_gaussian_noise_matches_seeded_value` | Seed both `np.random.seed(0)` and `torch.manual_seed(0)`; with `noise_type="gaussian"`, `noise_std=0.1`, assert `out[0, 0].item() == pytest.approx(<golden_value>, abs=1e-6)` | Determinism gate — locks the RNG contract |
| `test_noisy_same_class_mixup_dropout_zeros_expected_fraction` | Seed `torch.manual_seed(0)`; with `noise_type="dropout"`, `dropout_prob=0.5`, assert fraction of exact zeros in `out` is within `[0.4, 0.6]` over `(B=64, D=128)` | Covers the `dropout` branch (lines 84–87) |
| `test_noisy_same_class_mixup_both_combines_dropout_and_gaussian` | With `noise_type="both"`, assert (a) some entries are dominated by gaussian noise (non-zero where dropout would have zeroed) and (b) RNG path is the `"both"` branch (lines 88–92) by comparing against parallel calls with only-dropout and only-gaussian seeded identically | Covers the `both` branch |
| `test_efficient_batch_same_class_mixup_no_op_when_no_same_class_pair_in_shuffle` | Construct `labels=[0,1,2,3]` so the random permutation can never produce a same-class pair; assert `out == emb` exactly | Covers the no-pair fallback in the vectorized variant |

### Invocation
```bash
pytest tests/unit/test_mixup.py -v \
    --cov=src/utils/mixup --cov-report=term-missing --cov-fail-under=100
```

### Existing tests that must continue to pass
All 6 Phase 0 baseline tests (`tests/baseline/`).

### Fixtures used
None (pure tensor inputs constructed in-test). No GPU required (all `device="cpu"`).

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

This scope-item is split across **Phase 3 (model)** and **Phase 4 (algorithm, linear-probe path)**, both documented here.

### Phase 3 — DINOv3 model

**What lands:** `src/models/dinov3.py` (skeleton from research report §Step 2, ~90 lines) + 1-line addition to `src/models/__init__.py`.

#### Test plan

**File:** `tests/unit/test_dinov3_model.py` (NEW)

**Fixture:** `tests/conftest.py` adds a `mock_hf_automodel` fixture that monkey-patches `transformers.AutoModel.from_pretrained` to return a `StubBackbone(hidden_size=64)` — an `nn.Module` exposing `.config.hidden_size=64`, taking `pixel_values=Tensor(B,3,H,W)`, returning a `SimpleNamespace(last_hidden_state=Tensor(B, 257, 64))`. This is the SINGLE HF-mocking primitive used across Phases 3–7. Documented in `tests/conftest.py` docstring.

| Test name | Assertion |
|---|---|
| `test_dinov3_classifier_infers_hidden_size_from_backbone_config` | `model.embedding_dim == 64` when stub reports `hidden_size=64`, no `embedding_dim` passed |
| `test_dinov3_classifier_raises_when_hidden_size_missing_and_no_embedding_dim` | Stub with `del config.hidden_size`; `pytest.raises(ValueError, match="hidden_size")` |
| `test_dinov3_classifier_freeze_backbone_disables_grad` | After `__init__(freeze_backbone=True)`, `all(not p.requires_grad for p in model.feature.parameters())` and `all(p.requires_grad for p in model.classifier.parameters())` |
| `test_dinov3_classifier_unfreeze_backbone_enables_grad` | After `model.unfreeze_backbone()`, `all(p.requires_grad for p in model.feature.parameters())` |
| `test_dinov3_classifier_forward_returns_logits_of_shape_BxNumCls` | `model(torch.zeros(2,3,224,224)).shape == (2, num_cls)` |
| `test_dinov3_classifier_encode_returns_cls_token` | `model.encode(x).shape == (B, 64)` (CLS only, not all 257 tokens) |
| `test_dinov3_classifier_classifier_head_init_is_deterministic_under_seed` | Seed `torch.manual_seed(0)`; assert `model.classifier.weight[0, 0].item() == pytest.approx(<golden>, abs=1e-6)` and `model.classifier.bias.abs().sum().item() == 0.0` |

### Phase 4 — DINOv3Linear algorithm (linear-probe path)

**What lands:** `src/algorithms/dinov3_linear.py` (skeleton from research report §Step 4, ~180 lines) + 1-line addition to `src/algorithms/__init__.py`. **No PEFT branch yet** (Phase 6); the `_wrap_with_peft` method is implemented but the test path keeps `dinov3.peft.enable: False`.

#### Test plan

**File 1:** `tests/unit/test_dinov3_linear_algorithm.py` (NEW)

| Test name | Assertion |
|---|---|
| `test_dinov3_linear_init_instantiates_via_models_registry` | After `DINOv3Linear(conf)`, `isinstance(learner.net, DINOv3Classifier)` — confirms `models.__dict__[conf.model_name]` resolves |
| `test_dinov3_linear_training_step_returns_finite_loss` | `learner.training_step(batch, 0).item()` is finite, > 0 |
| `test_dinov3_linear_configure_optimizers_returns_adamw_and_cosine` | `opt["optimizer"].__class__.__name__ == "AdamW"`; `opt["lr_scheduler"].__class__.__name__ == "CosineAnnealingLR"`; `opt["lr_scheduler"].T_max == conf.num_epochs` |
| `test_dinov3_linear_with_frozen_backbone_only_trains_classifier` | `len([p for p in learner.parameters() if p.requires_grad]) == 2` (weight + bias of `classifier`) |

**File 2:** `tests/integration/test_dinov3_linear_cli.py` (NEW) — invokes Lightning training end-to-end on a 4-image, 2-class hermetic dataset.

| Test name | Assertion | Setup |
|---|---|---|
| `test_dinov3_linear_cli_writes_checkpoint_after_one_epoch` | After running `pl.Trainer(max_epochs=1, accelerator="cpu").fit(learner, datamodule)`, a `.ckpt` file appears under `tmp_path/weights/<log_dir>/DINOv3Linear/`; the ckpt loads via `torch.load`; `ckpt["state_dict"]` contains keys prefixed `net.feature.` and `net.classifier.` | 4-image fixture from Phase 0; mocked HF |
| `test_dinov3_linear_cli_checkpoint_loads_with_strict_true_into_fresh_model` | `learner2 = DINOv3Linear(conf); learner2.load_state_dict(torch.load(ckpt)["state_dict"], strict=True)` succeeds — no missing/unexpected keys | Same ckpt as above |

### Invocation
```bash
pytest tests/unit/test_dinov3_model.py tests/unit/test_dinov3_linear_algorithm.py \
       tests/integration/test_dinov3_linear_cli.py -v
```

### Existing tests that must continue to pass
Phase 0 baseline (`tests/baseline/`) + all Phase 1 + Phase 2 tests.

### Artifact produced
`tests/_artifacts/dinov3_linear_md_ckpt.ckpt` (committed via `git lfs` or as a tiny synthetic ckpt — see Phase 7 fixture strategy).

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

### What lands
No new files. Verify the existing `freeze_backbone: False` branch of `DINOv3Classifier` and `DINOv3Linear.configure_optimizers` correctly trains the backbone. This is mostly a **test-only** phase that proves Phase 3/4 already supports full FT.

### Test plan

**File:** `tests/unit/test_dinov3_full_finetune.py` (NEW)

| Test name | Assertion |
|---|---|
| `test_unfreeze_backbone_includes_backbone_params_in_trainable_set` | With `freeze_backbone: False`, `len([p for p in learner.parameters() if p.requires_grad]) > 2` (classifier + at least one stub-backbone param) |
| `test_full_finetune_backbone_grad_is_nonzero_after_backward` | Run `loss.backward()` on `learner.training_step(batch, 0)`; assert at least one `model.feature.*` param has `p.grad.abs().sum() > 0` |

### Invocation
```bash
pytest tests/unit/test_dinov3_full_finetune.py -v
```

### Existing tests that must continue to pass
All previous phases.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

### What lands
- `_wrap_with_peft` activated via `dinov3.peft.enable: True` in `src/algorithms/dinov3_linear.py` (already coded in Phase 4 skeleton; this phase verifies it).
- Lazy `from peft import …` confirmed to not import at module load time.

### Test plan

**File:** `tests/unit/test_dinov3_peft.py` (NEW)

**Fixture:** A second `mock_hf_automodel_with_named_linears` returning a stub backbone whose submodules include `nn.Linear` named exactly `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj` so `peft.LoraConfig(target_modules=[...])` finds matchable layers. This stub keeps `peft` actually exercised in unit tests without needing real DINOv3.

| Test name | Assertion |
|---|---|
| `test_peft_lora_wrap_reduces_trainable_parameter_count_by_at_least_95pct` | `trainable_after / trainable_before < 0.05` when LoRA `r=8` is applied to the 7 target modules of the stub |
| `test_peft_pissa_method_sets_init_lora_weights_to_pissa_niter_4` | Mock `peft.get_peft_model`; assert it is called with a `LoraConfig` whose `init_lora_weights == "pissa_niter_4"` when `method: pissa` |
| `test_peft_classifier_head_remains_outside_peft_wrap` | After wrap, `isinstance(learner.net.classifier, torch.nn.Linear)` AND `not hasattr(learner.net.classifier, "base_model")` |
| `test_peft_disabled_does_not_import_peft_module` | `import importlib, sys; "peft" not in sys.modules` after constructing `DINOv3Linear` with `peft.enable: False` (using a fresh subprocess `python -c` via `subprocess.run`) |

**File:** `tests/integration/test_dinov3_peft_ckpt_roundtrip.py` (NEW)

| Test name | Assertion |
|---|---|
| `test_peft_lora_checkpoint_roundtrip_preserves_adapter_weights` | Train 1 step with LoRA; save Lightning `.ckpt`; instantiate a fresh `DINOv3Linear` (PEFT-enabled), `load_state_dict(strict=False)`; assert `torch.equal` on every LoRA `lora_A` / `lora_B` tensor between source and reload |

### Invocation
```bash
pytest tests/unit/test_dinov3_peft.py tests/integration/test_dinov3_peft_ckpt_roundtrip.py -v
```

### Existing tests
All prior phases.

### Artifact produced
`tests/_artifacts/dinov3_linear_peft_md_ckpt.ckpt` (consumed by Phase 7).

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

### What lands
- `src/models/classification/dinov3_base/__init__.py`, `base_classifier.py`, `custom_weights.py` per research report §Step 7.
- 1-line addition to `src/models/classification/__init__.py`.

### Test plan — the three-format auto-detect is the load-bearing test

**File:** `tests/unit/test_dinov3_custom_weights_state_dict_extract.py` (NEW) — tests the pure static method `_extract_state_dict` in isolation.

| Test name | Assertion |
|---|---|
| `test_extract_state_dict_strips_net_prefix_for_md_cls_lightning_ckpt` | Input `{"state_dict": {"net.feature.foo": t, "net.classifier.bar": t}}` → output keys `["feature.foo", "classifier.bar"]` |
| `test_extract_state_dict_remaps_backbone_to_feature_for_sinchi_raw_ckpt` | Input `{"model_state_dict": {"backbone.foo": t, "classifier.bar": t}}` → output keys `["feature.foo", "classifier.bar"]` |
| `test_extract_state_dict_preserves_peft_adapter_keys_in_sinchi_peft_ckpt` | Input `{"model_state_dict": {"backbone.base_model.model.q_proj.lora_A.default.weight": t, "classifier.bar": t}}` → output keys include `"feature.base_model.model.q_proj.lora_A.default.weight"` (LoRA delta preserved through the `backbone.→feature.` remap) |
| `test_extract_state_dict_accepts_bare_tensor_dict` | Input `{"feature.foo": t, "classifier.bar": t}` (no envelope) → returned unchanged |
| `test_extract_state_dict_raises_on_unknown_envelope` | Input `{"unknown_key": {}}` → `pytest.raises(ValueError, match="weights_key")` |
| `test_extract_state_dict_honors_explicit_weights_key_and_prefix_override` | `weights_key="custom"`, `weights_prefix="model."` correctly applied |

**File:** `tests/integration/test_dinov3_custom_weights_roundtrip.py` (NEW) — exercises full load → forward on synthetic fixtures.

| Test name | Assertion | Fixture |
|---|---|---|
| `test_custom_weights_loads_md_cls_lightning_ckpt_and_forwards` | `clf = DINOv3CustomWeights(weights=<md_cls_ckpt>, class_names=["a","b"])` → no missing/unexpected non-buffer keys; `clf(torch.zeros(1,3,224,224)).shape == (1, 2)` | Phase 4 artifact `dinov3_linear_md_ckpt.ckpt` |
| `test_custom_weights_loads_sinchi_raw_pt_with_backbone_prefix_remap` | Construct a synthetic SINCHI-shaped state dict (`backbone.*` keys aligned with stub backbone's parameters + `classifier.*`); save as `.pt` via `torch.save({"model_state_dict": ..., "epoch": 0, "config": {}})`; load via `DINOv3CustomWeights`; assert no unexpected keys | Synthetic, constructed in the test from `DINOv3Classifier(...).state_dict()` with key rename `feature.` → `backbone.` |
| `test_custom_weights_loads_sinchi_peft_pt_with_peft_config_kwarg` | Construct synthetic PEFT-wrapped state dict; load with `DINOv3CustomWeights(..., peft_config={"r":8,"lora_alpha":8,"target_modules":[...]})`; assert `strict=False` produces no missing keys for adapter weights | Phase 6 artifact + key rename |
| `test_custom_weights_results_generation_matches_existing_schema` | `clf.results_generation(logits, ["img1.jpg"])` returns a list of dicts with keys exactly `{"img_id","prediction","class_id","confidence","all_confidences"}` — same shape as `CustomWeights.results_generation` | Static logits tensor |
| `test_lazy_import_of_transformers_does_not_fire_when_module_imported` | In a subprocess with `transformers` uninstalled (simulated by monkey-patching `sys.modules["transformers"]=None` in a fresh interpreter), `from src.models import classification` succeeds; `DINOv3CustomWeights(...)` raises `ImportError` with substring `"pip install transformers"` | Subprocess via `subprocess.run([sys.executable, "-c", ...])` |

### Invocation
```bash
pytest tests/unit/test_dinov3_custom_weights_state_dict_extract.py \
       tests/integration/test_dinov3_custom_weights_roundtrip.py -v
```

### Existing tests
All prior phases. Critically, `tests/baseline/test_plain_resnet_inference_unchanged.py` (Phase 0) and `tests/regression/test_custom_weights_resnet50_default_unchanged.py` (Phase 1) must remain green — i.e. adding the `dinov3_base` subpackage to `src/models/classification/__init__.py` must not perturb existing wildcard imports.

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

### What lands
- `requirements.txt`: append `transformers>=4.40.0` and `peft>=0.10.0` (the latter marked optional in a comment).
- `configs/config.yaml`: append the commented `dinov3:` / `training:` / `label_smoothing` block from research report §Step 5.
- `configs/config_dinov3_linear.yaml`: NEW, fully-uncommented DINOv3 config (research report §Step 10).
- `docs/usage.md`: append "Fine-tuning with DINOv3" section per §Step 9.
- `README.md`: capability bullet + zoo-example update per §Step 9.

### Test plan

**File:** `tests/unit/test_config_yaml.py` (NEW)

| Test name | Assertion |
|---|---|
| `test_default_config_yaml_round_trips_through_munch_unchanged` | `yaml.safe_dump(Munch(yaml.full_load(open("configs/config.yaml")))) == <baseline_hash>` — golden hash check on the canonical default config, ensuring the appended commented block does not perturb the loaded structure (comments don't load) |
| `test_config_dinov3_linear_yaml_loads_and_has_required_keys` | `conf = Munch(yaml.full_load(...))`; assert `conf.algorithm == "DINOv3Linear"`, `conf.model_name == "DINOv3Classifier"`, `conf.dinov3.freeze_backbone in (True, False)`, etc. — every key referenced by `DINOv3Linear.__init__` is present |

**File:** `tests/unit/test_doc_examples.py` (NEW)

| Test name | Assertion |
|---|---|
| `test_readme_dinov3_custom_weights_snippet_executes_under_mocked_hf` | Extract the code block tagged `<!-- doctest:dinov3 -->` from `README.md`; `exec` it under the `mock_hf_automodel` fixture; assert it completes without raising | Avoids stale doc examples |

### Invocation
```bash
pytest tests/unit/test_config_yaml.py tests/unit/test_doc_examples.py -v
```

### Existing tests
All prior phases.

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

### What lands
Patch `src/models/classification/resnet_base/custom_weights.py`:
- Add `num_layers: int = 50` to `__init__` signature.
- Pass `num_layers=num_layers` (instead of literal `50`) to `super().__init__`.

That's the entire diff (~2 logical lines).

### Test plan — must land BEFORE the patch, fail, then pass

**File:** `tests/regression/test_custom_weights_bug_fix.py` (NEW; this is Phase 1 of the timeline but the bug-fix scope item is listed here)

| Test name | Assertion |
|---|---|
| `test_custom_weights_default_num_layers_is_50_for_backward_compat` | `inspect.signature(CustomWeights).parameters["num_layers"].default == 50` (post-fix) |
| `test_custom_weights_accepts_num_layers_18_kwarg_without_typeerror` | `CustomWeights(weights=None, class_names=["a","b"], num_layers=18)` constructs without `TypeError` (will fail pre-fix because positional `num_layers=50` is hard-coded inside `super().__init__`) — to construct without real weights we need to monkey-patch `_load_checkpoint` or pass weights=path-to-fixture |
| `test_custom_weights_resnet18_checkpoint_loads_with_num_layers_18` | Given a Phase 0 fixture `tests/_artifacts/plain_resnet18.ckpt` (a 4-image, 2-class, 1-epoch ResNet-18 Lightning ckpt), `CustomWeights(weights=ckpt, class_names=["a","b"], num_layers=18)` loads without missing/unexpected keys; forward shape correct |
| `test_custom_weights_resnet50_default_unchanged` | Given a synthetic ResNet-50 state dict (`nn.Linear(2048, 2)` head), default-arg `CustomWeights(weights=ckpt, class_names=["a","b"])` still loads with `num_layers=50` — regression on existing callers | This is the **anti-regression** test for the bug fix |

### Invocation
```bash
pytest tests/regression/test_custom_weights_bug_fix.py -v
```

### Existing tests
Phase 0 baseline.

### Sequencing requirement
The new tests must be **committed and demonstrably failing** on the pre-fix commit (`pytest --runxfail` or via CI showing red). Only then does the 2-line fix land in the next commit. This makes the bug-fix's behavioral claim auditable — required by the inquisitor's verification skill.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

The repo today has **no test that proves the `Plain` ResNet pipeline produces a given output**. Every guarantee the report makes about "existing pipeline unchanged" is currently un-falsifiable. Phase 0 fixes that.

### Phase 0 deliverables (NEW)

1. **`pyproject.toml`** (NEW) — minimal pytest config:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   addopts = "-ra --strict-markers"
   markers = [
     "gpu: requires a CUDA device (deselected by default in CI)",
     "slow: > 10s; deselect with -m 'not slow'",
   ]
   ```
   Plus `[project.optional-dependencies] test = ["pytest>=7", "pytest-cov", "pytest-mock"]`.
2. **`tests/conftest.py`** — global fixtures: `mock_hf_automodel`, `mock_hf_automodel_with_named_linears`, `tiny_dataset_csv`, `tiny_image_dir`, `seed_all`.
3. **`tests/fixtures/tiny_dataset/`** — 4 RGB images (32×32, committed as PNG, < 5 KB total), `train_annotations_cropped.csv`, `val_annotations_cropped.csv`, `test_annotations_cropped.csv` with 2 classes. Hermetic, no `batch_detection_cropping` needed (pre-cropped).
4. **`tests/baseline/`** — the regression snapshot suite:

| Test file | Test name | Assertion |
|---|---|---|
| `test_imports_unchanged.py` | `test_top_level_imports_succeed_without_optional_deps` | `from src import algorithms; from src import datasets; from src.models import classification` all succeed even if `transformers`/`peft` not installed (verified via subprocess with `sys.modules` masking) |
| `test_imports_unchanged.py` | `test_algorithms_dict_contains_plain` | `"Plain" in algorithms.__dict__` |
| `test_imports_unchanged.py` | `test_models_classification_dict_contains_customweights` | `"CustomWeights" in dir(classification)` |
| `test_plain_resnet_pipeline_smoke.py` | `test_plain_resnet18_training_completes_one_step_on_tiny_dataset` | `pl.Trainer(max_epochs=1, accelerator="cpu", fast_dev_run=True).fit(Plain(conf), datamodule)` exits 0 |
| `test_plain_resnet_pipeline_smoke.py` | `test_plain_resnet18_state_dict_hash_is_stable_after_seed_and_one_step` | Seed everything; run `fast_dev_run=True`; compute SHA-256 of `learner.state_dict()` serialized via `torch.save`; compare to golden in `tests/_artifacts/plain_state_dict.sha256`. **First run on `main` writes the golden; subsequent runs (post each phase) must match.** |
| `test_main_cli_help_unchanged.py` | `test_main_cli_help_text_matches_golden` | Run `python main.py --help` via `subprocess.run`; assert exit 0 and `--config`/`--gpus`/`--logger-type` substrings present. Golden snapshot of help text (with version markers stripped) committed to `tests/_artifacts/cli_help.txt` |

5. **CI integration** — extend `.github/workflows/` with a new `tests.yml` workflow (currently only `deploy-docs.yml` exists):
   ```yaml
   name: tests
   on: [push, pull_request]
   jobs:
     pytest:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: {python-version: "3.10"}
         - run: pip install -r requirements.txt -e .[test]
         - run: pytest -m "not gpu and not slow" --cov=src --cov-report=term
   ```

### Invariants enforced going forward

1. `Plain` state-dict hash unchanged → guards `plain.py`, `plain_resnet.py`.
2. CLI help text unchanged → guards `main.py` Typer signature.
3. Top-level imports succeed without `transformers`/`peft` → guards the lazy-import contract.
4. `algorithms.__dict__["Plain"]` resolvable → guards the registry pattern.

Any phase whose changes break any of these is **rolled back** automatically by the CI gate.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Test pyramid

| Tier | Count (post-all-phases) | Runtime budget | When run |
|---|---|---|---|
| **Unit** (pure functions, mocked deps) | ~35 | < 30 s total | every commit, CI |
| **Component** (single class with stub HF) | ~12 | < 60 s total | every commit, CI |
| **Integration** (Lightning `fast_dev_run` end-to-end on 4-image fixture) | ~6 | < 3 min total | every commit, CI |
| **Regression / baseline** (snapshot hashes, CLI help) | 6 | < 30 s | every commit, CI |
| **Slow** (1-epoch real Lightning training) | 2 | ~3 min | nightly CI, local manual |
| **GPU** (forward on real DINOv3 weights) | 3 | ~5 min | manual on dev workstation; not in CI |

Total CI runtime target: **< 6 minutes**.

### Fixture strategy

- **Hermetic by default.** No fixture pulls from the network. No fixture requires GPU. No fixture requires HF auth.
- **Stub HF backbone** is the single most important fixture. Defined in `tests/conftest.py` as a class returning a `SimpleNamespace`-typed output with `last_hidden_state` of shape `(B, 257, hidden_size)`. Used by Phases 3, 4, 5, 6, 7, 8.
- **Tiny dataset** (4 PNGs + 3 CSVs, ~5 KB total) committed to `tests/fixtures/tiny_dataset/`. Reused for `Plain` regression and `DINOv3Linear` integration alike.
- **Synthetic checkpoints** constructed in-test from `DINOv3Classifier(...).state_dict()` with key renames. No need to actually train a SINCHI model to test the loader — the loader's contract is purely about key shape, which is constructable.
- **Real DINOv3 forward** (Phase 4/6 GPU tests, marked `@pytest.mark.gpu`) loads `facebook/dinov3-vits16-pretrain-lvd1689m` (smallest variant, ~80 MB) **only on the developer's workstation** with HF login; gated behind `pytest -m gpu` and not run in CI.

### Determinism rules (mandatory in every test that uses randomness)

```python
@pytest.fixture(autouse=True)
def seed_all():
    import numpy as np, torch, random
    np.random.seed(0); torch.manual_seed(0); random.seed(0)
    torch.use_deterministic_algorithms(False)  # CPU only; SDPA paths
```
Plus any test asserting on numerical output uses `pytest.approx(..., abs=1e-6)` or exact `torch.equal`, never bare `==` on floats.

### Test naming convention (enforced by review)

`test_<verb>_<subject>_<expectation>` — e.g. `test_freeze_backbone_disables_grad`. Never `test_mixup_1` or `test_it_works`. Stated explicitly in `tests/README.md`.

### Coverage targets

- `src/utils/mixup.py`: **100%** (small pure file, gated by `--cov-fail-under=100`).
- `src/models/dinov3.py`: **≥ 90%**.
- `src/algorithms/dinov3_linear.py`: **≥ 85%** (Lightning hooks under test).
- `src/models/classification/dinov3_base/**`: **≥ 90%** (loader is the highest-leverage code; key extraction logic must be exhaustively covered).
- Untouched ResNet code: no coverage target (we add regression tests, not coverage).

### What runs in CI vs. local

| Test set | CI | Local pre-commit | Nightly |
|---|---|---|---|
| Unit + component + integration + regression | ✅ | ✅ | ✅ |
| Slow (1-epoch on tiny dataset) | ⛔ | optional | ✅ |
| GPU (real DINOv3) | ⛔ | dev workstation | ⛔ |

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

The research report's §8 has 10 validation items. Mapping each to a concrete pytest or scripted command:

| § | Validation item | Maps to |
|---|---|---|
| 1 | Existing ResNet path unchanged | `tests/baseline/test_plain_resnet_pipeline_smoke.py::test_plain_resnet18_state_dict_hash_is_stable_after_seed_and_one_step` (Phase 0) |
| 2 | New linear-probe path runs and saves ckpt | `tests/integration/test_dinov3_linear_cli.py::test_dinov3_linear_cli_writes_checkpoint_after_one_epoch` (Phase 4) |
| 3 | Full fine-tune backbone grads non-zero | `tests/unit/test_dinov3_full_finetune.py::test_full_finetune_backbone_grad_is_nonzero_after_backward` (Phase 5) |
| 4 | Mixup toggle differs from baseline | New: `tests/integration/test_dinov3_linear_mixup_changes_loss.py::test_mixup_enabled_yields_different_loss_than_disabled_under_seed` (Phase 4 extension) |
| 5 | LoRA + PiSSA paths function | `tests/unit/test_dinov3_peft.py::test_peft_lora_wrap_reduces_trainable_parameter_count_by_at_least_95pct` AND `::test_peft_pissa_method_sets_init_lora_weights_to_pissa_niter_4` (Phase 6) |
| 6 | Inference loading of MD-Cls ckpt | `tests/integration/test_dinov3_custom_weights_roundtrip.py::test_custom_weights_loads_md_cls_lightning_ckpt_and_forwards` (Phase 7) |
| 7 | Inference loading of SINCHI raw ckpt | `::test_custom_weights_loads_sinchi_raw_pt_with_backbone_prefix_remap` (Phase 7) |
| 8 | Inference loading of SINCHI PEFT ckpt | `::test_custom_weights_loads_sinchi_peft_pt_with_peft_config_kwarg` (Phase 7) |
| 9 | `CustomWeights` regression (ResNet-18) | `tests/regression/test_custom_weights_bug_fix.py::test_custom_weights_resnet18_checkpoint_loads_with_num_layers_18` AND `::test_custom_weights_resnet50_default_unchanged` (Phase 1) |
| 10 | Lazy import of `transformers` | `tests/integration/test_dinov3_custom_weights_roundtrip.py::test_lazy_import_of_transformers_does_not_fire_when_module_imported` (Phase 7) AND `tests/baseline/test_imports_unchanged.py::test_top_level_imports_succeed_without_optional_deps` (Phase 0) |

**No item maps to "the user verifies manually."** The two items that intrinsically need a real GPU (§2 with real DINOv3 weights, §5 with real LoRA) have **both** a CI-friendly mocked version (in the table above) AND a `@pytest.mark.gpu` real-weights version in `tests/gpu/` that runs on the developer workstation.

### Acceptance gate (declared "done")

The integration is "done" when:
```bash
pytest -m "not gpu" --cov=src --cov-report=term tests/
```
exits 0 AND coverage thresholds in §test-strategy are met AND, on the dev workstation:
```bash
pytest -m gpu tests/gpu/
```
exits 0 with HF login present.

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

### Risk register

| Risk | Phase | Likelihood | Impact | Mitigation | Rollback |
|---|---|---|---|---|---|
| `transformers` import-time side-effects pollute `Plain` path | 3, 7 | Low | High | Lazy import inside `__init__` (mirrors `speciesnet_tf.py` pattern); Phase 0 test `test_top_level_imports_succeed_without_optional_deps` enforces | Revert Phase 3 commit; Phase 0 test catches in CI |
| `peft` version incompatible with `transformers` pin | 6 | Medium | Medium | Pin both in `requirements.txt`; Phase 6 ckpt-roundtrip test exercises real `peft` API | Loosen / pin `peft<X`; revert single line |
| HF auth gating (DINOv3 weights are gated) blocks all CI | 3–7 | High | None (we mock) | All CI tests use the stub backbone; real-weight tests are `@pytest.mark.gpu`, deselected in CI | n/a |
| Stub-backbone test passes but real DINOv3 forward shape differs | 3 | Low | Medium | One `@pytest.mark.gpu` test (`tests/gpu/test_real_dinov3_forward_shape.py`) verifies on the smallest real variant before any release | Adjust `encode()` fallback logic |
| GPU OOM at 518×518 default | 4, 5 | Medium (local) | Low (CI uses 32×32) | Doc note in `docs/usage.md` re. `precision="16-mixed"` and small `batch_size`; tests use 32×32 fixture | n/a (configuration concern) |
| `CustomWeights` bug-fix changes positional-arg ordering | 1 | Low | High | Add `num_layers` as **trailing kwarg** with default `50` → all existing positional calls unaffected; Phase 1 regression test enforces | Revert single commit |
| Lightning `state_dict` hash drift from PyTorch version upgrade | 0+ | Medium | Medium (golden must be regenerated) | Golden hash file `plain_state_dict.sha256` includes a header line with `torch.__version__` / `pl.__version__`; CI matrix pins one combo | Regenerate golden in a documented commit (`chore: refresh plain_state_dict hash for torch X.Y`) |
| `_extract_state_dict` mis-detects a future checkpoint shape | 7 | Medium | High | Tests cover all 3 known shapes + `weights_key`/`weights_prefix` overrides; raise `ValueError` with actionable message on unknown shape | Add a 4th branch + a new unit test |
| New `dinov3_base` wildcard import shadows an existing symbol in `classification` namespace | 7 | Low | Medium | Each subpackage uses explicit `__all__`; Phase 0 test `test_models_classification_dict_contains_customweights` enforces no name collision | Rename DINOv3 export |
| Mixup determinism breaks across `numpy` versions | 2 | Low | Low | Seed `np.random.seed(0)`; if the numerical assertion fails on a future numpy, regenerate golden with `np.__version__` note in fixture header | Pin numpy minor in `requirements.txt` |

### Rollback procedure

Each phase produces **one squash-merged commit** to `main` (or one PR). Rollback is:
```bash
git revert <phase-commit-sha>
pytest -m "not gpu"
```
Because every phase keeps prior-phase tests green, reverting any single phase always leaves the repo in a green state. **No phase modifies a file already modified by a prior phase in this plan**, except Phase 4–6 which all touch `src/algorithms/dinov3_linear.py` (and in that case the partial-revert strategy is documented in the phase's commit message: revert by re-applying the Phase 4 version of the file).

Phase-by-phase rollback points are explicit. There is no "all-or-nothing" merge.

### Specific failure-mode preempts

- **If Phase 3 stub-backbone tests pass but a real DINOv3 forward fails on the dev workstation**, the failure mode is almost certainly the `outputs.last_hidden_state[:, 0, :]` CLS extraction. The `encode()` fallback path (`pooler_output`) is tested in Phase 3 (`test_dinov3_classifier_encode_returns_cls_token` extended with a parametrize for the fallback). One additional GPU test `tests/gpu/test_real_dinov3_cls_extraction_shape.py` covers the real model.
- **If Phase 7's PEFT-roundtrip test passes but a real SINCHI PEFT checkpoint fails to load**, the most likely culprit is the `backbone.base_model.model.` prefix shape varying with `peft` version. The fix is in the same one method (`_extract_state_dict`), and a new branch + new test is the standard remediation.

---

## Rationale (testing lens)

Three meta-decisions deserve naming:

1. **Phase 0 (scaffolding) is non-negotiable and goes first.** A common objection — "let's just add tests as we go" — is the failure mode that produced the repo's current state (zero tests). The cost of bootstrapping pytest + 6 baseline tests on day 1 is < 1 day; the cost of trying to retrofit regression tests AFTER the DINOv3 work has landed and the diff is huge is several days plus the risk that the snapshot is taken on already-changed behavior.
2. **Bug fix (Phase 1) before any DINOv3 work.** Some plans would group the bug fix with Phase 7 (it's about `CustomWeights`-shape loading). I separate it because it touches a different file (`resnet_base/custom_weights.py`) and is a behavioral change to existing code, not an additive change. Behavioral changes to existing code have a different risk profile and deserve their own commit, their own tests-fail-first proof, and their own revert point.
3. **Stub HF backbone is the single most important fixture.** Once it exists (Phase 0 / Phase 3), Phases 3–7 all become CPU-only, network-free, < 30 s test suites. Without it, the test plan is either GPU-bound (CI failure) or network-bound (CI flake). I recommend it be the very first new test infrastructure committed after `pyproject.toml`.

### Pre-empting likely objections

- **From architect-1 (phasing).** Likely objection: "Why not collapse Phase 5 (full FT) into Phase 4?" Answer: They share an implementation file but have independent test surfaces. Splitting them produces two clean commits with two clean test additions, allowing partial rollback if full-FT-mode reveals a problem that linear-probe-mode does not. The marginal cost is one extra commit; the marginal benefit is one extra revert point.
- **From architect-3 (risk).** Likely objection: "Phase 0 adds a CI requirement that didn't exist before — is that a new failure surface?" Answer: Yes, and intentionally. The existing repo has zero CI tests (only `deploy-docs.yml`). Any new CI job introduces a possibility of red builds. That's a feature: a red build that catches a real regression is the entire point. The CI job is scoped to `not gpu and not slow` so its surface is < 6 min and uses only pip-installable deps already in `requirements.txt` plus pytest.
- **Likely objection: "the doc-snippet doctest in Phase 8 is over-engineering."** Counter: stale README examples are the single most common cause of bad first-impressions in open-source ML repos. A 5-line doctest that catches "the snippet doesn't even run" is high ROI.

## Discussion summary
*(round 1: skip)*

## Consensus points
*(round 1: skip)*

## Disagreements
*(round 1: skip)*

## Evolution
*(round 1: skip)*

## NEW This Round
*(round 1: skip)*

STATUS: NEW=0 DISAGREEMENTS=0
