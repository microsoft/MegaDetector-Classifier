# DINOv3 integration — phased plan (architect-3: risk/backwards-compat perspective)

> Perspective: **assume every change can break someone mid-experiment.** The
> entire plan is built around the invariant that a user pulling this version
> and re-running their existing `python main.py` with their existing config
> gets bit-identical training, bit-identical checkpoints, and zero new failed
> imports. DINOv3 is opt-in at every layer.

---

## Executive overview

The research report specifies *what* to build (10 steps, code skeletons). This
plan organizes them into **8 sequential phases**, each independently
revertable with a single `git revert <sha>`, each leaving the repo in a
working state, and each carrying its own numbered risk register with
likelihood × impact and mitigation. Phases are ordered so that the
highest-risk additive changes (HuggingFace, PEFT) sit **behind** the lowest-
risk preparation work (bug fix, mixup util port). The CustomWeights bug fix
is given its **own phase** because it is the only behaviour-changing edit to
existing code and must be regression-tested in isolation. The PEFT path is
gated behind an extras_require group so a user with no `peft` installed never
sees a missing-dep error unless they opt in.

The plan declares **12 invariants** of the existing repo (§Regression
protection) that become automated tests. The single load-bearing invariant
is: *with the unchanged default `configs/config.yaml`, running `python
main.py` imports zero DINOv3 code, requires zero new pip deps, and produces a
checkpoint byte-equivalent to the pre-change checkpoint (given identical seed
+ data).*

---

## Phase index (table)

| # | Phase | Touches existing code? | New runtime deps | Rollback risk | DAG predecessor |
|---|---|---|---|---|---|
| 0 | Test scaffolding + regression baseline | NO (tests only) | none | trivial | — |
| 1 | Mixup utility port | NO | none | trivial | 0 |
| 2 | CustomWeights `num_layers` bug fix | **YES** (1 file, +1 defaulted kwarg) | none | low | 0 |
| 3 | Optional-deps wiring (`requirements-dinov3.txt`, extras_require) | NO (additive only) | none in default; transformers/peft in extras | trivial | 0 |
| 4 | Model + linear-probe algorithm (frozen backbone) | NO (3 new files + 2 `__init__.py` re-export lines) | none in default; transformers loaded only on instantiation | low | 1, 3 |
| 5 | Full fine-tune toggle | NO (extends Phase 4 algorithm with a `freeze_backbone: False` branch + dual param-group optimizer) | none additional | low | 4 |
| 6 | PEFT (LoRA / PiSSA) toggle | NO (extends Phase 4 algorithm; lazy `peft` import) | none in default; `peft` in extras | medium (peft API drift) | 5 |
| 7 | Inference loader (`DINOv3CustomWeights` + 3-format auto-detect) | NO (new subpackage + 1 `__init__.py` re-export line guarded by `try/except`) | none in default | **medium-high** (auto-detect is the riskiest unit) | 4 |
| 8 | Config + docs (commented YAML block, README, usage.md, sample config) | YES — comment-only append to `configs/config.yaml` | none | trivial | 4, 7 |

Total: 8 phases. Phases 4–7 are pure adds. Phases 2 and 8 are the only ones
that touch already-shipped files; both are backward-compatible by
construction (defaulted kwarg / commented-out YAML).

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

### Ordering rationale (risk lens)

1. **Phase 0 first** (tests/regression baseline) so that every later phase has
   a "did I break the ResNet path?" check. Without this we are flying blind.
2. **Phase 1 (mixup port) before Phase 4** because the mixup utility is a
   self-contained pure function with no external deps; landing it first
   lets us write its unit tests independently and de-risks the import path
   from `src/algorithms/dinov3_linear.py` later.
3. **Phase 2 (CustomWeights bug fix) early and standalone** because it is
   the *only* edit to behaviour-relevant existing code and we want it to
   land alone so a bisect points cleanly at it if a ResNet-inference user
   reports a regression.
4. **Phase 3 before Phase 4** because the extras_require/requirements file
   must be in place before any module imports `transformers` lazily — so
   docs and CI both have the right install target before code that needs
   it lands.
5. **Phase 4 → 5 → 6** in increasing-risk order: frozen-backbone linear
   probe (no backbone gradient, smallest blast radius) → full fine-tune
   (gradient through the whole 86M-param backbone, OOM-prone) → PEFT
   (introduces a third-party library whose API has historically churned).
   Each subsequent phase only extends config branches; if Phase 6 is
   reverted, Phases 4–5 still work.
6. **Phase 7 (inference loader) after Phase 4** but parallelizable with 5/6.
   It only depends on `DINOv3Classifier` existing. We schedule it after 6
   in the linear sequence so it can also test PEFT checkpoint loading in
   the same phase.
7. **Phase 8 (config + docs) last** so docs reflect the actual code that
   shipped, not speculative API.

### Entry/exit criteria (per phase)

Every phase has the same shape:

- **Entry**: (a) predecessor phase merged, (b) regression test suite
  (Phase 0) green on `main`, (c) risk register reviewed, (d) rollback SHA
  recorded.
- **Exit**: (a) phase-specific tests green, (b) regression suite still
  green (no invariant violated), (c) rollback dry-run executed in a
  scratch worktree and verified to leave the repo green, (d) phase entry
  appended to `COVERAGE_LOG.jsonl`.

### Per-phase deliverables summary

| Phase | New files | Modified files | Test files added |
|---|---|---|---|
| 0 | `tests/regression/test_invariants.py`, `tests/conftest.py`, `tests/fixtures/ckpt_baseline_hash.txt` | none | (this IS the test scaffold) |
| 1 | `src/utils/mixup.py` | none | `tests/unit/test_mixup.py` |
| 2 | none | `src/models/classification/resnet_base/custom_weights.py` (+1 defaulted kwarg) | `tests/regression/test_customweights_num_layers.py` |
| 3 | `requirements-dinov3.txt`, `pyproject.toml` extras (if pyproject exists; else docs note) | `requirements.txt` (UNCHANGED — see Phase-3 §Files MODIFIED) | `tests/regression/test_no_dinov3_imports_in_default.py` |
| 4 | `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py` | `src/models/__init__.py` (+1 line, guarded), `src/algorithms/__init__.py` (+1 line, guarded) | `tests/unit/test_dinov3_model.py`, `tests/integration/test_dinov3_linear_probe.py` |
| 5 | none | `src/algorithms/dinov3_linear.py` (extends `configure_optimizers`) | `tests/unit/test_dinov3_finetune_param_groups.py` |
| 6 | none | `src/algorithms/dinov3_linear.py` (`_wrap_with_peft`) | `tests/integration/test_dinov3_peft_lora.py`, `tests/integration/test_dinov3_peft_pissa.py` |
| 7 | `src/models/classification/dinov3_base/{__init__.py, base_classifier.py, custom_weights.py}` | `src/models/classification/__init__.py` (+1 line, guarded) | `tests/integration/test_dinov3_inference_three_formats.py`, `tests/unit/test_extract_state_dict.py` |
| 8 | `configs/config_dinov3_linear.yaml`, `docs/dinov3.md` (new) | `configs/config.yaml` (append commented block only), `README.md` (small touch-up), `docs/usage.md` (append section) | `tests/regression/test_default_config_unchanged.py` |

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

### Goal
Port `noisy_same_class_mixup` (and `efficient_batch_same_class_mixup`)
verbatim from `Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py` to
`src/utils/mixup.py`. Pure-PyTorch + numpy. No new pip deps.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1.1 | `src/utils/__init__.py` re-exports clash (existing `from . import utils` in `src/__init__.py` chain) | LOW | LOW | Do NOT add `from .mixup import *` to `src/utils/__init__.py`. Require explicit import (`from src.utils.mixup import noisy_same_class_mixup`). Confirmed safe: existing modules in `src/utils/` are imported explicitly (e.g. `from src.utils import batch_detection_cropping` in `main.py`). |
| 1.2 | `numpy` import differs in version from SINCHI's pin → minor numerical drift in Beta-sample | LOW | LOW | Mixup is non-default (opt-in via config). Unit test asserts deterministic output given `torch.manual_seed(0)` + `np.random.seed(0)`, NOT bit-exact cross-environment. |
| 1.3 | Future code accidentally imports mixup at module top level of `algorithms/__init__.py` and pulls it onto every ResNet user's import path | LOW | LOW | Lint rule (custom test): grep that `src/algorithms/__init__.py` does NOT contain `mixup`. |

### Files MODIFIED vs CREATED
- **CREATED**: `src/utils/mixup.py`, `tests/unit/test_mixup.py`.
- **MODIFIED**: none. (Deliberately — see Risk 1.1.)

### Dependency footprint added
None. `numpy` and `torch` already required by `requirements.txt` (transitively
via `lightning`).

### Rollback
`git revert <PHASE_1_SHA>` → file deleted. Verify with:
`python -c "import src.algorithms, src.models, src.datasets; print('ok')"`
(must succeed and produce identical importable surface as pre-phase).

### DAG
Depends only on Phase 0.

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

### Goal
Land `src/models/dinov3.py` (`DINOv3Classifier` with `.feature` / `.classifier`
split) and `src/algorithms/dinov3_linear.py` (`DINOv3Linear` LightningModule,
frozen-backbone path only — full fine-tune and PEFT branches arrive in
Phases 5 and 6). Wire both into the `__dict__` registry **without forcing
import** on ResNet users.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 4.1 | `from .dinov3 import *` at top of `src/models/__init__.py` triggers `transformers` import for every user, including ResNet users with no transformers installed → `ImportError` on `import src` | HIGH (if naive) | HIGH (breaks every existing user) | `src/models/dinov3.py` must NOT import `transformers` at module top level. The `from transformers import AutoModel` happens *inside* `DINOv3Classifier.__init__`, mirroring the `speciesnet_tf.py:22–25` pattern (verified). The model class itself is defined with only `torch` + `torch.nn`. The `__init__.py` line `from .dinov3 import *` therefore loads only the class definition. Test (Phase 0 invariant): `python -c "import src.models"` succeeds in an env with transformers UNINSTALLED. |
| 4.2 | Same risk for `src/algorithms/__init__.py` + `dinov3_linear.py` | HIGH (if naive) | HIGH | Same mitigation: `transformers`/`peft`/`huggingface_hub` imports live inside `DINOv3Linear._wrap_with_peft` and `DINOv3Classifier.__init__`. The module-top imports are only `torch`, `pytorch_lightning`, and `src.utils.mixup`. The mixup import is pure-python (no heavy deps). |
| 4.3 | Default `configs/config.yaml` accidentally triggers DINOv3 instantiation due to a typo or default key promotion | LOW | HIGH | Phase 4 does NOT modify `configs/config.yaml` (that is Phase 8, comment-only). Algorithm string remains `Plain` by default; `algorithms.__dict__['Plain']` resolves unchanged. Invariant test asserts `conf.algorithm == 'Plain'` and `conf.model_name == 'PlainResNetClassifier'` for the default config. |
| 4.4 | First-time `AutoModel.from_pretrained(...)` hits HF Hub, fails on airgapped box, training crashes mid-experiment | MEDIUM (for DINOv3 users) | MEDIUM | (a) Document `HF_HOME` / `TRANSFORMERS_OFFLINE=1` in `docs/dinov3.md`. (b) Raise a clear error with install/login instructions wrapping the `HfHubHTTPError` (mirror SINCHI's lines 51–56). (c) Tests do not invoke `from_pretrained`; they monkeypatch `AutoModel.from_pretrained` to return a tiny stub. |
| 4.5 | `DINOv3Classifier` instantiated with model id whose backbone hidden_size cannot be auto-detected → silent wrong shape | LOW | HIGH | Raise `ValueError` explicitly (already in skeleton at research report line 195–198). Unit test asserts that a fake config without `hidden_size` raises with an actionable message. |
| 4.6 | Lightning's `save_hyperparameters(conf.__dict__)` serializes the entire `transformers` `PretrainedConfig` object into the checkpoint, bloating it by MBs and creating a peft/transformers version-lock on load | MEDIUM | MEDIUM | Save only YAML-derivable hparams: `self.save_hyperparameters(ignore=['conf', 'train_class_counts', 'id_to_labels'])` and store the *string* model_name, not the AutoModel object. Mirror `Plain.__init__` line 42. Test: load checkpoint without `transformers` installed → unpickle succeeds (only standard types in hparams). |
| 4.7 | `val/test/predict` transform still uses 224 (existing `Classification_Inference_Transform` default) → DINOv3 model fed wrong input size, silent accuracy collapse | HIGH (if not handled) | MEDIUM | Algorithm class exposes `IMAGE_SIZE = 518`. Either (a) inject transform via dataset config (preferred — requires a 2-line addition to `Custom_Crop` to honor `image_size` kwarg, which is additive), or (b) override `transfer_batch_to_device` to resize. Decision deferred to design discussion w/ architect-1 but the risk is *flagged* and tested: integration test asserts the input tensor to `DINOv3Classifier.forward` is 518×518. |

### Files MODIFIED vs CREATED
- **CREATED**: `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py`.
- **MODIFIED**:
  - `src/models/__init__.py`: was `from .plain_resnet import *` + `from . import classification`. Becomes `from .plain_resnet import *` + `from .dinov3 import *` + `from . import classification`. **Semantic-identity proof**: existing call sites resolve `models.__dict__['PlainResNetClassifier']` (in `plain.py:45–46`). Adding `DINOv3Classifier` to `__dict__` does not shadow `PlainResNetClassifier`; symbol tables are disjoint. Regression test confirms `models.__dict__['PlainResNetClassifier']` is the same class object pre- and post-phase.
  - `src/algorithms/__init__.py`: was `from . import utils` + `from .plain import *`. Becomes `from . import utils` + `from .plain import *` + `from .dinov3_linear import *`. Same semantic-identity argument.

### Dependency footprint added
- Module-top imports added by phase: none beyond stdlib + torch + pytorch_lightning (already in `requirements.txt` via `lightning`).
- Lazy-imported (only on DINOv3 instantiation): `transformers`.

### Rollback
`git revert <PHASE_4_SHA>` → both new files deleted; the 2 `__init__.py` lines removed. Verify:
```
python -c "import src; import src.algorithms; import src.models"
python main.py --config configs/config.yaml --dev   # 1-epoch smoke
diff <(python -c "...hash of state_dict...") tests/fixtures/ckpt_baseline_hash.txt
```

### DAG
Depends on Phases 1 (mixup) and 3 (extras_require for docs/CI). Does NOT depend on Phase 2 (independent code).

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

### Goal
Extend `DINOv3Linear` to support `dinov3.freeze_backbone: False` and emit two
SGD/AdamW param groups (feature LR + classifier LR), mirroring `Plain`'s
existing two-group SGD pattern. No new files.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 5.1 | Unfreezing 86M ViT-B params at FP32 with `batch_size: 32` (existing default) → OOM on anything under 24GB | HIGH (for users who naively flip the flag) | HIGH | (a) Phase 8 docs declare a `batch_size: 4, precision: 16-mixed` default for full-FT. (b) Sample `config_dinov3_finetune.yaml` ships with safe defaults. (c) Algorithm logs `trainable_params` count at `on_train_start`; if > 10M params, warn user. |
| 5.2 | `DINOv3Classifier.freeze_backbone()` calls `self.feature.eval()`, which sets BatchNorm to eval; if the user re-enables training via `unfreeze_backbone()` mid-run, BN stats stay frozen. DINOv3 ViT uses LayerNorm so this is moot, but if someone swaps backbone... | LOW | LOW | Document that `freeze_backbone` is a startup-only flag; do not toggle mid-run. Algorithm enforces by raising if called after `setup` finished. |
| 5.3 | Two-param-group optimizer with `requires_grad=False` params silently includes them in AdamW state → wasted memory | LOW | LOW | `trainable = [p for p in self.parameters() if p.requires_grad]` filter already in skeleton. Unit test asserts `len(opt.param_groups[0]['params']) == expected_count` under both freeze settings. |
| 5.4 | Mixing AdamW (DINOv3 path) vs SGD (Plain path) in the same repo creates inconsistent docs / user confusion | LOW | LOW | Doc page (Phase 8) explicitly notes: "DINOv3 path uses AdamW + CosineAnnealingLR, deliberately matching SINCHI semantics; ResNet path retains SGD + StepLR." |

### Files MODIFIED vs CREATED
- **CREATED**: none.
- **MODIFIED**: `src/algorithms/dinov3_linear.py` (extends `configure_optimizers` and adds `on_train_start` logging). Semantic-identity for **existing call sites of this file**: this file was created in Phase 4; no other phase consumed it. Phase 4's `dinov3.freeze_backbone: True` config path is unchanged — the new branch only activates when `freeze_backbone: False`.

### Dependency footprint added
None.

### Rollback
`git revert <PHASE_5_SHA>` → linear-probe path remains functional, full fine-tune YAML configs would error with `KeyError` on the missing branch. Verify with the linear-probe integration test (Phase 4 test) still green.

### DAG
Depends on Phase 4.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

### Goal
Add the `_wrap_with_peft` branch to `DINOv3Linear` plus checkpoint roundtrip
tests. `peft` is lazy-imported only when `dinov3.peft.enable: True`.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 6.1 | `peft` API drift: `LoraConfig`/`get_peft_model`/`TaskType.FEATURE_EXTRACTION` field names changed between 0.10 and 0.13 | MEDIUM | MEDIUM | Pin `peft>=0.10.0,<0.14.0` in `requirements-dinov3.txt`. Wrap `_wrap_with_peft` in try/except that raises an actionable upgrade message. CI matrix tests against pinned floor + ceiling. |
| 6.2 | `init_lora_weights="pissa_niter_4"` requires `peft>=0.10`; older versions silently fall back to gaussian init (no error) | MEDIUM | HIGH (silent quality regression) | Test: after PiSSA wrap, assert `lora_A` weights differ from a fresh gaussian init by > epsilon. If assertion fails, raise version-mismatch error. |
| 6.3 | `get_peft_model` wraps the backbone with extra attribute layers (`base_model.model.*`), changing the state_dict key namespace → the inference loader (Phase 7) must handle this OR loading own-trained PEFT checkpoint fails round-trip | HIGH | HIGH | Phase 7's `_extract_state_dict` explicitly handles the `feature.base_model.model.*` prefix. **Integration test in this phase** roundtrips a fake PEFT checkpoint (save → load → forward → match logits) within Phase 6 itself, not deferred to Phase 7. |
| 6.4 | PEFT wrapping classifier head accidentally (target_modules glob too broad) → head no longer trainable | LOW | HIGH | Default `target_modules` list explicitly enumerates only backbone projections (`q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj`). Test: post-wrap, `self.net.classifier.weight.requires_grad == True`. |
| 6.5 | A user enabling PEFT without `peft` installed gets an ImportError mid-training (after data loaded, GPU allocated) — wasted minutes | MEDIUM | LOW | `_wrap_with_peft` is called from `__init__`, so the error fires *before* `Trainer.fit`. ImportError message includes the exact pip install command and points at `requirements-dinov3.txt`. |
| 6.6 | PEFT checkpoint saved without `peft_config` metadata → at inference time, `DINOv3CustomWeights` can't know to re-wrap before load | HIGH | HIGH | At save time (via Lightning `on_save_checkpoint` hook), embed `peft_config_dict` into the Lightning checkpoint under a new key `dinov3_peft_config`. At load time, Phase-7 loader auto-detects this key and re-wraps. Documented in §Acceptance. |

### Files MODIFIED vs CREATED
- **CREATED**: none.
- **MODIFIED**: `src/algorithms/dinov3_linear.py` (adds `_wrap_with_peft`, `on_save_checkpoint`). Semantic-identity for the linear-probe and full-FT paths: both still take the `if self.use_peft: ...` branch as False; behaviour bit-identical. Unit test: instantiate `DINOv3Linear` with `dinov3.peft.enable: False` → state of `self.net.feature` is `unittest.mock.ANY`-equal to a fresh `AutoModel`, no PEFT attributes attached.

### Dependency footprint added
- Module-top imports: none added.
- Lazy-imported (only on `peft.enable: True`): `peft`.
- `requirements-dinov3.txt` gains `peft>=0.10.0,<0.14.0` (Phase 3 already created this file).

### Rollback
`git revert <PHASE_6_SHA>` → PEFT YAML configs error with `KeyError`/`AttributeError`. Linear-probe and full-FT paths unchanged. Verify with Phase 4 + 5 tests green.

### DAG
Depends on Phase 5. (Strictly only depends on Phase 4 in code, but ordering-wise we sequence it after 5 so phase risks accumulate predictably.)

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

### Goal
Add `src/models/classification/dinov3_base/` subpackage with
`DINOv3BaseInference` and `DINOv3CustomWeights`. The auto-detect logic in
`_extract_state_dict` is the **single highest-risk function** in the entire
integration; this phase has the longest test list.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 7.1 | New `from .dinov3_base import *` line in `src/models/classification/__init__.py` triggers `transformers` import for users with only `transformers`-free env, breaking `from src.models import classification as pw_classification` for ResNet users | HIGH (if naive) | HIGH | Wrap in `try/except ImportError: pass`, **mirroring** the existing `speciesnet_base/__init__.py:1–4` pattern (verified in repo). Test (invariant): in an env without `transformers`, `from src.models import classification` succeeds, `classification.CustomWeights` still resolves, `classification.DINOv3CustomWeights` raises `AttributeError`. |
| 7.2 | Corrupt checkpoint file (truncated, wrong torch version) → `torch.load` raises mid-function, user gets unhelpful traceback | MEDIUM | LOW | Wrap `torch.load` call in try/except, re-raise as `RuntimeError` with the path and a "is this a torch checkpoint?" suggestion. Test: pass a 16-byte garbage file → assertion on exception message. |
| 7.3 | Old SINCHI checkpoint missing `model_state_dict` key (e.g. saved with an even older script) — falls through to `raise ValueError` (good) but error doesn't list the actual top-level keys | LOW | LOW | Error message includes `list(ckpt.keys())[:10]` so the user can pass `weights_key=` explicitly. |
| 7.4 | PEFT checkpoint where `peft_config` is missing from the checkpoint AND user did not pass `peft_config=` kwarg → state-dict load succeeds with `strict=False` but the backbone has the wrong shape (no LoRA adapters) → silently wrong outputs | **HIGH** | **HIGH** | Inspect the state-dict keys: if ANY key matches `^feature\.base_model\.model\.` AND no `peft_config` was provided AND no embedded `dinov3_peft_config` was found in the checkpoint dict → raise `ValueError("Checkpoint contains LoRA adapter keys but no peft_config was provided; pass peft_config={...} or use a non-PEFT checkpoint")`. Test: synthesize state-dict with LoRA-style keys, instantiate without peft_config → assert ValueError. |
| 7.5 | MD-Cls own DINOv3 checkpoint with `weights_only=True` future-default (torch 2.6+) → load fails because PEFT objects in checkpoint aren't on safe list | MEDIUM | MEDIUM | Pass `weights_only=False` explicitly (already in skeleton). Document the security trade-off in `docs/dinov3.md`: "only load checkpoints from sources you trust." |
| 7.6 | `strict=False` swallows real bugs (typo in a key remap) — silent silently-wrong inference | MEDIUM | HIGH | After `load_state_dict(state, strict=False)`, count `missing` and `unexpected` keys. If `len(missing) > 5` OR ANY missing key starts with `classifier.` → raise (not warn). Test: provide state-dict missing the `classifier.weight` key → assertion on raise. |
| 7.7 | `hidden_size` mismatch between the checkpoint's model and the user-supplied `model_name` (e.g. user trained on ViT-B but passes `model_name=ViT-L`) → silent classifier-weight-shape mismatch swallowed by strict=False | MEDIUM | HIGH | Before `load_state_dict`, compare `state['classifier.weight'].shape[1]` against `self.net.embedding_dim`; raise if mismatch with the actionable suggestion. Test included. |
| 7.8 | Wget download fallback (`url=`) hits a redirect / 404 / requires HF token → cryptic wget error | LOW | LOW | Recommend in docs to download via `huggingface_hub.snapshot_download` instead; keep wget fallback for parity with `timm_base`. |

### Files MODIFIED vs CREATED
- **CREATED**: `src/models/classification/dinov3_base/__init__.py`, `.../base_classifier.py`, `.../custom_weights.py`.
- **MODIFIED**: `src/models/classification/__init__.py` — adds one line wrapped in try/except. Semantic-identity proof: all existing wildcard re-exports (`from .resnet_base import *`, `from .timm_base import *`, `from .speciesnet_base import *`, `from .base_classifier import *`) untouched. New line is additive and **guarded** so an ImportError doesn't cascade.

### Dependency footprint added
- Module-top imports: none added to non-DINOv3 paths (try/except guard).
- Lazy-imported (only on DINOv3 inference instantiation): `transformers`, optionally `peft`.

### Rollback
`git revert <PHASE_7_SHA>` → subpackage gone, `DINOv3CustomWeights` no longer resolvable. ResNet inference (`CustomWeights`) unaffected. Verify:
```
python -c "from src.models.classification import CustomWeights; print('ok')"
pytest tests/integration/test_resnet_inference.py    # Phase 0 baseline
```

### DAG
Depends on Phase 4 (needs `DINOv3Classifier`) and Phase 6 (PEFT roundtrip tests).

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

### Goal
Append commented YAML block to `configs/config.yaml`, ship a fully-uncommented
`configs/config_dinov3_linear.yaml` sample, write `docs/dinov3.md`, add a
section to `docs/usage.md`, update `README.md` capability bullet. Update
`requirements.txt` ONLY with a comment pointing at `requirements-dinov3.txt`.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 8.1 | YAML append breaks `yaml.safe_load(open('configs/config.yaml'))` because of trailing whitespace / indentation drift | LOW | HIGH | Add a Phase 0 test that loads the default config and asserts identical dict (ignoring the new commented-out keys, which yaml.safe_load drops). |
| 8.2 | README update reorders bullet points (git diff noise) → stakeholders think more changed than did | LOW | LOW | Strictly *append* one bullet to the capability list, do not re-order. |
| 8.3 | `docs/usage.md` example invokes `DINOv3CustomWeights` with a HF model name that requires auth → doc reader fails | MEDIUM | LOW | Doc example uses a placeholder (`facebook/dinov3-vits16-pretrain-lvd1689m`) with explicit "you must `huggingface-cli login` first" callout BEFORE the code block. |
| 8.4 | New `configs/config_dinov3_linear.yaml` uses model defaults (ViT-B) that OOM on the user's 6GB GPU → bad first impression | HIGH (for small-GPU users) | MEDIUM | **Default sample config ships ViT-S (`facebook/dinov3-vits16-pretrain-lvd1689m`)** which fits in ~3GB at batch_size=4 + fp16. Doc table lists ViT-B/L/H memory requirements separately. |
| 8.5 | `requirements.txt` change accidentally inserts `transformers`/`peft` directly → breaks Invariant #2 | (would-be) HIGH | HIGH | Phase 8 does **NOT** modify `requirements.txt` except to add a comment: `# For DINOv3 fine-tuning, see requirements-dinov3.txt`. The actual deps go in `requirements-dinov3.txt` (Phase 3) and `pyproject.toml` extras. Regression test in Phase 0 hashes `requirements.txt` runtime-deps section. |

### Files MODIFIED vs CREATED
- **CREATED**: `configs/config_dinov3_linear.yaml`, `docs/dinov3.md`.
- **MODIFIED**:
  - `configs/config.yaml` — append commented block only. Semantic-identity proof: `yaml.safe_load` output equals pre-phase output (comments stripped).
  - `docs/usage.md` — append section. No existing sections modified.
  - `README.md` — append one capability bullet + one zoo example block. No existing prose modified.
  - `requirements.txt` — add one comment line `# DINOv3 extras: see requirements-dinov3.txt`. `pip install -r requirements.txt` resolves to the same package set as before.

### Dependency footprint added
None (Phase 3 already added the optional file).

### Rollback
`git revert <PHASE_8_SHA>` → docs revert. Code paths from Phases 4–7 still work but discoverability drops. Verify with default-config invariant test.

### DAG
Depends on Phases 4 and 7.

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

### Goal
Fix `src/models/classification/resnet_base/custom_weights.py` line 36 where
`num_layers=50` is hard-coded. Add `num_layers: int = 50` defaulted kwarg to
the constructor and forward it. The default `50` preserves all current call
sites. Add regression test asserting both ResNet-18 and ResNet-50 user-trained
checkpoints load successfully.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 2.1 | A downstream user passes `num_layers` as a positional arg already (impossible per current signature `(weights, class_names, device)`, but defensive) | NEAR-ZERO | LOW | New kwarg appended at end of signature; positional callers unaffected. Grep across repo for `CustomWeights(` to confirm zero callers pass beyond 3 positional args. |
| 2.2 | A user has a working "wrong" workflow where they pass `num_layers=18` to a ResNet-50 model expecting silent failure (very unlikely) | NEAR-ZERO | LOW | N/A. Document the fix in CHANGELOG. |
| 2.3 | A ResNet-50 checkpoint that was previously loaded successfully via the hard-coded path now fails because of a typo introduced in the patch | LOW | HIGH | Regression test exercises BOTH a ResNet-50 checkpoint (existing default) AND a ResNet-18 checkpoint. Both must load with `strict=True` (matching existing `base_classifier.py:120` behaviour). |
| 2.4 | Bug fix landed inside a different phase's commit → bisect can't isolate | (would-be) LOW | MEDIUM | Phase 2 is its **own commit** with `git revert`-clean SHA. |

### Files MODIFIED vs CREATED
- **CREATED**: `tests/regression/test_customweights_num_layers.py`, `tests/fixtures/resnet18_tiny.ckpt` (synthesized in conftest, not committed binary — generated on demand from `PlainResNetClassifier(num_layers=18)` with `num_cls=2`).
- **MODIFIED**: `src/models/classification/resnet_base/custom_weights.py`. Diff:
  ```
  -    def __init__(self, weights=None, class_names=None, device="cpu"):
  +    def __init__(self, weights=None, class_names=None, device="cpu", num_layers: int = 50):
  ...
  -                                                   num_cls=self.num_cls, num_layers=50, url=None)
  +                                                   num_cls=self.num_cls, num_layers=num_layers, url=None)
  ```
  **Semantic-identity proof for existing callers**: `PlainResNetInference.__init__` (line 110) accepts `num_layers=50` as a defaulted kwarg. The previous `CustomWeights` always passed `num_layers=50`; the new `CustomWeights` defaults to `num_layers=50` and forwards it. All existing call sites (`CustomWeights(weights=..., class_names=..., device=...)`) get `num_layers=50` exactly as before. Verified by grep: no in-repo caller passes a 4th positional arg.

### Dependency footprint added
None.

### Rollback
`git revert <PHASE_2_SHA>` → hard-code restored; regression test fails (correctly indicates the bug is back). Verify:
```
pytest tests/regression/test_customweights_num_layers.py::test_resnet50_still_loads  # green
pytest tests/regression/test_customweights_num_layers.py::test_resnet18_loads        # red (expected after revert)
```

### DAG
Depends only on Phase 0.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

### Repo invariants (12) — promised NOT violated

These become Phase 0 automated tests. **Each invariant fails → block PR.**

| # | Invariant | Test name |
|---|---|---|
| I1 | `python -c "import src"` succeeds in an env with ONLY `requirements.txt` installed (no `transformers`, no `peft`, no `huggingface_hub`) | `test_default_env_imports_clean` |
| I2 | `pip install -r requirements.txt` resolves to the same package set as on the pre-change commit (no new runtime deps) | `test_requirements_diff_runtime_clean` (hash check) |
| I3 | `python main.py --config configs/config.yaml --dev` runs to completion on the demo dataset and produces a checkpoint whose `state_dict` SHA-256 equals the pre-change baseline (given fixed seed) | `test_resnet_baseline_checkpoint_hash` |
| I4 | `yaml.safe_load(open('configs/config.yaml'))` returns a dict equal to the pre-change dict | `test_default_config_unchanged` |
| I5 | `from src.models import classification; classification.CustomWeights(weights=..., class_names=[...])` succeeds against a ResNet-50 checkpoint (no DINOv3 deps installed) | `test_customweights_no_dinov3_deps` |
| I6 | `algorithms.__dict__['Plain']` resolves to the same class object pre- and post-change | `test_plain_algorithm_class_identity` |
| I7 | `models.__dict__['PlainResNetClassifier']` resolves to the same class object | `test_plain_resnet_class_identity` |
| I8 | `src/algorithms/plain.py` byte-equal to pre-change (file hash) | `test_plain_py_unchanged` |
| I9 | `src/models/plain_resnet.py` byte-equal to pre-change | `test_plain_resnet_py_unchanged` |
| I10 | `src/datasets/custom.py` byte-equal to pre-change | `test_custom_dataset_unchanged` |
| I11 | `main.py` byte-equal to pre-change | `test_main_unchanged` |
| I12 | `CustomWeights` 3-positional-arg call sites still work: `CustomWeights(weights=w, class_names=c, device=d)` produces a working model | `test_customweights_legacy_call_signature` |

I8–I11 are **byte-equality** checks (SHA-256) — the strictest possible
form of "no edits to existing code." If a phase legitimately needs to edit
one of these (Phase 2 edits `custom_weights.py`, Phase 8 edits
`configs/config.yaml`, README, usage.md), the corresponding invariant test
is **updated in that phase's commit** to the new SHA, so the byte-equality
guarantee is preserved on a per-phase basis. The invariants on `main.py`,
`plain.py`, `plain_resnet.py`, `custom.py` are NEVER updated.

### Cross-cutting regression policy

- Every phase MUST run the full `tests/regression/` suite before merging.
- Phase 0 sets up `tests/conftest.py` that synthesizes the baseline ResNet
  checkpoint hash at first run and pins it in `tests/fixtures/`.
- CI runs the regression suite on two environments: (a) `requirements.txt`
  only, (b) `requirements.txt + requirements-dinov3.txt`. (a) is the
  load-bearing one for backward-compat; (b) ensures DINOv3 tests don't
  break ResNet behaviour.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Pyramid

| Layer | Count target | Per-phase distribution |
|---|---|---|
| **Unit** (pure-Python, no GPU, no network, < 1s each) | ~15 | Mixup 4, model 3, algorithm 4, extract_state_dict 4 |
| **Integration** (CPU OK, tiny stub backbone, synthesized data, ~10s each) | ~10 | Linear probe 2, full FT 2, LoRA 1, PiSSA 1, inference roundtrip ×3 formats 3, transform 1 |
| **Regression** (invariants — byte/hash/import) | 12 | All in Phase 0; some updated by Phases 2, 8 |
| **E2E** (real DINOv3 download, GPU, multi-epoch) | 3 — manual only | Linear probe, full FT, PEFT — run by humans, **never in CI** |

### Stub strategy

Tests MUST NOT download a real DINOv3 model. Strategy:

```python
# tests/conftest.py
@pytest.fixture
def tiny_dinov3_stub(monkeypatch):
    """Replace AutoModel.from_pretrained with a 2-layer toy ViT
    whose hidden_size=64, so all tests run in <1s on CPU."""
    class _TinyConfig:
        hidden_size = 64
    class _TinyModel(nn.Module):
        config = _TinyConfig()
        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            class _Out: pass
            o = _Out()
            o.last_hidden_state = torch.randn(B, 1, 64)
            return o
    from transformers import AutoModel
    monkeypatch.setattr(AutoModel, "from_pretrained",
                        staticmethod(lambda *a, **kw: _TinyModel()))
```

### Fixtures needed

- `tests/fixtures/resnet50_baseline_state_dict.sha256` (Phase 0).
- `tests/fixtures/resnet18_tiny.ckpt` (Phase 2, synthesized in conftest).
- `tests/fixtures/sinchi_raw_synthetic.pt` (Phase 7, synthesized: contains `model_state_dict` with `backbone.*` keys).
- `tests/fixtures/sinchi_peft_synthetic.pt` (Phase 7, synthesized: contains `backbone.base_model.model.*` + LoRA adapter keys).

### CI vs local

- **CI (every PR)**: unit + integration + regression suites, on env (a) and (b).
- **Local-only (developer responsibility, gated by `--run-e2e` marker)**: E2E suite.
- **Pre-merge manual gate**: regression-suite green on `pip install -r requirements.txt` only (env (a)) is a HARD gate.

### Test-running command per phase

| Phase | Command before merge |
|---|---|
| 0 | `pytest tests/regression -x` |
| 1 | `pytest tests/unit/test_mixup.py tests/regression -x` |
| 2 | `pytest tests/regression/test_customweights_num_layers.py tests/regression -x` |
| 3 | `pytest tests/regression -x` (no new test code, just deps) |
| 4 | `pytest tests/unit/test_dinov3_model.py tests/integration/test_dinov3_linear_probe.py tests/regression -x` |
| 5 | `pytest tests/unit/test_dinov3_finetune_param_groups.py tests/regression -x` |
| 6 | `pytest tests/integration/test_dinov3_peft_lora.py tests/integration/test_dinov3_peft_pissa.py tests/regression -x` |
| 7 | `pytest tests/unit/test_extract_state_dict.py tests/integration/test_dinov3_inference_three_formats.py tests/regression -x` |
| 8 | `pytest tests/regression/test_default_config_unchanged.py tests/regression -x` |

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

Mapped to the research report's 10-item validation checklist (§8, lines 769–781).

| # | Research checklist item | Phase that lands the test | Concrete command | Expected output |
|---|---|---|---|---|
| 1 | Existing ResNet path unchanged | 0 | `python main.py --config configs/config.yaml --dev` + hash check | `sha256(state_dict) == baseline_hash.txt` |
| 2 | New linear-probe path | 4 | `python main.py --config configs/config_dinov3_linear.yaml --dev` | checkpoint at `weights/Crop/DINOv3Linear/*.ckpt`; backbone grad norms == 0 |
| 3 | New full fine-tune path | 5 | same with `freeze_backbone: False` | backbone grad norms > 0; trainable_params ≈ 86M |
| 4 | Mixup toggle | 4/6 | toggle `use_noisy_same_class_mixup: True`, compare loss curve to baseline | curves differ; no shape errors |
| 5 | LoRA path | 6 | `peft.enable: True, method: lora` | trainable_params ≈ 0.5M; PiSSA path: weights differ from gaussian init |
| 6 | Inference loading — MD-Cls own | 7 | `DINOv3CustomWeights(weights=<ckpt from #2>, class_names=[...])` | zero missing/unexpected keys |
| 7 | Inference loading — SINCHI raw | 7 | same with synthetic SINCHI-style `.pt` | loads with `backbone.→feature.` remap; zero missing |
| 8 | Inference loading — SINCHI PEFT | 7 | same with `peft_config={...}` and synthetic PEFT `.pt` | loads with `strict=False`; LoRA adapter keys absorbed |
| 9 | `CustomWeights` regression (num_layers) | 2 | `CustomWeights(weights=..., class_names=[...], num_layers=18)` + default-50 path | both work |
| 10 | Lazy-import | 0 / 7 | env without `transformers`: `python -c "from src.models import classification"` then `classification.CustomWeights(...)` | succeeds. `classification.DINOv3CustomWeights(...)` raises ImportError with install instructions |

### Definition of "done" (acceptance gate)

All 12 invariants pass AND all 10 research-checklist tests pass AND CI green
on both env (a) and env (b) AND `docs/dinov3.md` reviewed by a domain expert
for HF auth / GPU memory / PEFT default sanity.

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

### Top-level risk register (project-wide, not per-phase)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A phase lands that breaks Invariant I1 (default env import) — every existing ResNet user's `import src` fails on `pip install --upgrade` | MEDIUM (would happen w/o discipline) | CATASTROPHIC | Phase 0 + per-phase regression gate; both `__init__.py` re-exports are guarded; lazy imports verified in CI env (a). |
| R2 | DINOv3 HF model is gated and requires `huggingface-cli login`; user without token gets cryptic error mid-training | HIGH (for first-time users) | MEDIUM | Phase 4 wraps `AutoModel.from_pretrained` in a try/except producing an actionable error. Phase 8 docs include explicit `huggingface-cli login` step BEFORE any DINOv3 command. CI never authenticates against HF. |
| R3 | GPU OOM on user's box for the documented "default" full-FT config | HIGH | MEDIUM | Sample config defaults to ViT-S + `batch_size: 4` + `precision: 16-mixed`. Doc table maps model size → minimum GPU memory. |
| R4 | `peft` API drift breaks Phase 6 between releases | MEDIUM | MEDIUM | Pin in `requirements-dinov3.txt`. CI matrix tests floor + ceiling. |
| R5 | Auto-detect (`_extract_state_dict`) misclassifies a checkpoint and silently loads garbage weights | MEDIUM | HIGH | Phase 7 risks 7.4, 7.6, 7.7 cover this; raise rather than warn on key mismatches; refuse to silently `strict=False` past a missing `classifier.*` key. |
| R6 | A phase cannot be reverted standalone (e.g., Phase 5 depends transitively on Phase 4 schema such that reverting 5 with 6 still present breaks 6) | LOW (with DAG) | MEDIUM | DAG enforced: Phase 6 reverted before Phase 5; Phase 7 reverted before Phase 4. Documented rollback order: 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0. |
| R7 | Lightning checkpoint format change between minor Lightning versions makes Phase 4 checkpoints unloadable later | LOW | LOW | Pin Lightning floor in `requirements.txt` (not a Phase 4 change — flag for follow-up). |
| R8 | A user's existing checkpoint trained with the pre-bug-fix `CustomWeights(num_layers=50)` workflow won't be reproducible because they had a workaround that's no longer needed | NEAR-ZERO | LOW | Default behaviour unchanged (kwarg defaults to 50). Workaround still works. |

### Per-phase rollback DAG

Reverse order, verified rollback per phase:

```
PHASE_8_SHA  →  revert  →  pytest tests/regression  →  green
PHASE_7_SHA  →  revert  →  pytest tests/regression tests/unit/test_dinov3_model.py tests/integration/test_dinov3_linear_probe.py  →  green
PHASE_6_SHA  →  revert  →  pytest tests/regression tests/integration/test_dinov3_linear_probe.py  →  green
PHASE_5_SHA  →  revert  →  pytest tests/regression tests/integration/test_dinov3_linear_probe.py  →  green
PHASE_4_SHA  →  revert  →  pytest tests/regression tests/unit/test_mixup.py  →  green
PHASE_3_SHA  →  revert  →  pytest tests/regression  →  green
PHASE_2_SHA  →  revert  →  pytest tests/regression tests/regression/test_customweights_num_layers.py::test_resnet50_still_loads  →  green (ResNet-18 test now fails, as expected — confirms revert)
PHASE_1_SHA  →  revert  →  pytest tests/regression  →  green
PHASE_0_SHA  →  revert  →  baseline removed; manual confirmation only
```

### Single-command "emergency revert" recipe

If a regression is detected post-merge of all 8 phases:

```bash
git revert --no-edit PHASE_8_SHA PHASE_7_SHA PHASE_6_SHA PHASE_5_SHA \
                     PHASE_4_SHA PHASE_3_SHA PHASE_2_SHA PHASE_1_SHA
# Phase 0 (tests) stays — they will all pass since the new code is gone
pytest tests/regression
```

Repo lands back at functionally pre-change state with the regression test
suite still in place (a strict win over before).

---

## Rationale (risk/backwards-compat lens)

The plan is structured around a single mental model: **"if a colleague pulls
this branch in the middle of their dissertation experiment, will their next
`python main.py` produce a bit-identical checkpoint?"** Every design choice
flows from that question:

- **8 phases instead of 10** (collapsing research-report Steps 1, 6, 9, 10
  into Phases 3 and 8) because phase boundaries should align with revert
  units, not narrative units.
- **Test scaffolding (Phase 0) precedes everything** because we cannot
  assert "backward-compatible" without a baseline.
- **Bug fix (Phase 2) is its own phase** because it is the single
  behavior-changing edit to shipped code, and isolating it preserves
  bisect-ability.
- **Optional deps (Phase 3) before the code that uses them (Phase 4+)**
  because docs/CI need the install target before the importing code lands.
- **Lazy imports + try/except wildcard re-export guard** because the
  `speciesnet_base` pattern in this very repo proves it works (`__init__.py`
  lines 1–4); we adopt it verbatim for `dinov3_base`.
- **Auto-detect (Phase 7) gets the longest risk register** because it is
  the function with the most degrees of freedom that the user has the
  least visibility into; we trade `strict=False` permissiveness for
  loud-failure assertions on every category of likely misuse.
- **Regression invariants are byte-equality where possible** because diff
  noise hides real risk; if Phase 5 accidentally re-touches `plain.py`
  while extending `dinov3_linear.py`, invariant I8 catches it immediately.

The plan deliberately does NOT optimize for fewest commits or shortest
calendar time. It optimizes for **the ability to ship 6 of the 8 phases and
hold the rest** if PEFT or HF auth turn out to be too risky for the user
base. Phases 0–5 + 7-without-PEFT + 8 deliver linear-probe + full-FT
DINOv3 with no `peft` dependency at all — a clean partial-rollout path.

---

## New ledger items proposed

(None. All 12 scope items addressed above.)

---

## Discussion summary
(round 1: skip)

## Consensus points
(round 1: skip)

## Disagreements
(round 1: skip)

## Evolution
(round 1: skip)

## NEW This Round
(round 1: skip)

STATUS: NEW=0 DISAGREEMENTS=0
