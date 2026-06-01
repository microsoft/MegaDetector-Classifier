# DINOv3 integration — phased plan (architect-3: risk / backwards-compat / additive-purity, round 3 — convergence)

> Perspective unchanged. Round 3 is convergence: every binding directive
> from the round-2 inquisitor's final-synthesis section is folded in;
> the two NEW critiques against this proposal (A3-N1, A3-N2) are fixed
> at the contract level; stale disagreements are removed per the
> inquisitor's instruction (8 MISTAKEN-BY=architect-3 across rounds 1–2;
> one RESOLVED-TOWARD=architect-3 on the transform kwarg, which we keep).

---

## Round-3 changes at a glance (so reviewers can diff fast)

1. **A3-N1 (absent-DINOv3 class contract):** unified to a single surface
   — `ImportError` raised from a package-level `__getattr__` with the
   exact message substring `"pip install -r requirements-dinov3.txt"`.
   All four references in the document (Risk 5.1, Invariant I13,
   Acceptance row 10, test `test_dinov3_absent_actionable_error`) now
   match. The acceptance command catches `ImportError` explicitly and
   asserts the message substring.
2. **A3-N2 (PEFT-disabled regression used a state-dict hash):** replaced
   with a behavioral snapshot — fixed RNG seed → one forward pass on a
   fixed input tensor → `torch.allclose(logits, golden, atol=1e-5)` AND
   one SGD step → `math.isclose(loss, golden_loss, abs_tol=1e-5)`.
   Golden-tensor versioning policy documented (frozen until model
   architecture changes; regenerated via committed script).
3. **Transform mechanism** — the lead refers to `Custom_Crop`; verified
   that `Custom_Crop` is the `LightningDataModule` in
   `src/datasets/custom.py` (line 240). The kwarg is added there as
   `image_size: int = 224` and threaded into instance-level transform
   construction. Module-level `data_transforms = {...}` (lines 33–47) is
   preserved only as a *fallback constant* and is no longer referenced
   when the new kwarg is present. Default path is behaviour-identical.
4. **Disagreements section is empty** (inquisitor directive: report
   `DISAGREEMENTS=0`).
5. **Single-dependency policy** restated as binding:
   `requirements-dinov3.txt` contains exactly three packages —
   `transformers`, `peft`, `huggingface_hub` — all three lazy at every
   import site, all three covered by absent-env tests.

---

## Executive overview

The 12 ledger scope items are realized as **9 phases** (numbered 0–9
excluding a renumbering; see Phase index). Phase order:

scaffolding → bug-fix → mixup → optional-deps → model+linear-probe →
loader-v1 (MD-Cls only) → full-FT → PEFT → loader-v2 (SINCHI raw + PEFT)
→ config+docs.

Load-bearing invariant: *unchanged default `configs/config.yaml` +
unchanged `pip install -r requirements.txt` + same seed/data ⇒
bit-identical training, checkpoints, and predictions to the pre-change
repo for the ResNet pipeline.*

---

## Phase index (table)

| # | Phase | Touches existing code? | New runtime deps | Rollback risk | DAG predecessor |
|---|---|---|---|---|---|
| 0 | Test scaffolding + regression baseline | NO (tests/ only; `tests/_artifacts/` is the only writable test root) | none | trivial | — |
| 1 | CustomWeights `num_layers` bug fix | YES (1 file, +1 defaulted kwarg) | none | low | 0 |
| 2 | Mixup utility port | NO | none | trivial | 0 |
| 3 | Optional-deps wiring (`requirements-dinov3.txt`) | NO | none in default; `transformers`+`peft`+`huggingface_hub` via dinov3 file | trivial | 0 |
| 4 | Model + linear-probe algorithm + `image_size=224` kwarg on `Custom_Crop` | NO (3 new files + 2 `__init__.py` re-export lines + 1 defaulted kwarg on `Custom_Crop.__init__`) | none in default; `transformers` lazy | low | 2, 3 |
| 5 | Loader-v1: `DINOv3CustomWeights` for MD-Cls `.ckpt` only | NO (new subpackage; package-level `__getattr__` for actionable `ImportError`) | none in default | low | 4 |
| 6 | Full fine-tune toggle | NO (extends Phase 4 algorithm file) | none additional | low | 4 |
| 7 | PEFT (LoRA / PiSSA) toggle | NO (extends Phase 4 algorithm file) | `peft` (lazy) via dinov3 reqs | medium | 6 |
| 8 | Loader-v2: extend `_extract_state_dict` for SINCHI raw + SINCHI PEFT | NO (extends Phase 5 file) | none additional | medium-high (real-SINCHI evidence gate) | 5, 7 |
| 9 | Config + docs | YES — commented append to `configs/config.yaml`; new docs files | none | trivial | 4, 5, 8 |

Total: 10 commits (0–9). Phases 4–8 are pure adds. Phase 1 and the
commented append in Phase 9 are the only edits to already-shipped files;
both are backwards-compatible by construction.

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

### Ordering rationale (risk lens)

1. **Phase 0** — baseline `python -c "import src"`, behavior-snapshot
   ResNet test, and `tests/regression/dryrun_revert.sh` land first.
2. **Phase 1 (bug fix)** — adopted from architect-1; the only edit to
   behaviour-relevant existing code, landed in isolation for bisect.
   Phase 0 ships `tests/regression/test_resnet18_loads.py` marked
   `@pytest.mark.xfail(strict=True)`; Phase 1 flips the marker.
3. **Phase 2 (mixup)** — pure leaf utility, no callers.
4. **Phase 3 (optional-deps file)** — install target must exist before
   docs/CI reference it.
5. **Phase 4 (model + linear-probe + `Custom_Crop.image_size` kwarg)** —
   one commit because the 518-pixel transform only makes sense once a
   DINOv3 consumer exists.
6. **Phase 5 (loader-v1 MD-Cls only)** — MVP capability ships here;
   `peft` and SINCHI binaries not yet required.
7. **Phase 6 → 7** — full-FT before PEFT in increasing-risk order.
8. **Phase 8 (loader-v2 SINCHI)** — gated on a real SINCHI checkpoint
   key-inspection evidence file at
   `docs/design/dinov3-integration/sinchi_key_layouts.md` (tracked docs
   path per inquisitor directive). If unobtainable, Phase 8 is deferred
   and the v1 loader is the shipped surface.
9. **Phase 9 (config + docs)** — last, so docs match shipped behaviour.

### Entry/exit criteria (per-phase, unified template)

- **Entry**: (a) predecessor phases merged; (b) `pytest tests/regression
  -x` green on `main`; (c) phase risk register re-read; (d) rollback SHA
  recorded.
- **Exit**: (a) phase tests green; (b) regression suite green; (c)
  rollback dry-run executed via `bash tests/regression/dryrun_revert.sh
  <PHASE_SHA>` (script uses `git worktree add` to a repo-local
  `.revert-dryrun-$$` sibling and removes it on success/failure via
  `trap`); (d) phase entry appended to `COVERAGE_LOG.jsonl`.

### Per-phase deliverables summary

| Phase | New files | Modified files | Test files added |
|---|---|---|---|
| 0 | `tests/conftest.py`, `tests/regression/test_invariants.py`, `tests/regression/dryrun_revert.sh`, `tests/fixtures/baseline_predictions.json`, `tests/fixtures/peft_disabled_golden.pt` (golden logits+loss tensor) | none | scaffold |
| 1 | none | `src/models/classification/resnet_base/custom_weights.py` (+1 defaulted kwarg) | `tests/regression/test_customweights_num_layers.py` (xfail flip) |
| 2 | `src/utils/mixup.py` | none | `tests/unit/test_mixup.py` |
| 3 | `requirements-dinov3.txt` | `requirements.txt` (+1 comment line only) | `tests/regression/test_no_dinov3_imports_in_default_env.py` |
| 4 | `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py` | `src/models/__init__.py` (+1 line), `src/algorithms/__init__.py` (+1 line), `src/datasets/custom.py` (+1 defaulted kwarg on `Custom_Crop.__init__`) | `tests/unit/test_dinov3_model.py`, `tests/integration/test_dinov3_linear_probe.py`, `tests/regression/test_custom_crop_default_is_224.py`, `tests/integration/test_dinov3_dataloader_emits_518.py` |
| 5 | `src/models/classification/dinov3_base/{__init__.py, base_classifier.py, custom_weights.py}` | `src/models/classification/__init__.py` (+1 narrowly-guarded import + package-level `__getattr__`) | `tests/integration/test_dinov3_custom_weights_mdckpt.py`, `tests/unit/test_extract_state_dict_v1.py`, `tests/regression/test_dinov3_absent_actionable_error.py` (the I13 test) |
| 6 | none | `src/algorithms/dinov3_linear.py` (extends `configure_optimizers`) | `tests/unit/test_dinov3_finetune_param_groups.py`, `tests/integration/test_dinov3_finetune_grad_flow.py` |
| 7 | `tests/regression/regenerate_peft_disabled_golden.py` (committed regeneration script) | `src/algorithms/dinov3_linear.py` (adds `_wrap_with_peft`, `on_save_checkpoint`) | `tests/integration/test_dinov3_peft_lora.py`, `tests/integration/test_dinov3_peft_pissa.py`, `tests/unit/test_peft_target_modules_validated.py`, `tests/regression/test_peft_disabled_behavior_snapshot.py` |
| 8 | `docs/design/dinov3-integration/sinchi_key_layouts.md` (evidence file under tracked docs path, per A2-N1 directive) | `src/models/classification/dinov3_base/base_classifier.py` (extends `_extract_state_dict`) | `tests/integration/test_dinov3_custom_weights_sinchi_raw.py`, `tests/integration/test_dinov3_custom_weights_sinchi_peft.py` |
| 9 | `configs/config_dinov3_linear.yaml`, `configs/config_dinov3_finetune.yaml`, `configs/config_dinov3_lora.yaml`, `docs/dinov3.md` | `configs/config.yaml` (commented append only), `README.md` (capability bullet), `docs/usage.md` (section), `CHANGELOG.md` | `tests/regression/test_default_config_unchanged.py`, `tests/docs/test_sample_configs_parse.py` |

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

### Goal
Port `noisy_same_class_mixup` and `efficient_batch_same_class_mixup`
verbatim from `Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py` to
`src/utils/mixup.py`. Pure-Python (torch + numpy, already required).
**Do not** edit `src/utils/__init__.py`.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 2.1 | Re-export pollution via `src/utils/__init__.py` | LOW | LOW | Do not edit. Callers `import src.utils.mixup` explicitly. |
| 2.2 | Numerical drift across numpy versions | LOW | LOW | Mixup is opt-in. Tests assert deterministic-given-seed, not cross-env bit-equal. |
| 2.3 | Test fixtures in `/tmp` (forbidden in sandboxed env) | LOW | LOW | All fixtures under `tests/_artifacts/` (Phase 0 policy). |

### Files MODIFIED vs CREATED
- **CREATED**: `src/utils/mixup.py`, `tests/unit/test_mixup.py`.
- **MODIFIED**: none.

### Dependency footprint added
None.

### Rollback
`git revert <PHASE_2_SHA>`. `python -c "import src.algorithms, src.models, src.datasets, src.utils"` succeeds.

### DAG
Depends only on Phase 0.

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

### Goal
Land `src/models/dinov3.py` (`DINOv3Classifier`) and
`src/algorithms/dinov3_linear.py` (`DINOv3Linear`, frozen-backbone path
only). Wire into the `__dict__` registries without forcing `transformers`
on ResNet users. **Also**: add a defaulted `image_size: int = 224` kwarg
to `Custom_Crop.__init__` in `src/datasets/custom.py` (lead-confirmed
target; existing `Custom_Crop` class begins at line 240 of that file).
The kwarg is forwarded to the instance-built `transforms.Resize(...)`
and `transforms.RandomResizedCrop(...)` ops; default 224 reproduces the
module-level `data_transforms` dict (lines 33–47) op-for-op for the
ResNet path.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 4.1 | Top-level `transformers` import in `dinov3.py` breaks ResNet users | HIGH (if naive) | HIGH | `transformers` import lives INSIDE `DINOv3Classifier.__init__`. Module-top imports: `torch`, `torch.nn` only. Pattern mirrors `speciesnet_base/__init__.py:1–4`. Invariant I1 enforces. |
| 4.2 | Same for `dinov3_linear.py` | HIGH (if naive) | HIGH | Module-top: `torch`, `pytorch_lightning`, `src.utils.mixup`, `src.models` only. `transformers`/`peft`/`huggingface_hub` live in method bodies. |
| 4.3 | Default `configs/config.yaml` accidentally instantiates DINOv3 | LOW | HIGH | Phase 4 does NOT touch `configs/config.yaml`. Invariant I4. |
| 4.4 | HF auth failure on `AutoModel.from_pretrained` (gated model) | MEDIUM | MEDIUM | Wrap with try/except over `HfHubHTTPError`, `OSError`; re-raise `RuntimeError("Failed to download {model_name}. If gated: run 'huggingface-cli login'. If offline: set HF_HOME and TRANSFORMERS_OFFLINE=1. Original: {e}")`. Tests use `mock_hf_automodel`. |
| 4.5 | `hidden_size` missing on stub configs | LOW | HIGH | Raise `ValueError("Could not infer embedding_dim from model {name}; pass embedding_dim=... explicitly.")`. |
| 4.6 | `save_hyperparameters` serialises a HF `PretrainedConfig` | MEDIUM | MEDIUM | `self.save_hyperparameters(ignore=['conf', 'train_class_counts', 'id_to_labels'])`. Only string `model_name` saved. Test: `torch.load(ckpt)` succeeds with `transformers` uninstalled. |
| 4.7 | 518 transform breaking ResNet's 224 path | RESOLVED | — | Defaulted `Custom_Crop.__init__(..., image_size: int = 224)`. Instance-built transforms use that size. Test pair: `test_custom_crop_default_is_224` (no kwarg → val Resize is `(224,224)` AND train RandomResizedCrop size is `(224,224)`) AND `test_dinov3_dataloader_emits_518` (with kwarg=518, batch shape `(B,3,518,518)`). |
| 4.8 | DINOv3 license / gated weights | LEGAL | LEGAL | Phase 9 `docs/dinov3.md` documents license + citation. Phase 4 introduces no automatic download. |

### Files MODIFIED vs CREATED
- **CREATED**: `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py`.
- **MODIFIED**:
  - `src/models/__init__.py`: append `from .dinov3 import *`. Identity proof: `models.__dict__['PlainResNetClassifier']` resolves to same class object pre/post (regression I7).
  - `src/algorithms/__init__.py`: append `from .dinov3_linear import *`. Same (I6).
  - `src/datasets/custom.py`: `Custom_Crop.__init__` gains `image_size: int = 224` kwarg; transforms are now built at instance time using that size. Default behaviour is op-for-op identical to the prior module-level `data_transforms` dict.

### Dependency footprint added
- Module-top: none.
- Lazy: `transformers` (in `DINOv3Classifier.__init__`).

### Rollback
```
git revert <PHASE_4_SHA>
python -c "import src; import src.algorithms; import src.models"
pytest tests/regression -x        # I1..I13 green
```

### DAG
Depends on Phases 2 (mixup) and 3 (optional-deps file).

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

### Goal
Extend `DINOv3Linear` to support `dinov3.freeze_backbone: False` with
two AdamW param groups (feature LR + classifier LR). No new files.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 6.1 | OOM with default `batch_size: 32` on <24GB GPU | HIGH | HIGH | Sample `configs/config_dinov3_finetune.yaml` ships `batch_size: 4`, `precision: 16-mixed`. Doc table maps model → min measured GPU memory with rows labelled MEASURED or ESTIMATED. |
| 6.2 | Mid-run freeze toggle inconsistency | LOW | LOW | Startup-only flag; raise if changed after `setup` finishes. |
| 6.3 | AdamW state allocated for `requires_grad=False` params | LOW | LOW | `trainable = [p for p in self.parameters() if p.requires_grad]` filter. Unit test asserts param-group counts under both flags. |
| 6.4 | Non-CUDA execution | MEDIUM | LOW | Sample comment: "`precision: 16-mixed` is CUDA-only; CPU/MPS users set `precision: 32`." |
| 6.5 | DDP scope | MEDIUM | LOW | Out of scope for v1; explicit non-goal in `docs/dinov3.md`. Trainer defaults to `strategy="auto"`. |

### Files MODIFIED vs CREATED
- **CREATED**: none.
- **MODIFIED**: `src/algorithms/dinov3_linear.py` (extends `configure_optimizers`; adds `on_train_start` `trainable_params` log line).
- **NOTE on revertability**: Phases 4, 6, 7 modify the same file. Plan does not claim independent revertability; it claims **reverse-DAG revertability** (9 → 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0). `dryrun_revert.sh` refuses out-of-order revert.

### Dependency footprint added
None.

### Rollback
`git revert <PHASE_7_SHA> <PHASE_6_SHA>` (mandatory order). Linear-probe path (Phase 4) remains functional.

### DAG
Depends on Phase 4.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

### Goal
Add `_wrap_with_peft` and `on_save_checkpoint` to `DINOv3Linear`. `peft`
lazy-imported only when `dinov3.peft.enable: True`. Use
`TaskType.FEATURE_EXTRACTION` per inquisitor industry-comparison citation.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 7.1 | `peft` API drift | MEDIUM | MEDIUM | Pin `peft>=0.10.0,<0.14.0` in `requirements-dinov3.txt`. CI matrix tests floor + ceiling at Phase 7 (matrix activates at Phase 3). |
| 7.2 | `TaskType.FEATURE_EXTRACTION` correctness | LOW | MEDIUM | HF PEFT docs define `FEATURE_EXTRACTION` as "provides hidden states used as embeddings or features for downstream tasks" — matches backbone → CLS → external classifier. Unit test asserts `self.net.classifier` remains a plain `nn.Linear` (not absorbed by PEFT). |
| 7.3 | `target_modules` mismatch with backbone module names | HIGH | HIGH | **Runtime validation** in `_wrap_with_peft`: iterate `net.feature.named_modules()`, collect leaf-name suffixes, compute intersection with `target_modules`. **Empty intersection ⇒ raise `ValueError`** listing actual suffixes. **Partial intersection ⇒ `warnings.warn(UserWarning)`** listing unmatched targets. Unit test `test_peft_target_modules_validated` covers (a) silent match, (b) ValueError on empty, (c) warning on partial. |
| 7.4 | `init_lora_weights="pissa_niter_4"` silent fallback on `peft<0.10` | MEDIUM | HIGH | Floor-pin `peft>=0.10.0` + post-wrap assertion that LoRA-A differs from fresh Gaussian init by > 1e-4 RMS. |
| 7.5 | `get_peft_model` changes state-dict key namespace | HIGH | HIGH | Phase 7 embeds `peft_config_dict` (rank, alpha, dropout, target_modules, init, task_type) into the Lightning checkpoint under key `dinov3_peft_config` via `on_save_checkpoint`. Phase 8 loader-v2 reads this key and re-wraps before `load_state_dict`. Round-trip test in this phase. |
| 7.6 | Target globs accidentally wrap classifier | LOW | HIGH | After `get_peft_model`, assert `self.net.classifier.weight.requires_grad == True` AND no `classifier.*` key under `base_model.*` prefix. |
| 7.7 | Missing `peft` install when user enables PEFT | MEDIUM | LOW | Lazy import in `_wrap_with_peft`; error includes `pip install -r requirements-dinov3.txt`. |
| 7.8 | **A3-N2 fix: PEFT-disabled regression must be behavioral, not a state-dict hash** | (was MEDIUM) | (was HIGH) | **RESOLVED.** See `test_peft_disabled_behavior_snapshot` below: fixed seed → 1 forward → `torch.allclose(logits, golden_logits, atol=1e-5)`; 1 SGD step → `math.isclose(loss, golden_loss, abs_tol=1e-5)`. Golden tensor at `tests/fixtures/peft_disabled_golden.pt`, regenerated by `tests/regression/regenerate_peft_disabled_golden.py`, frozen until the model architecture changes. |

### A3-N2: `test_peft_disabled_behavior_snapshot` — concrete contract

```python
# tests/regression/test_peft_disabled_behavior_snapshot.py
import torch, math, pytorch_lightning as pl
from src.algorithms.dinov3_linear import DINOv3Linear

GOLDEN_PATH = "tests/fixtures/peft_disabled_golden.pt"
ATOL = 1e-5

def test_peft_disabled_logits_match_golden(mock_hf_automodel):
    pl.seed_everything(42, workers=True)
    model = DINOv3Linear(conf={"dinov3": {"peft": {"enable": False},
                                          "freeze_backbone": True,
                                          "model_name": "stub"},
                               "num_classes": 4})
    model.eval()
    x = torch.load("tests/fixtures/peft_disabled_input.pt")  # fixed (2,3,224,224)
    with torch.no_grad():
        logits = model(x)
    golden = torch.load(GOLDEN_PATH)["logits"]
    assert torch.allclose(logits, golden, atol=ATOL), \
        f"PEFT-disabled forward drifted; max abs diff {(logits-golden).abs().max()}"

def test_peft_disabled_one_step_loss_matches_golden(mock_hf_automodel):
    pl.seed_everything(42, workers=True)
    model = DINOv3Linear(conf={"dinov3": {"peft": {"enable": False},
                                          "freeze_backbone": True,
                                          "model_name": "stub"},
                               "num_classes": 4})
    model.train()
    x = torch.load("tests/fixtures/peft_disabled_input.pt")
    y = torch.load("tests/fixtures/peft_disabled_target.pt")
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=1e-2, momentum=0.0)
    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward(); opt.step()
    golden_loss = torch.load(GOLDEN_PATH)["loss"].item()
    assert math.isclose(loss.item(), golden_loss, abs_tol=ATOL), \
        f"PEFT-disabled 1-step loss drifted: {loss.item()} vs {golden_loss}"
```

**Golden-tensor versioning policy:**
- File: `tests/fixtures/peft_disabled_golden.pt` (dict with keys `logits`,
  `loss`, `regenerated_at_sha`, `model_arch_version`).
- Regeneration script: `tests/regression/regenerate_peft_disabled_golden.py`
  (committed; must be invoked explicitly).
- **Frozen** under all changes EXCEPT a deliberate change to
  `DINOv3Classifier` architecture (e.g. swapping pooling, changing
  classifier head shape). When that happens, the same PR that changes
  the architecture must (a) run the regeneration script, (b) commit the
  new golden, (c) document the change in `CHANGELOG.md` under
  "Behavioral baselines updated".
- Comments / formatting / docs changes MUST NOT trigger regeneration.
- Atol `1e-5` is the contract; tightening or loosening requires
  architect sign-off in the PR.

### Files MODIFIED vs CREATED
- **CREATED**: `tests/regression/regenerate_peft_disabled_golden.py`, `tests/regression/test_peft_disabled_behavior_snapshot.py`.
- **MODIFIED**: `src/algorithms/dinov3_linear.py`.

### Dependency footprint added
- Module-top: none.
- Lazy: `peft`.
- `requirements-dinov3.txt`: `peft>=0.10.0,<0.14.0`.

### Rollback
`git revert <PHASE_7_SHA>` — linear-probe and full-FT unchanged. Phase 6 grad-flow test green.

### DAG
Depends on Phase 6 (sequencing); in code, only Phase 4.

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

Split into **Phase 5 (loader-v1, MD-Cls only)** and **Phase 8 (loader-v2,
SINCHI raw + PEFT)**.

### Phase 5 — Loader v1: MD-Cls `.ckpt` only

#### Goal
Land `src/models/classification/dinov3_base/` with `DINOv3BaseInference`
and `DINOv3CustomWeights`. `_extract_state_dict` handles only the
`"state_dict"` + `net.` prefix flavour. SINCHI flavours raise
`NotImplementedError("SINCHI checkpoint flavours land in Phase 8")`.

#### A3-N1 fix — unified absent-DINOv3 contract

**Single surface: `ImportError`.** The class symbol `DINOv3CustomWeights`
is exposed on `src.models.classification` via package-level
`__getattr__`. When `transformers` (or `peft`, or `huggingface_hub`) is
absent, attribute access raises `ImportError` with the **exact substring**
`"pip install -r requirements-dinov3.txt"` in the message.

```python
# src/models/classification/__init__.py  (end of file)
try:
    from .dinov3_base import *  # noqa: F401,F403
except ModuleNotFoundError as e:
    if e.name not in {"transformers", "huggingface_hub", "peft"}:
        raise

_DINOV3_NAMES = {"DINOv3CustomWeights", "DINOv3BaseInference"}

def __getattr__(name):
    if name in _DINOV3_NAMES:
        raise ImportError(
            f"{name} requires the optional DINOv3 dependencies. "
            f"Install with: pip install -r requirements-dinov3.txt"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Test contract (`test_dinov3_absent_actionable_error`, the I13 test):**

```python
# tests/regression/test_dinov3_absent_actionable_error.py
import subprocess, sys, textwrap, pytest

SCRIPT = textwrap.dedent("""
    from src.models import classification
    try:
        _ = classification.DINOv3CustomWeights
    except ImportError as e:
        assert "pip install -r requirements-dinov3.txt" in str(e), str(e)
        print("OK")
    else:
        raise AssertionError("expected ImportError, got none")
""")

def test_absent_dinov3_raises_actionable_import_error(tmp_path):
    # uv-isolated env with ONLY requirements.txt
    result = subprocess.run(
        ["uv", "run", "--isolated", "--with-requirements", "requirements.txt",
         "python", "-c", SCRIPT],
        capture_output=True, text=True, cwd="."
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
```

All four references in this document (Risk 5.1, Invariant I13,
Acceptance row 10, this test) now match: **`ImportError`** with substring
**`"pip install -r requirements-dinov3.txt"`**, asserted by `subprocess`
run in a `uv`-isolated env.

#### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 5.1 | Top-level wildcard import breaks ResNet users without `transformers` | HIGH | HIGH | `except ModuleNotFoundError as e: if e.name not in {"transformers", "huggingface_hub", "peft"}: raise`. Bugs in our own code propagate. Absent-class surface is `ImportError` via package `__getattr__` (see A3-N1 block above). |
| 5.2 | Corrupt `.ckpt` → unhelpful traceback | MEDIUM | LOW | Wrap `torch.load` in try/except re-raising `RuntimeError(f"Failed to torch.load({path}); valid checkpoint? Original: {e}")`. Test with 16-byte garbage. |
| 5.3 | `weights_only=True` future-default in torch ≥ 2.6 | MEDIUM | MEDIUM | Pass `weights_only=False` explicitly. Docs note: "Only load checkpoints from trusted sources." |
| 5.4 | `strict=False` swallows real bugs | MEDIUM | HIGH | v1 uses `strict=True`. Phase 8 v2 path: count missing/unexpected; raise if `len(missing) > 5` OR any starts with `classifier.` or `feature.` (modulo LoRA exceptions). |
| 5.5 | MD-Cls hidden-size vs supplied `model_name` mismatch | MEDIUM | HIGH | Pre-check: `state['classifier.weight'].shape[1] == self.net.embedding_dim`; raise listing both numbers. |

#### Files MODIFIED vs CREATED
- **CREATED**: `src/models/classification/dinov3_base/{__init__.py, base_classifier.py, custom_weights.py}`, `tests/regression/test_dinov3_absent_actionable_error.py`.
- **MODIFIED**: `src/models/classification/__init__.py` — append narrowly-guarded import + package-level `__getattr__` (above). Identity-proof: all four existing wildcard re-exports unchanged; `id(classification.CustomWeights)` pre/post matches.

#### DAG
Depends on Phase 4.

### Phase 8 — Loader v2: SINCHI raw + SINCHI PEFT

#### Goal
Extend `_extract_state_dict` to handle (a) SINCHI raw `.pt` with
`"model_state_dict"` + `backbone.*` → `feature.*` remap; (b) SINCHI PEFT
`.pt` with `backbone.base_model.model.*` keys, requiring either an
embedded `dinov3_peft_config` (own MD-Cls flavour from Phase 7) or an
explicit `peft_config=` kwarg.

#### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 8.1 | Auto-detect misclassifies → silent wrong-weights load | MEDIUM | HIGH | Three-way detection by key-prefix inspection. Ordering: (1) top-level `"state_dict"` + sub-key `"net."` → MD-Cls v1; (2) top-level `"model_state_dict"` + sub-key contains `"base_model.model."` → SINCHI PEFT; (3) top-level `"model_state_dict"` + sub-key `"backbone."` → SINCHI raw; (4) else `ValueError` listing `list(ckpt.keys())[:10]`. Each branch unit-tested. |
| 8.2 | LoRA adapter keys present but no `peft_config` | HIGH | HIGH | Hard-raise: `ValueError("Checkpoint contains LoRA adapter keys (matched: <list>) but no peft_config was provided AND no 'dinov3_peft_config' is embedded in the checkpoint. Pass peft_config={r:..., lora_alpha:..., target_modules:[...]} or use a non-PEFT checkpoint.")`. |
| 8.3 | Synthetic-only fixtures hide real format drift | HIGH | HIGH | **MANDATORY ENTRY GATE**: at least one real SINCHI raw `.pt` and one real SINCHI PEFT `.pt` are loaded; their key layouts are recorded in `docs/design/dinov3-integration/sinchi_key_layouts.md` (tracked docs path, per A2-N1 directive). If unobtainable, Phase 8 is deferred; v1 loader remains the shipped surface. |
| 8.4 | hidden_size mismatch | MEDIUM | HIGH | Pre-check applied to remapped keys. |
| 8.5 | `wget` fallback hits redirect/auth | LOW | LOW | Docs recommend `huggingface_hub.snapshot_download`; wget kept only for direct-URL `.pt`. |

#### Files MODIFIED vs CREATED
- **CREATED**: `docs/design/dinov3-integration/sinchi_key_layouts.md`.
- **MODIFIED**: `src/models/classification/dinov3_base/base_classifier.py`.

#### DAG
Depends on Phases 5 and 7.

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

### Goal
Append commented YAML block to `configs/config.yaml`; ship three sample
configs; write `docs/dinov3.md`; append to `docs/usage.md` and
`README.md`; add `CHANGELOG.md` entry.

> Minimal `docs/dinov3.md` v0.1 (~40 lines: HF auth + memory warnings) ships
> with **Phase 5** so users don't hit HF auth before docs exist. Phase 9 is
> the full expansion.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 9.1 | YAML append breaks parse | LOW | HIGH | I4 (dict-equality) regression test. |
| 9.2 | Sample config OOMs on small GPU | HIGH | MEDIUM | Default sample = ViT-S (`facebook/dinov3-vits16-pretrain-lvd1689m`). Memory table labels rows MEASURED or ESTIMATED. |
| 9.3 | Doc example uses gated model without `huggingface-cli login` callout | MEDIUM | LOW | Every DINOv3 code block prefaced with `pip install -r requirements-dinov3.txt` AND `huggingface-cli login`. |
| 9.4 | `requirements.txt` accidentally gains heavy deps | (would-be) HIGH | HIGH | Phase 9 modifies `requirements.txt` only to add `# DINOv3 training/inference: pip install -r requirements-dinov3.txt`. I2 hashes runtime-deps lines. |
| 9.5 | Discoverability (users follow only default install) | HIGH | MEDIUM | README capability bullet + `docs/usage.md` opener + sample-config comment-header all repeat: *"Requires `pip install -r requirements-dinov3.txt` and `huggingface-cli login`."* |
| 9.6 | License/citation | LEGAL | LEGAL | `docs/dinov3.md` includes License, Citation (BibTeX), Redistribution-of-derived-weights. |
| 9.7 | Error taxonomy | MEDIUM | LOW | `docs/dinov3.md` Troubleshooting table maps each `RuntimeError`/`ValueError`/`ImportError` raised by Phase-4–8 to cause + fix. |
| 9.8 | Reproducibility | MEDIUM | MEDIUM | `docs/dinov3.md` Reproducibility section: `seed_everything`, `Trainer(deterministic=True)`, pinned versions, embedded `dinov3_peft_config`. |

### Files MODIFIED vs CREATED
- **CREATED**: `configs/config_dinov3_linear.yaml`, `configs/config_dinov3_finetune.yaml`, `configs/config_dinov3_lora.yaml`, `docs/dinov3.md`, `CHANGELOG.md` (if absent).
- **MODIFIED**: `configs/config.yaml` (commented append), `README.md`, `docs/usage.md`, `requirements.txt` (1-line comment).

### Dependency footprint added
None.

### DAG
Depends on Phases 4, 5, 7, 8.

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

### Goal
Fix `src/models/classification/resnet_base/custom_weights.py` where
`num_layers=50` is hard-coded. Add `num_layers: int = 50` defaulted
kwarg. Sequenced as **Phase 1**.

### Failing-test-first protocol
Phase 0 ships `tests/regression/test_resnet18_loads.py` marked
`@pytest.mark.xfail(strict=True, reason="bug fixed in Phase 1")`. Phase
1's single commit (a) deletes the marker and (b) applies the kwarg
patch.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1.1 | Downstream caller passing 4th positional arg | NEAR-ZERO | LOW | `grep "CustomWeights(" src` confirms zero 4-positional callers. |
| 1.2 | ResNet-50 regresses due to patch typo | LOW | HIGH | Regression test exercises ResNet-18 AND ResNet-50 ckpt load with `strict=True`. |
| 1.3 | Bisect ambiguity | (would-be) LOW | MEDIUM | Phase 1 is its own commit. |
| 1.4 | Migration note hidden in final docs | MEDIUM | LOW | CHANGELOG entry lands in Phase 1, not Phase 9. |

### Files MODIFIED vs CREATED
- **CREATED**: `tests/regression/test_customweights_num_layers.py` (xfail flip from Phase 0); `CHANGELOG.md` entry.
- **MODIFIED**: `src/models/classification/resnet_base/custom_weights.py`:
  ```
  -    def __init__(self, weights=None, class_names=None, device="cpu"):
  +    def __init__(self, weights=None, class_names=None, device="cpu", num_layers: int = 50):
  ...
  -                                                   num_cls=self.num_cls, num_layers=50, url=None)
  +                                                   num_cls=self.num_cls, num_layers=num_layers, url=None)
  ```

### Rollback
`git revert <PHASE_1_SHA>` restores hard-code; regression fails for ResNet-18 (correct).

### DAG
Depends only on Phase 0.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

### Repo invariants (13)

| # | Invariant | Test name | Mechanism |
|---|---|---|---|
| I1 | `python -c "import src"` succeeds in env with ONLY `requirements.txt` installed | `test_default_env_imports_clean` | Subprocess via `uv run --isolated --with-requirements requirements.txt python -c "..."` |
| I2 | `requirements.txt` runtime deps unchanged modulo comments | `test_requirements_runtime_lines_hash` | Hash of non-comment, non-blank lines |
| I3 | ResNet baseline predictions stable (**behavior snapshot**, not state-dict hash) | `test_resnet_baseline_predictions` | argmax labels + confidence within 1e-4 vs `tests/fixtures/baseline_predictions.json`; `seed_everything(42)` + `Trainer(deterministic=True)` |
| I4 | `yaml.safe_load(open('configs/config.yaml'))` returns pre-change dict | `test_default_config_yaml_equal` | **dict equality** (not yaml-dump string) |
| I5 | `CustomWeights(weights=..., class_names=[...])` loads ResNet-50 in env with no DINOv3 deps | `test_customweights_no_dinov3_deps` | Subprocess isolated env |
| I6 | `algorithms.__dict__['Plain']` is the same class object | `test_plain_algorithm_class_identity` | `id()` comparison via subprocess on pre-change SHA |
| I7 | `models.__dict__['PlainResNetClassifier']` is the same class object | `test_plain_resnet_class_identity` | same |
| I8 | `main.py` SHA-256 unchanged | `test_main_unchanged` | Byte-equality (PERMANENT) |
| I9 | `src/algorithms/plain.py` SHA-256 unchanged | `test_plain_py_unchanged` | Byte-equality (PERMANENT) |
| I10 | `src/models/plain_resnet.py` SHA-256 unchanged | `test_plain_resnet_py_unchanged` | Byte-equality (PERMANENT) |
| I11 | `src/datasets/custom.py` SHA-256 unchanged EXCEPT in Phase 4 (kwarg) where hash updates in same commit + `test_custom_crop_default_is_224` becomes permanent | `test_custom_dataset_byte_or_behaviour` | Hash check + behaviour check |
| I12 | `CustomWeights(weights=w, class_names=c, device=d)` legacy 3-arg call works | `test_customweights_legacy_call_signature` | Live import + load |
| I13 | **In env without `transformers`, `from src.models import classification` succeeds AND `classification.DINOv3CustomWeights` raises `ImportError` whose message contains substring `"pip install -r requirements-dinov3.txt"`** | `test_dinov3_absent_actionable_error` | Subprocess in `uv`-isolated env; see A3-N1 block above. **A3-N1 RESOLVED.** |

I8–I10 are PERMANENT byte-equality. I11 is byte-equality except for the
one allowed transform-kwarg commit. Everything else is protected by
behaviour / dict / import tests, not byte-hashes (A3-N2 directive: no
state-dict hashes as gates).

### Cross-cutting policy

- Every phase MUST run `pytest tests/regression/ -x` before merging.
- `tests/regression/dryrun_revert.sh` (committed in Phase 0):
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  SHA="$1"
  WT="$(pwd)/.revert-dryrun-$$"
  trap 'git worktree remove --force "$WT" 2>/dev/null || true' EXIT
  git worktree add "$WT" HEAD
  pushd "$WT"
  git revert --no-edit "$SHA"
  pytest tests/regression -x
  popd
  ```
  Repo-local sibling worktree (no `/tmp`); cleanup via `trap`.
- CI matrix activates **at Phase 3**: job (a) `pip install -r requirements.txt`; job (b) `pip install -r requirements.txt -r requirements-dinov3.txt`.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Pyramid

| Layer | Count target | Distribution |
|---|---|---|
| **Unit** (CPU, <1s, no network) | ~18 | Mixup 4, model 4, algorithm 4, extract_state_dict v1+v2 6 |
| **Integration** (CPU, stub backbone, ~10s) | ~12 | Linear probe 2, full FT 2, LoRA 1, PiSSA 1, loader v1 2, loader v2 SINCHI raw+PEFT 2, transform-518 dataloader 1, PEFT roundtrip 1 |
| **Regression** (invariants + behavior snapshots) | 13 (I1–I13) + 1 (`test_peft_disabled_behavior_snapshot`) | All in Phase 0 except I13 (Phase 5) and PEFT-disabled snapshot (Phase 7) |
| **E2E** (real HF download, GPU) | 4 — manual, never in CI | Linear probe, full FT, LoRA, PiSSA |

### Stub strategy (`tests/conftest.py`)

```python
import torch, torch.nn as nn
from types import SimpleNamespace
import pytest

class _StubConfig:
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size

class _StubBackbone(nn.Module):
    """Tiny ViT-shaped stub. Module names mirror HF DINOv3ViT so PEFT
    target_modules tests are meaningful."""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = _StubConfig(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, pixel_values=None, **kw):
        B = pixel_values.shape[0]
        h = self.pool(pixel_values.mean(1, keepdim=True)).flatten(1)
        h = h.expand(B, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=h.unsqueeze(1))

@pytest.fixture
def mock_hf_automodel(monkeypatch):
    def _from_pretrained(*a, **kw):
        return _StubBackbone(hidden_size=64)
    import transformers
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        staticmethod(_from_pretrained))
```

### Fixtures (all under `tests/_artifacts/` or `tests/fixtures/` — NO `/tmp`)

- `tests/fixtures/baseline_predictions.json` (Phase 0)
- `tests/fixtures/resnet18_synth.ckpt` (Phase 1)
- `tests/fixtures/peft_disabled_input.pt`, `peft_disabled_target.pt`, `peft_disabled_golden.pt` (Phase 0, consumed by Phase 7 — see A3-N2 contract)
- `tests/_artifacts/dinov3_linear_mdcls.ckpt` (Phase 4 → 5 → 7)
- `tests/fixtures/sinchi_raw_synth.pt`, `sinchi_peft_synth.pt` (Phase 8)
- `docs/design/dinov3-integration/sinchi_key_layouts.md` (Phase 8 — tracked docs path, per A2-N1 directive)

### HF auth handling

CI never authenticates with HF. All Phase 4–8 tests use
`mock_hf_automodel`. Real-HF tests are marked `@pytest.mark.requires_hf`
and gated on `HF_TOKEN` env; default `pytest` excludes the marker.
`huggingface_hub` references gate via `pytest.importorskip("huggingface_hub")`.

### CI vs local

- **CI baseline (Phase 0+)**: unit + integration (mocked HF) + regression.
- **CI matrix (Phase 3+)**: (a) default-only, (b) default + dinov3. CUDA/HF/network disabled.
- **Local nightly**: `pytest -m requires_hf` and `pytest -m gpu`. Never CI.

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

Mapped to research checklist (§8). Every command executable; expected outputs concrete.

| # | Checklist item | Phase | Command | Expected | Tier |
|---|---|---|---|---|---|
| 1 | ResNet path behaviour stable | 0 | `pytest tests/regression/test_resnet_baseline_predictions.py -x` | predictions match `baseline_predictions.json` (atol=1e-4) | CI |
| 2 | Linear-probe trains | 4 | `python main.py --config configs/config_dinov3_linear.yaml --dev` | ckpt at `weights/Crop/DINOv3Linear/*.ckpt`; backbone grad-norms == 0 | local CPU (mocked HF for CI) |
| 3 | Full-FT trains | 6 | same, `freeze_backbone: False` | grad-norms > 0; trainable_params ≈ 86M | local GPU |
| 4 | Mixup toggle | 4+6 | `use_noisy_same_class_mixup: True` | loss curves differ; no shape errors | local CPU |
| 5 | LoRA + PiSSA | 7 | `peft.enable: True` (both methods) | trainable_params < 5% total; PiSSA LoRA-A diverges from gaussian init | local GPU |
| 6 | Load own MD-Cls ckpt | 5 | `DINOv3CustomWeights(weights=<ckpt#2>, class_names=[...])` | missing==[], unexpected==[] | CI (synthetic) + local |
| 7 | Load SINCHI raw | 8 | same + real SINCHI raw `.pt` | strict-False load; remap applied | local (real artifact) |
| 8 | Load SINCHI PEFT | 8 | same + `peft_config={...}` + real PEFT `.pt` | re-wrap succeeds, adapter keys consumed | local (real artifact) |
| 9 | CustomWeights regression | 1 | `pytest tests/regression/test_customweights_num_layers.py` | num_layers=18 AND 50 load with `strict=True` | CI |
| 10 | **Default env import + actionable error (A3-N1 fix)** | 0/5 | `uv run --isolated --with-requirements requirements.txt python -c "from src.models import classification; \\ntry: classification.DINOv3CustomWeights\\nexcept ImportError as e: assert 'pip install -r requirements-dinov3.txt' in str(e); print('OK')\\nelse: raise SystemExit(1)"` | exit 0, stdout `"OK"` | CI |
| 11 | Transform sizes | 4 | `pytest tests/regression/test_custom_crop_default_is_224.py tests/integration/test_dinov3_dataloader_emits_518.py` | ResNet `(B,3,224,224)`, DINOv3 `(B,3,518,518)` | CI |
| 12 | Real SINCHI key evidence | 8 | review `docs/design/dinov3-integration/sinchi_key_layouts.md` | file present, lists top-level + sample sub-keys for both flavours | manual gate |
| 13 | PEFT target_modules validation | 7 | `pytest tests/unit/test_peft_target_modules_validated.py` | empty→ValueError, partial→warning, full→silent | CI |
| 14 | **PEFT-disabled behavior snapshot (A3-N2 fix)** | 7 | `pytest tests/regression/test_peft_disabled_behavior_snapshot.py` | `allclose(logits, golden, atol=1e-5)` AND `isclose(loss, golden_loss, abs_tol=1e-5)` | CI |

### Definition of "done"

All 13 invariants + all 14 checklist tests pass + CI green on both
matrix jobs + `docs/dinov3.md` reviewed for HF auth / GPU memory / PEFT
defaults / license / citation + `sinchi_key_layouts.md` populated (or
Phase 8 explicitly deferred with documented "experimental" status).

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

### Top-level risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A phase breaks I1 (default env import) | MEDIUM | CATASTROPHIC | Per-phase regression gate; lazy imports; narrowed try/except; subprocess-isolated env tests (I1, I5, I13). |
| R2 | DINOv3 HF model gated → cryptic auth error | HIGH | MEDIUM | Wrap `from_pretrained` with actionable `RuntimeError`; docs include `huggingface-cli login`; CI never authenticates. |
| R3 | GPU OOM on full-FT default | HIGH | MEDIUM | Sample ships ViT-S + bs=4 + fp16; memory table labels MEASURED vs ESTIMATED. |
| R4 | `peft` API drift | MEDIUM | MEDIUM | Pin `peft>=0.10.0,<0.14.0`; CI matrix tests both bounds. |
| R5 | Loader auto-detect silently misclassifies | MEDIUM | HIGH | Key-prefix detection; fail-loud on missing classifier keys; hidden-size pre-check; LoRA-without-peft_config hard raises. |
| R6 | Reverting Phase 6 with 7 present | LOW (with DAG) | MEDIUM | Reverse-DAG enforced; `dryrun_revert.sh` refuses out-of-order; documented order 9→0. |
| R7 | Lightning checkpoint format drift | LOW | LOW | Pin Lightning floor in `requirements-dinov3.txt`. |
| R8 | Reproducibility | MEDIUM | MEDIUM | `seed_everything` + `Trainer(deterministic=True)`; embedded `dinov3_peft_config` + `model_name` + `image_size` in checkpoint metadata. |
| R9 | DINOv3 license/redistribution | LEGAL | LEGAL | Phase 9 docs; no automatic download; checkpoint metadata records source `model_name`. |
| R10 | Non-CUDA execution | MEDIUM | LOW | Sample configs use `precision: 16-mixed` with "CUDA-only" comment; default fallback `precision: 32`. |
| R11 | **Golden-tensor drift on accidental architecture changes** (A3-N2) | LOW | MEDIUM | Frozen golden tensor; regeneration requires explicit script invocation + `CHANGELOG` entry; failing snapshot test blocks PR. |

### Per-phase rollback DAG (reverse order)

```
PHASE_9 → revert → pytest tests/regression -x                                                                    → green
PHASE_8 → revert → pytest tests/regression tests/integration/test_dinov3_custom_weights_mdckpt.py -x             → green
PHASE_7 → revert → pytest tests/regression tests/integration/test_dinov3_finetune_grad_flow.py -x                → green
PHASE_6 → revert → pytest tests/regression tests/integration/test_dinov3_linear_probe.py -x                      → green
PHASE_5 → revert → pytest tests/regression tests/integration/test_dinov3_linear_probe.py -x                      → green
PHASE_4 → revert → pytest tests/regression tests/unit/test_mixup.py -x                                           → green
PHASE_3 → revert → pytest tests/regression -x                                                                    → green
PHASE_2 → revert → pytest tests/regression -x                                                                    → green
PHASE_1 → revert → pytest tests/regression -x → test_resnet18_loads RED (expected); test_resnet50 GREEN
PHASE_0 → revert → manual confirmation; baseline removed
```

### Single-command emergency revert

```bash
git revert --no-edit \
  $PHASE_9_SHA $PHASE_8_SHA $PHASE_7_SHA $PHASE_6_SHA \
  $PHASE_5_SHA $PHASE_4_SHA $PHASE_3_SHA $PHASE_2_SHA $PHASE_1_SHA
# Phase 0 (tests) stays: better tests, zero DINOv3 code
pytest tests/regression -x
```

---

## Rationale (risk/backwards-compat lens, round-3 final)

The plan converges on the inquisitor's eight binding directives:

1. **Single dependency policy.** `requirements-dinov3.txt` contains
   `transformers`, `peft`, `huggingface_hub`. All three lazy-imported.
   All three covered by I1 / I13 absent-env tests.
2. **Runtime PEFT target validation.** Empty intersection → `ValueError`;
   partial → `warnings.warn(UserWarning)`. `TaskType.FEATURE_EXTRACTION`
   retained per HF docs citation.
3. **Transform = defaulted `image_size=224` kwarg on `Custom_Crop`.**
   (My round-2 proposal won; kept verbatim, retargeted from the
   nonexistent `Classification_Inference_Transform` to the actual
   `Custom_Crop` class at line 240 of `src/datasets/custom.py`.)
4. **SINCHI key-layout evidence under `docs/design/dinov3-integration/sinchi_key_layouts.md`**
   (tracked docs path, not transient artifacts).
5. **No packaging migration.** No `pyproject.toml` introduced; requirements-only.
6. **Behavior snapshots for all regression gates.** Zero state-dict hashes
   as pass/fail gates (A3-N2 resolved).
7. **Absent-DINOv3 surface = `ImportError`** with substring
   `"pip install -r requirements-dinov3.txt"`. Tests assert exactly this
   (A3-N1 resolved).
8. **Disagreements purged** per inquisitor directive.

---

## New ledger items proposed

(None. All 12 scope items addressed; nothing new is required for convergence.)

---

## Consensus points (round-3 confirmed)

- **Loader split v1 (MD-Cls) / v2 (SINCHI)** — adopted from architect-1 in round 2; kept.
- **Failing-test-first protocol for CustomWeights bug fix** — adopted from architect-2.
- **Bug fix as Phase 1** — adopted from architect-1.
- **Single shared `mock_hf_automodel` fixture** — adopted from architect-2; extended with PEFT-aware module names.
- **Subprocess isolated env for "no transformers" tests** — adopted from architect-2.
- **Behavior-snapshot ResNet regression** — adopted from architect-1's framing (F-03).
- **Embed `peft_config` in checkpoints** — universal consensus.
- **`requirements-dinov3.txt` (not `pyproject.toml` extras)** — universal consensus in round 2.
- **Coverage thresholds are not a DINOv3 acceptance gate** — universal consensus.
- **PEFT lands in Phase 7, not earlier** — universal consensus.

---

## Disagreements

*(empty — per inquisitor final-synthesis directive. All round-1 / round-2
disagreements I raised against architect-1 and architect-2 were
adjudicated MISTAKEN-BY=architect-3. The transform-mechanism question
was RESOLVED-TOWARD=architect-3; my approach is the agreed final.
Re-reading architect-1 and architect-2 round-2 proposals confirms: A1
uses `requirements-dinov3.txt`, A2 says coverage is advisory and PEFT
lands in Phase 7. No live disputes remain.)*

---

## Evolution (round 2 → round 3)

1. **A3-N1 resolved.** Absent-DINOv3 contract unified to a single
   surface: `ImportError` raised from package-level `__getattr__` with
   substring `"pip install -r requirements-dinov3.txt"`. All four
   document references (Risk 5.1, I13, Acceptance row 10, test) now
   match. Test landed: `test_dinov3_absent_actionable_error` runs a
   subprocess in `uv`-isolated env and asserts both the exception type
   AND the message substring.
2. **A3-N2 resolved.** PEFT-disabled regression switched from state-dict
   hash to a behavioral snapshot: fixed seed → 1 forward → logits
   `allclose` golden (atol=1e-5) AND 1 SGD step → loss `isclose` golden
   (abs_tol=1e-5). Golden tensor at `tests/fixtures/peft_disabled_golden.pt`,
   regenerated by committed `regenerate_peft_disabled_golden.py`, frozen
   except on deliberate `DINOv3Classifier` architecture changes.
3. **Transform target corrected** from `Classification_Inference_Transform`
   (nonexistent) to `Custom_Crop.__init__` (line 240 of
   `src/datasets/custom.py`), matching the lead directive and verified
   by file inspection.
4. **Acceptance row 10** rewritten to use an `ImportError`-catching
   one-liner that asserts the message substring (was a bare import).
5. **`huggingface_hub` explicitly named** in `requirements-dinov3.txt`
   (previously implicit), per A1-N1.
6. **Disagreements section emptied** per inquisitor directive.
7. **Acceptance count** 13 → 14 (added `test_peft_disabled_behavior_snapshot`).
8. **R11 added** to top-level risk register (golden-tensor governance).

---

## NEW This Round

1. **A3-N1 fix landed**: unified absent-DINOv3 surface to `ImportError`
   with required-substring contract; concrete `__getattr__`
   implementation and `subprocess` test code provided inline.
2. **A3-N2 fix landed**: PEFT-disabled regression converted to
   behavioral snapshot (logits `allclose` + 1-step loss `isclose`);
   golden-tensor versioning policy documented (frozen, regeneration via
   committed script, change requires `CHANGELOG` entry).
3. **`huggingface_hub` named explicitly** in the optional-deps contract
   and absent-env tests (closes A1-N1 against the joint synthesis).
4. **Transform target verified by code inspection**: kwarg lands on
   `Custom_Crop.__init__` in `src/datasets/custom.py` (line 240);
   instance-built transforms replace module-level dict only when the
   kwarg is non-default. Test name updated to
   `test_custom_crop_default_is_224`.

STATUS: NEW=4 DISAGREEMENTS=0
