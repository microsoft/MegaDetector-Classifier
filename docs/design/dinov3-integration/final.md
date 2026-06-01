# DINOv3 integration — final phased plan

> **Status**: converged after 3 design rounds (3 architects + 1 inquisitor).
> See `round_03/inquisitor_review.md` for the convergence verdict
> (`CONVERGED SCOPE_CHECK=PASS COVERED=12/12 NEW=0`).
>
> **Scope source**: `/shared/home/aahernandez/.copilot/session-state/fa5699aa-9c12-4f4d-9834-b337cda7f3d5/research/what-are-the-steps-necessary-to-allow-the-repo-on-.md`
> **Target repo**: `/shared/home/aahernandez/MegaDetector-Classifier/`
> **Reference repo**: `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/`

---

## 1. Executive overview

### Why
Add fine-tuning capability (linear probing, full fine-tune, PEFT/LoRA/PiSSA)
using DINOv3 backbones, and the ability to load checkpoints produced by
the new pipeline AND by the existing SINCHI training code, while
preserving 100% of the current ResNet pipeline's behavior.

### What
Ten ordered phases (0–9). Phase 0 establishes baseline tests. Phases 1–9
land code/config/docs additively. Two ship gates (after Phase 5 and after
Phase 8) for user feedback.

### Load-bearing invariant
> Unchanged default `configs/config.yaml` + unchanged
> `pip install -r requirements.txt` + same seed/data ⇒
> bit-identical training, checkpoints, and predictions to the
> pre-change repo for the ResNet pipeline.

Enforced by 13 regression invariants (§6) tested in CI.

### Eight binding directives (from round-2 inquisitor adjudication)
1. **Single dependency policy.** `requirements.txt` unchanged; `requirements-dinov3.txt` contains exactly `transformers`, `peft`, `huggingface_hub`. All three lazy/import-local; all three covered by absent-env tests (I1, I13).
2. **PEFT target validation.** Empty intersection of `target_modules` with backbone module names ⇒ `ValueError`. Partial ⇒ `warnings.warn(UserWarning)` listing unmatched targets. Keep `TaskType.FEATURE_EXTRACTION`.
3. **Transform mechanism = defaulted kwarg on existing class.** Add `image_size: int = 224` to `Custom_Crop.__init__` in `src/datasets/custom.py`. DINOv3 path overrides to 518. No separate transform shim.
4. **SINCHI key-layout evidence under tracked docs.** File lives at `docs/design/dinov3-integration/sinchi_key_layouts.md`. Tests parse fenced code blocks from this doc; no `tests/_artifacts/` dependency.
5. **No packaging migration.** Test installs are requirements-only (`pip install -r requirements.txt -r requirements-dev.txt`). `pyproject.toml` (if any) carries only `[tool.pytest.ini_options]` — no `[project]`/`[project.optional-dependencies]`/`[build-system]`.
6. **Behavior snapshots, not state-dict hashes.** Regression gates assert logits/loss/predictions within tolerance. State-dict hashes are not pass/fail gates.
7. **Absent-DINOv3 surface = `ImportError`.** Package-level `__getattr__` in `src/models/classification/__init__.py` raises `ImportError` with the exact substring `"pip install -r requirements-dinov3.txt"`. Tests assert both the exception type and the message substring.
8. **Coverage threshold (`--cov-fail-under`) is advisory only**, not a DINOv3 acceptance gate.

### Scope items (12)
Every section below references the scope_item id it covers. Inquisitor
verifies via `<a name="...">` anchors in this document.

```
phase-decomposition          mixup-utility-phase       model-and-algorithm-phase
full-finetune-phase          peft-phase                inference-loader-phase
config-and-docs-phase        bug-fix-phase             regression-protection
test-strategy                acceptance-and-validation risk-and-rollback
```

---

## 2. Phase index

| # | Phase | New files | Modified files | New deps | Revert unit | Risk |
|---|---|---|---|---|---|---|
| 0 | Test scaffolding + behavior baseline | `tests/conftest.py`, `tests/regression/test_invariants.py`, `tests/regression/dryrun_revert.sh`, `tests/fixtures/{baseline_predictions.json, peft_disabled_*.pt}`, `scripts/rollback_dry_run.sh`, (optional) `pyproject.toml` with pytest block only | none | none | PR #0 | trivial |
| 1 | `CustomWeights` `num_layers` bug fix | `tests/regression/test_customweights_num_layers.py` (xfail flip) | `src/models/classification/resnet_base/custom_weights.py` (+1 defaulted kwarg) | none | PR #1 | low |
| 2 | Mixup utility port | `src/utils/mixup.py`, `tests/unit/test_mixup.py` | none | none | PR #2 | trivial |
| 3 | Optional-deps wiring + `Custom_Crop` kwarg | `requirements-dinov3.txt` | `requirements.txt` (1 comment line), `src/datasets/custom.py` (+1 defaulted kwarg) | none in default; `transformers`+`peft`+`huggingface_hub` via dinov3 file | PR #3 | low |
| 4 | DINOv3 model + linear-probe algorithm (**MVP**) | `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py` | `src/models/__init__.py`, `src/algorithms/__init__.py` (+1 line each) | none in default; `transformers` lazy | PR #4 | low |
| 5 | Loader v1 (MD-Cls `.ckpt` only) + minimal docs | `src/models/classification/dinov3_base/{__init__,base_classifier,custom_weights}.py`, `docs/dinov3_experimental.md` (HF auth + memory + license), `tests/regression/test_dinov3_absent_actionable_error.py` | `src/models/classification/__init__.py` (+1 guarded import + `__getattr__`) | none in default | PR #5 | low |
| — | **Gate 1**: ship + collect feedback (≥1 week) | — | — | — | — | — |
| 6 | Full fine-tune toggle | none | `src/algorithms/dinov3_linear.py` (extend `configure_optimizers`) | none additional | PR #6 | low |
| 7 | PEFT (LoRA + PiSSA) + checkpoint roundtrip | `tests/regression/{regenerate_peft_disabled_golden.py, test_peft_disabled_behavior_snapshot.py}` | `src/algorithms/dinov3_linear.py` (adds `_wrap_with_peft`, `on_save_checkpoint`) | `peft` (lazy) via dinov3 reqs | PR #7 | medium |
| 8 | Loader v2 (SINCHI raw + SINCHI PEFT) | `docs/design/dinov3-integration/sinchi_key_layouts.md` | `src/models/classification/dinov3_base/base_classifier.py` (extend `_extract_state_dict`) | none additional | PR #8 | medium-high |
| — | **Gate 2**: ship + collect feedback (≥1 week) | — | — | — | — | — |
| 9 | Config + docs consolidation | `configs/config_dinov3_{linear,finetune,lora}.yaml`, `docs/dinov3.md`, `CHANGELOG.md` | `configs/config.yaml` (commented append), `README.md`, `docs/usage.md` | none | PR #9 | trivial |

Total: 10 PRs (0–9). Phases 4–8 are pure adds. Phase 1, Phase 3 (Custom_Crop kwarg), and the Phase 9 commented append are the only edits to already-shipped files; all backwards-compatible by construction.

### Phase DAG

```
P0 ──┬─► P1 ───────────────────────────────────────────────────────────┐
     ├─► P2 ──┐                                                        │
     ├─► P3 ──┼─► P4 ──┬─► P5 ──┬───────────── Gate 1 ──┬─► P6 ──► P7 ─┴─► P8 ── Gate 2 ──► P9
     └────────┘        │        │                       │
                       │        └──────────► P8 (DAG dep)│
                       └──────► P9 (DAG dep) ────────────┘
```

P1, P2, P3 are mutually independent prior to P4. Reverse-DAG rollback
chain: P9 → P8 → P7 → P6 → P5 → P4 → P3 → P2 → P1 → P0.

---

<a name="phase-decomposition"></a>
## 3. Phase decomposition — addresses scope_item:phase-decomposition

### Ordering rationale (risk-and-MVP lens)
1. **P0 first** — establishes behavior baselines (logits/predictions/configs), `rollback_dry_run.sh`, and the I1–I13 invariant suite. Nothing lands without these gates in place.
2. **P1 in isolation** — only edit to behavior-relevant existing code; isolated commit enables clean bisect. P0 ships `test_resnet18_loads.py` marked `xfail(strict=True)`; P1 flips the marker AND applies the patch in one commit (failing-test-first inside the PR).
3. **P2 (mixup)** — leaf utility, no callers yet. Pure add.
4. **P3 (deps + transform kwarg)** — the `requirements-dinov3.txt` file must exist before P4 lazy imports point to it; the `Custom_Crop` kwarg lands here so P4 can simply *use* it.
5. **P4 (MVP core)** — DINOv3Classifier + DINOv3Linear with `freeze_backbone: True` only. PEFT and full-FT explicitly `NotImplementedError`. Mixup wired but `use_mixup: False`.
6. **P5 (loader v1, MD-Cls only)** — MVP CAPABILITY ships. Users can train via P4 and load their own checkpoints via P5. SINCHI flavors raise `NotImplementedError("SINCHI checkpoint flavours land in Phase 8")`.
7. **Gate 1** — ship. Collect feedback for ≥1 week before continuing.
8. **P6 → P7** — full-FT before PEFT, in increasing-risk order.
9. **P8 (loader v2, SINCHI)** — gated on real SINCHI key-inspection evidence at `docs/design/dinov3-integration/sinchi_key_layouts.md`. If unobtainable, P8 deferred; v1 loader remains the shipped surface.
10. **Gate 2** — ship.
11. **P9 (config + docs)** — last, so docs match shipped behavior. A minimal docs stub (`docs/dinov3_experimental.md`, ~40 lines: HF auth + memory) ships with P5 so first users don't hit auth errors before docs exist.

### Entry / exit criteria (per phase, unified template)
- **Entry**: (a) predecessor phases merged; (b) `pytest tests/regression -x` green on `main`; (c) phase risk register re-read; (d) rollback SHA recorded.
- **Exit**: (a) phase tests green; (b) regression suite green; (c) rollback dry-run executed via `bash tests/regression/dryrun_revert.sh <PHASE_SHA>`; (d) phase entry appended to `CHANGELOG.md`.

### Per-phase deliverables summary
Already tabulated in §2. Detailed contents in §§4.1–4.10.

---

## 4. Per-phase specifications

### 4.0. Phase 0 — Test scaffolding + behavior baseline

**Goal.** Lock the current ResNet pipeline behavior with snapshot tests
before any code change. Ship the rollback helper and the I1–I12 invariant
suite. (I13 lands with P5 since it requires the `__getattr__` shim.)

**Files created.**
- `tests/conftest.py` — shared fixtures including `mock_hf_automodel` (stub backbone whose module names mirror HF DINOv3 ViT so PEFT `target_modules` tests are meaningful).
- `tests/regression/test_invariants.py` — I1, I2, I4–I12.
- `tests/regression/test_resnet_baseline_predictions.py` — behavior snapshot (I3).
- `tests/regression/dryrun_revert.sh` — `git worktree`-based revert sandbox with `trap` cleanup.
- `tests/fixtures/baseline_predictions.json` — golden ResNet predictions.
- `tests/fixtures/peft_disabled_input.pt`, `peft_disabled_target.pt`, `peft_disabled_golden.pt` — consumed by P7.
- `scripts/rollback_dry_run.sh` — top-level wrapper for the revert dry-run.
- (Optional) `pyproject.toml` containing **only** `[tool.pytest.ini_options]` (testpaths, addopts, markers, `tmp_path_retention_policy`). No `[project]`, no `[build-system]`.

**Files modified.** None.

**Tests.**
- `test_resnet_baseline_predictions.py` — fix seed, run inference, compare argmax + confidence to JSON golden within `atol=1e-4`.
- I1, I2, I4, I5, I6, I7, I8–I10, I11, I12 — see §6.
- 6 baseline tests covering ResNet-18 + ResNet-50 prediction snapshots.

**Acceptance.** `pytest tests/regression -x` green.

**Rollback.** Revert PR #0; no code paths affected.

**Risks.** None substantive; trivial phase.

---

### 4.1. Phase 1 — `CustomWeights` `num_layers` bug fix — addresses scope_item:bug-fix-phase

<a name="bug-fix-phase"></a>

**Goal.** Replace the hard-coded `num_layers=50` in
`src/models/classification/resnet_base/custom_weights.py` with a
defaulted kwarg so ResNet-18 checkpoints load correctly.

**Failing-test-first protocol.** P0 ships
`tests/regression/test_customweights_num_layers.py` marked
`@pytest.mark.xfail(strict=True, reason="bug fixed in Phase 1")`. P1's
single commit (a) deletes the xfail marker AND (b) applies the kwarg
patch. Reverting P1 restores both the bug AND the xfail marker.

**Files modified.**
```diff
- def __init__(self, weights=None, class_names=None, device="cpu"):
+ def __init__(self, weights=None, class_names=None, device="cpu", num_layers: int = 50):
  ...
-     num_cls=self.num_cls, num_layers=50, url=None)
+     num_cls=self.num_cls, num_layers=num_layers, url=None)
```

**Tests.**
- `test_customweights_num_layers.py` — loads a synthetic ResNet-18 ckpt AND a synthetic ResNet-50 ckpt; both must succeed with `strict=True`.
- Legacy call-signature compatibility (I12): `CustomWeights(weights=w, class_names=c, device=d)` continues to work.

**Acceptance.** `pytest tests/regression/test_customweights_num_layers.py` green; full regression suite green.

**Rollback.** `git revert <PHASE_1_SHA>` restores hard-code; regression fails for ResNet-18 (correct, expected).

**Risks.** R1.1 (4-positional caller exists): `grep "CustomWeights(" src` confirms zero. R1.2 (typo regresses ResNet-50): test exercises both architectures.

---

### 4.2. Phase 2 — Mixup utility port — addresses scope_item:mixup-utility-phase

<a name="mixup-utility-phase"></a>

**Goal.** Port `noisy_same_class_mixup` and
`efficient_batch_same_class_mixup` verbatim from
`Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py` to `src/utils/mixup.py`.
Pure-Python (torch + numpy, already required). **Do not** edit
`src/utils/__init__.py` (no re-export pollution); callers import the
submodule explicitly: `from src.utils.mixup import noisy_same_class_mixup`.

**Files created.**
- `src/utils/mixup.py` — verbatim port with provenance docstring (source repo path + commit SHA + RNG-source note).
- `tests/unit/test_mixup.py` — 6 tests:
  1. Output tensor shape preserved
  2. Label preservation when all-same-class
  3. `λ` clamp to valid range
  4. Singleton class edge case
  5. Noise-type modes (`uniform`, `gaussian`)
  6. Naive vs efficient implementations produce equivalent results (`torch.allclose`, `atol=1e-6`)

Determinism via monkeypatched `np.random.beta` and `torch.randint`.

**Acceptance.** `pytest tests/unit/test_mixup.py` green.

**Rollback.** `git revert <PHASE_2_SHA>`. `python -c "import src.utils.mixup"` fails after revert; no other path affected (no production callers until P4).

**Risks.** Numerical drift across numpy versions (LOW); mixup is opt-in.

---

### 4.3. Phase 3 — Optional-deps wiring + `Custom_Crop` kwarg — addresses scope_item:config-and-docs-phase (partial)

**Goal.** Land the optional-dependencies file so P4 can lazy-import
against it, AND add the defaulted `image_size` kwarg to `Custom_Crop` so
P4 can switch transform resolution per-model.

**Files created.**
- `requirements-dinov3.txt`:
  ```
  transformers>=4.43,<5
  peft>=0.10.0,<0.14.0
  huggingface_hub>=0.20,<1
  ```

**Files modified.**
- `requirements.txt` — single comment line: `# DINOv3 training/inference: pip install -r requirements-dinov3.txt`. No new runtime deps.
- `src/datasets/custom.py` — `Custom_Crop.__init__` (line 240) gains `image_size: int = 224`. Transforms are now built at instance time using that size; module-level `data_transforms = {...}` (lines 33–47) is preserved as a *fallback constant* and is no longer referenced when the kwarg is present. Default 224 reproduces the prior dict op-for-op.

**Tests.**
- `test_no_dinov3_imports_in_default_env.py` (subprocess in `uv`-isolated env): `import src` succeeds with ONLY `requirements.txt` installed.
- `test_custom_crop_default_is_224.py`: no kwarg → val Resize is `(224,224)` AND train RandomResizedCrop size is `(224,224)`.
- `test_dinov3_dataloader_emits_518.py`: `Custom_Crop(image_size=518)` → dataloader emits `(B, 3, 518, 518)`.
- `test_requirements_runtime_lines_hash.py` (I2): hash of non-comment, non-blank `requirements.txt` lines matches pre-change baseline.

**Acceptance.** `pytest tests/regression -x` green; default-env import test green.

**Rollback.** Revert PR #3; `requirements-dinov3.txt` removed; transforms revert to module-level dict.

**Risks.** R3.1 (kwarg breaks existing callers): the kwarg is *defaulted*; all existing call sites are unchanged. Identity-proof test confirms `id(Custom_Crop)` unchanged across revert.

---

### 4.4. Phase 4 — DINOv3 model + linear-probe algorithm (**MVP CORE**) — addresses scope_item:model-and-algorithm-phase

<a name="model-and-algorithm-phase"></a>

**Goal.** Land the frozen-backbone DINOv3 classifier and the linear-probe
algorithm. Wire into the existing `__dict__` registries without forcing
`transformers` on ResNet users.

**Files created.**
- `src/models/dinov3.py` — `DINOv3Classifier(model_name: str, num_classes: int, embedding_dim: Optional[int] = None, freeze_backbone: bool = True)`. Top-level imports: `torch`, `torch.nn` ONLY. `transformers` import lives **inside** `__init__`. Pattern mirrors `speciesnet_base/__init__.py:1–4`.
- `src/algorithms/dinov3_linear.py` — `DINOv3Linear(pl.LightningModule)`. Top-level imports: `torch`, `pytorch_lightning`, `src.utils.mixup`, `src.models` only. `transformers`/`peft`/`huggingface_hub` live in method bodies. `_wrap_with_peft` is a stub (`raise NotImplementedError("PEFT lands in Phase 7")`); `freeze_backbone: False` raises (`NotImplementedError("Full FT lands in Phase 6")`).

**Files modified.**
- `src/models/__init__.py` — append `from .dinov3 import *`. Identity-proof: `models.__dict__['PlainResNetClassifier']` resolves to the same class object pre/post (I7).
- `src/algorithms/__init__.py` — append `from .dinov3_linear import *`. Same identity-proof (I6).

**Tests.**
- `tests/unit/test_dinov3_model.py` — DINOv3Classifier instantiation with `mock_hf_automodel`; embedding_dim inference; freeze flag respected.
- `tests/integration/test_dinov3_linear_probe.py` — DINOv3Linear forward + 1 train_step on synthetic data; backbone params have `requires_grad=False`; classifier params updated.
- `tests/regression/test_custom_crop_default_is_224.py` (covered in §4.3).
- `tests/integration/test_dinov3_dataloader_emits_518.py` (covered in §4.3).

**HF auth handling.** All tests use `mock_hf_automodel` (no network). Real-HF tests marked `@pytest.mark.requires_hf` and gated on `HF_TOKEN`; default `pytest` excludes the marker.

**Acceptance.** `python main.py --config configs/config_dinov3_linear.yaml --dev` succeeds (linear-probe trains; backbone grad-norms == 0). Sample config NOT shipped until P9; P4 uses an inline `--config` in the dev shell or a test fixture YAML.

**Rollback.** `git revert <PHASE_4_SHA>`. `python -c "import src; import src.algorithms; import src.models"` succeeds. Regression suite (I1–I12) green.

**Risks.**
- R4.1 (top-level `transformers` import breaks ResNet users) — HIGH × HIGH ⇒ lazy import inside `__init__`. Invariant I1 enforces via subprocess in `uv`-isolated env.
- R4.2 (same for `dinov3_linear.py`) — same mitigation.
- R4.3 (default `configs/config.yaml` accidentally instantiates DINOv3) — LOW × HIGH ⇒ P4 does NOT touch `configs/config.yaml` (I4).
- R4.4 (HF auth failure on gated model) — wrap `from_pretrained` with try/except `HfHubHTTPError, OSError`; re-raise `RuntimeError("Failed to download {model_name}. If gated: run 'huggingface-cli login'. If offline: set HF_HOME and TRANSFORMERS_OFFLINE=1. Original: {e}")`.
- R4.5 (`hidden_size` missing on stub configs) — raise `ValueError("Could not infer embedding_dim from model {name}; pass embedding_dim=... explicitly.")`.
- R4.6 (`save_hyperparameters` serializes a HF `PretrainedConfig`) — `self.save_hyperparameters(ignore=['conf', 'train_class_counts', 'id_to_labels'])`. Test: `torch.load(ckpt)` succeeds with `transformers` uninstalled.
- R4.7 (518 transform breaks ResNet's 224 path) — RESOLVED via defaulted kwarg in P3.

---

### 4.5. Phase 5 — Loader v1 (MD-Cls `.ckpt` only) — addresses scope_item:inference-loader-phase (v1)

<a name="inference-loader-phase"></a>

**Goal.** Land `src/models/classification/dinov3_base/` with
`DINOv3BaseInference` and `DINOv3CustomWeights`. `_extract_state_dict`
handles only the `"state_dict"` + `net.` prefix flavor (own MD-Cls
ckpts). SINCHI flavors raise `NotImplementedError("SINCHI checkpoint flavours land in Phase 8")`.

**Files created.**
- `src/models/classification/dinov3_base/__init__.py`
- `src/models/classification/dinov3_base/base_classifier.py` — `DINOv3BaseInference(net: nn.Module)`; `_extract_state_dict(ckpt: dict) -> dict` (v1: handles `"state_dict"` + `net.` prefix only).
- `src/models/classification/dinov3_base/custom_weights.py` — `DINOv3CustomWeights(weights: Path, class_names: list[str], device: str = "cpu", model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m")`. Calls `_extract_state_dict`; loads with `strict=True`.
- `docs/dinov3_experimental.md` — minimal (~40 lines): HF auth (`huggingface-cli login`), GPU memory caveat, image_size note, scope statement, license pointer.
- `tests/regression/test_dinov3_absent_actionable_error.py` (I13).

**Files modified.**
- `src/models/classification/__init__.py` — append narrowly-guarded import + package-level `__getattr__`:

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

Identity-proof: all four existing wildcard re-exports unchanged;
`id(classification.CustomWeights)` pre/post matches (I7).

**Tests.**
- `tests/integration/test_dinov3_custom_weights_mdckpt.py` — synthesize an MD-Cls `.ckpt` via P4's `DINOv3Linear.save_checkpoint`; load it via `DINOv3CustomWeights`; assert `missing == [] AND unexpected == []`.
- `tests/unit/test_extract_state_dict_v1.py` — given a hand-rolled ckpt dict with `"state_dict"` and `net.` prefixed keys, `_extract_state_dict` returns the remapped dict.
- `tests/regression/test_dinov3_absent_actionable_error.py` (I13) — subprocess in `uv`-isolated env (only `requirements.txt`): `from src.models import classification` succeeds AND `classification.DINOv3CustomWeights` raises `ImportError` whose message contains `"pip install -r requirements-dinov3.txt"`.

```python
# tests/regression/test_dinov3_absent_actionable_error.py
import subprocess, sys, textwrap

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

def test_absent_dinov3_raises_actionable_import_error():
    result = subprocess.run(
        ["uv", "run", "--isolated", "--with-requirements", "requirements.txt",
         "python", "-c", SCRIPT],
        capture_output=True, text=True, cwd="."
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
```

**Acceptance.** Load own MD-Cls ckpt round-trip green. `ImportError` test green. Existing `CustomWeights` and `SpeciesNetCustomWeights` continue to work unchanged.

**Gate 1.** SHIP after P5 merges. Collect feedback ≥1 week before P6.

**Rollback.** `git revert <PHASE_5_SHA>`. Module shim removed; ResNet path unaffected.

**Risks.**
- R5.1 (top-level wildcard import breaks ResNet users) — narrowed try/except (`if e.name not in {"transformers", "huggingface_hub", "peft"}: raise`). Bugs in our own code propagate.
- R5.2 (corrupt `.ckpt` → unhelpful traceback) — wrap `torch.load` re-raising `RuntimeError(f"Failed to torch.load({path}); valid checkpoint? Original: {e}")`. Test with 16-byte garbage.
- R5.3 (`weights_only=True` future-default in torch ≥ 2.6) — pass `weights_only=False` explicitly; docs note "Only load checkpoints from trusted sources."
- R5.4 (`strict=False` swallows real bugs) — v1 uses `strict=True`. P8 v2 path: count missing/unexpected; raise if `len(missing) > 5` OR any starts with `classifier.` or `feature.` (modulo LoRA exceptions).
- R5.5 (MD-Cls hidden-size vs supplied `model_name` mismatch) — pre-check: `state['classifier.weight'].shape[1] == self.net.embedding_dim`; raise listing both numbers.

---

### 4.6. Phase 6 — Full fine-tune toggle — addresses scope_item:full-finetune-phase

<a name="full-finetune-phase"></a>

**Goal.** Extend `DINOv3Linear` to support `dinov3.freeze_backbone: False`
with two AdamW param groups (feature LR + classifier LR). No new files.

**Files modified.**
- `src/algorithms/dinov3_linear.py`:
  - `configure_optimizers` — when `freeze_backbone=False`, build two param groups: `{params: feature_params, lr: lr_feature}` and `{params: classifier_params, lr: lr_classifier}`. Filter `trainable = [p for p in self.parameters() if p.requires_grad]`.
  - `on_train_start` — log line: `Trainable params: {n} of {total}`.

**Tests.**
- `tests/unit/test_dinov3_finetune_param_groups.py` — assert 2 param groups under `freeze_backbone=False`; assert 1 under `True`; assert no AdamW state allocated for frozen params.
- `tests/integration/test_dinov3_finetune_grad_flow.py` — 1 train_step under `freeze_backbone=False`; assert feature params received non-zero grads; assert classifier params received grads.

**Acceptance.** `python main.py --config <full_ft_config> --dev` runs (trainable_params ≈ 86M for ViT-B). Local GPU run.

**Rollback.** `git revert <PHASE_7_SHA> <PHASE_6_SHA>` (mandatory order — both touch `dinov3_linear.py`). Linear-probe path (P4) remains functional.

**Risks.**
- R6.1 (OOM with default `batch_size: 32`) — sample `configs/config_dinov3_finetune.yaml` ships `batch_size: 4`, `precision: 16-mixed`. Doc table maps model → min measured GPU memory (rows labeled MEASURED or ESTIMATED).
- R6.2 (mid-run freeze toggle inconsistency) — startup-only flag; raise if changed after `setup` finishes.
- R6.4 (non-CUDA execution) — sample comment: `"precision: 16-mixed is CUDA-only; CPU/MPS users set precision: 32."`
- R6.5 (DDP scope) — out of scope for v1; explicit non-goal in `docs/dinov3.md`. Trainer defaults to `strategy="auto"`.

---

### 4.7. Phase 7 — PEFT (LoRA + PiSSA) — addresses scope_item:peft-phase

<a name="peft-phase"></a>

**Goal.** Add `_wrap_with_peft` and `on_save_checkpoint` to
`DINOv3Linear`. `peft` lazy-imported only when
`dinov3.peft.enable: True`. Use `TaskType.FEATURE_EXTRACTION`. Embed
`peft_config_dict` (rank, alpha, dropout, target_modules, init,
task_type) into the Lightning checkpoint under key `dinov3_peft_config`
so P8 can re-wrap before `load_state_dict`.

**Runtime target-module validation (binding directive #2).**
```python
def _wrap_with_peft(self, peft_config):
    from peft import get_peft_model, LoraConfig, TaskType
    suffixes = {n.rsplit(".", 1)[-1]
                for n, _ in self.net.feature.named_modules()}
    requested = set(peft_config.target_modules)
    inter = suffixes & requested
    if not inter:
        raise ValueError(
            f"None of target_modules {sorted(requested)} match any backbone "
            f"module-name suffix. Available suffixes: {sorted(suffixes)}"
        )
    missing = requested - inter
    if missing:
        warnings.warn(
            f"PEFT target_modules not found in backbone: {sorted(missing)}. "
            f"Proceeding with matched targets: {sorted(inter)}",
            UserWarning,
        )
    self.net = get_peft_model(self.net, peft_config)
```

**Files created.**
- `tests/regression/regenerate_peft_disabled_golden.py` — committed regeneration script for the golden behavior snapshot.
- `tests/regression/test_peft_disabled_behavior_snapshot.py` — behavioral regression replacing state-dict hashes:

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
    x = torch.load("tests/fixtures/peft_disabled_input.pt")
    with torch.no_grad():
        logits = model(x)
    golden = torch.load(GOLDEN_PATH)["logits"]
    assert torch.allclose(logits, golden, atol=ATOL)

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
    assert math.isclose(loss.item(), golden_loss, abs_tol=ATOL)
```

**Files modified.**
- `src/algorithms/dinov3_linear.py` — adds `_wrap_with_peft`, `on_save_checkpoint`, `on_load_checkpoint`.

**Tests.**
- `tests/integration/test_dinov3_peft_lora.py` — LoRA-wrapped model, 1 train_step, assert `r` rank, assert only LoRA params have `requires_grad=True` (excluding classifier).
- `tests/integration/test_dinov3_peft_pissa.py` — `init_lora_weights="pissa_niter_4"`; post-wrap assertion that LoRA-A differs from fresh Gaussian by RMS > 1e-4.
- `tests/unit/test_peft_target_modules_validated.py` — three cases: (a) full match silent, (b) empty intersection raises `ValueError`, (c) partial intersection warns `UserWarning` listing unmatched targets.
- `tests/integration/test_peft_checkpoint_roundtrip.py` — save → reload via P5 loader (PEFT path via `dinov3_peft_config` metadata) → assert state-dict shape/keys.
- `tests/regression/test_peft_disabled_behavior_snapshot.py` (above).

**Golden-tensor versioning policy.**
- File: `tests/fixtures/peft_disabled_golden.pt` (dict with keys `logits`, `loss`, `regenerated_at_sha`, `model_arch_version`).
- Regeneration script: `tests/regression/regenerate_peft_disabled_golden.py` (committed; must be invoked explicitly).
- **Frozen** under all changes EXCEPT deliberate changes to `DINOv3Classifier` architecture. When that happens, the same PR must (a) run the regeneration script, (b) commit the new golden, (c) document in `CHANGELOG.md` under "Behavioral baselines updated".
- Comments / formatting / docs changes MUST NOT trigger regeneration.
- `atol=1e-5` is the contract; tightening or loosening requires architect sign-off in the PR.

**Acceptance.** `pytest tests/regression/test_peft_disabled_behavior_snapshot.py tests/unit/test_peft_target_modules_validated.py` green. Local GPU smoke: LoRA + PiSSA train; `trainable_params < 5% total`; PiSSA LoRA-A diverges from gaussian init.

**Rollback.** `git revert <PHASE_7_SHA>`. Linear-probe (P4) and full-FT (P6) unchanged.

**Risks.**
- R7.1 (`peft` API drift) — pin `peft>=0.10.0,<0.14.0`. CI matrix tests floor + ceiling.
- R7.2 (`TaskType.FEATURE_EXTRACTION` correctness) — unit test asserts `self.net.classifier` remains a plain `nn.Linear` (not absorbed by PEFT).
- R7.3 (`target_modules` mismatch) — runtime validation per directive #2 (above).
- R7.4 (`init_lora_weights="pissa_niter_4"` silent fallback on `peft<0.10`) — floor-pin + post-wrap assertion.
- R7.5 (`get_peft_model` changes state-dict key namespace) — embed `peft_config_dict` in checkpoint; P8 re-wraps before load.
- R7.6 (target globs accidentally wrap classifier) — after `get_peft_model`, assert `self.net.classifier.weight.requires_grad == True` AND no `classifier.*` key under `base_model.*` prefix.
- R7.8 (state-dict-hash regression anti-pattern) — RESOLVED via behavior-snapshot test above.

---

### 4.8. Phase 8 — Loader v2 (SINCHI raw + SINCHI PEFT) — addresses scope_item:inference-loader-phase (v2)

**Goal.** Extend `_extract_state_dict` to handle (a) SINCHI raw `.pt`
with `"model_state_dict"` + `backbone.*` → `feature.*` remap; (b) SINCHI
PEFT `.pt` with `backbone.base_model.model.*` keys, requiring either an
embedded `dinov3_peft_config` (own MD-Cls flavor from P7) or an explicit
`peft_config=` kwarg.

**MANDATORY ENTRY GATE.** At least one real SINCHI raw `.pt` and one
real SINCHI PEFT `.pt` are loaded by a human; their key layouts are
recorded in `docs/design/dinov3-integration/sinchi_key_layouts.md` with
provenance (source path, SHA-256, date, reviewer initials). Tests parse
fenced code blocks from that doc and assert in-memory synthetic dicts
use those exact keys. If unobtainable, P8 is deferred; v1 loader (P5)
remains the shipped surface.

**Files created.**
- `docs/design/dinov3-integration/sinchi_key_layouts.md` — evidence file with two fenced code blocks (one per flavor), each listing top-level keys + first 20 sample sub-keys.

**Files modified.**
- `src/models/classification/dinov3_base/base_classifier.py` — extend `_extract_state_dict`:

```python
def _extract_state_dict(ckpt: dict, peft_config: Optional[dict] = None) -> dict:
    # (1) MD-Cls v1 (own)
    if "state_dict" in ckpt and any(k.startswith("net.") for k in ckpt["state_dict"]):
        return {k[len("net."):]: v for k, v in ckpt["state_dict"].items()
                if k.startswith("net.")}
    # (2) SINCHI PEFT raw
    if "model_state_dict" in ckpt and any("base_model.model." in k for k in ckpt["model_state_dict"]):
        embedded = ckpt.get("dinov3_peft_config")
        if embedded is None and peft_config is None:
            raise ValueError(
                "Checkpoint contains LoRA adapter keys but no peft_config was "
                "provided AND no 'dinov3_peft_config' is embedded. "
                "Pass peft_config={r:..., lora_alpha:..., target_modules:[...]} "
                "or use a non-PEFT checkpoint."
            )
        # caller re-wraps with PEFT before load_state_dict
        return _remap_sinchi_peft(ckpt["model_state_dict"], embedded or peft_config)
    # (3) SINCHI raw
    if "model_state_dict" in ckpt and any(k.startswith("backbone.") for k in ckpt["model_state_dict"]):
        return {k.replace("backbone.", "feature.", 1): v
                for k, v in ckpt["model_state_dict"].items()}
    raise ValueError(
        f"Unrecognised checkpoint format. Top-level keys: {list(ckpt.keys())[:10]}"
    )
```

**Tests.**
- `tests/integration/test_dinov3_custom_weights_sinchi_raw.py` — synthetic SINCHI raw `.pt` with key layout matching `sinchi_key_layouts.md` first block; loads via `DINOv3CustomWeights`; assert remap applied.
- `tests/integration/test_dinov3_custom_weights_sinchi_peft.py` — synthetic SINCHI PEFT `.pt` matching second block; loads with `peft_config=...`; assert re-wrap + adapter keys consumed.
- `tests/integration/test_corrupt_ckpt_errors.py` — 16-byte garbage file; assert `RuntimeError` with informative message.
- `tests/regression/test_extract_state_dict_loader_detection.py` — table-driven: 4 fixture dicts (MD-Cls, SINCHI raw, SINCHI PEFT, unrecognized) → expected flavor branch.

**Acceptance.** Load real SINCHI raw + real SINCHI PEFT checkpoints
locally; review `sinchi_key_layouts.md` for completeness. CI runs
synthetic-only tests.

**Gate 2.** SHIP after P8 merges. Collect feedback ≥1 week before P9.

**Rollback.** `git revert <PHASE_8_SHA>`. v1 loader (P5) intact; SINCHI
loaders raise `NotImplementedError` as in P5.

**Risks.**
- R8.1 (auto-detect silent misclassification) — three-way detection by key-prefix inspection with strict ordering; each branch unit-tested; else `ValueError` listing first 10 top-level keys.
- R8.2 (LoRA adapter keys present but no `peft_config`) — hard-raise listing matched adapter keys.
- R8.3 (synthetic-only fixtures hide real format drift) — RESOLVED via mandatory entry gate (real key evidence at tracked path).
- R8.4 (hidden_size mismatch) — pre-check applied to remapped keys; raise listing both numbers.

---

### 4.9. Phase 9 — Config + docs consolidation — addresses scope_item:config-and-docs-phase

<a name="config-and-docs-phase"></a>

**Goal.** Append commented YAML block to `configs/config.yaml`; ship
three sample configs; write `docs/dinov3.md`; append to `docs/usage.md`
and `README.md`; add `CHANGELOG.md` entry. **Minimal**
`docs/dinov3_experimental.md` (~40 lines) ships with P5 so first users
have HF auth + memory guidance before docs exist; P9 is the full
expansion.

**Files created.**
- `configs/config_dinov3_linear.yaml` — `dinov3.model_name`, `image_size: 518`, `peft.enable: False`, `freeze_backbone: True`, `batch_size: 32` (ViT-S default).
- `configs/config_dinov3_finetune.yaml` — `freeze_backbone: False`, `batch_size: 4`, `precision: 16-mixed`, comment about CUDA-only fp16.
- `configs/config_dinov3_lora.yaml` — `peft.enable: True`, `peft.method: lora`, `r: 16`, `lora_alpha: 32`, `target_modules: [q_proj, k_proj, v_proj, o_proj]`.
- `docs/dinov3.md` — full expansion: capability overview, install, HF auth, sample commands, memory table (MEASURED/ESTIMATED), troubleshooting (error taxonomy), reproducibility, license/citation (BibTeX), redistribution policy.
- `CHANGELOG.md` (if absent).

**Files modified.**
- `configs/config.yaml` — **commented append only** (no semantic change):
  ```yaml
  # DINOv3 fine-tuning (optional). Requires: pip install -r requirements-dinov3.txt
  # See configs/config_dinov3_linear.yaml, _finetune.yaml, _lora.yaml.
  ```
- `README.md` — capability bullet pointing to `docs/dinov3.md`.
- `docs/usage.md` — DINOv3 fine-tune + load section with copy-paste commands.

**Tests.**
- `tests/regression/test_default_config_unchanged.py` (I4) — `yaml.safe_load(open('configs/config.yaml'))` returns the pre-change dict (dict equality, not yaml-dump string).
- `tests/docs/test_sample_configs_parse.py` — all three sample configs parse with `yaml.safe_load` and contain required `dinov3.*` keys.

**Acceptance.** All sample configs parse; default config dict unchanged; README/docs reviewed for HF auth callout + GPU memory + PEFT defaults + license + citation.

**Rollback.** `git revert <PHASE_9_SHA>`; configs + docs removed; default `configs/config.yaml` returns to pre-P9 state (comment removed).

**Risks.**
- R9.1 (YAML append breaks parse) — I4 dict-equality test.
- R9.2 (sample config OOMs on small GPU) — ViT-S default; memory table labeled.
- R9.3 (doc example uses gated model without `huggingface-cli login` callout) — every DINOv3 code block prefaced with both `pip install -r requirements-dinov3.txt` and `huggingface-cli login`.
- R9.4 (`requirements.txt` accidentally gains heavy deps) — only comment-line change; I2 hashes runtime-deps lines.
- R9.5 (discoverability) — README + usage + sample-config header all repeat the install + auth callout.

---

<a name="regression-protection"></a>
## 5. Regression protection — addresses scope_item:regression-protection

### 5.1. Repo invariants (I1–I13)

| # | Invariant | Test name | Mechanism |
|---|---|---|---|
| I1 | `python -c "import src"` succeeds in env with ONLY `requirements.txt` installed | `test_default_env_imports_clean` | Subprocess via `uv run --isolated --with-requirements requirements.txt python -c "..."` |
| I2 | `requirements.txt` runtime deps unchanged modulo comments | `test_requirements_runtime_lines_hash` | Hash of non-comment, non-blank lines |
| I3 | ResNet baseline predictions stable (**behavior snapshot**, not state-dict hash) | `test_resnet_baseline_predictions` | argmax labels + confidence within `1e-4` vs `tests/fixtures/baseline_predictions.json`; `seed_everything(42)` + `Trainer(deterministic=True)` |
| I4 | `yaml.safe_load(open('configs/config.yaml'))` returns pre-change dict | `test_default_config_yaml_equal` | **dict equality** (not yaml-dump string) |
| I5 | `CustomWeights(weights=..., class_names=[...])` loads ResNet-50 in env with no DINOv3 deps | `test_customweights_no_dinov3_deps` | Subprocess isolated env |
| I6 | `algorithms.__dict__['Plain']` is the same class object | `test_plain_algorithm_class_identity` | `id()` comparison via subprocess on pre-change SHA |
| I7 | `models.__dict__['PlainResNetClassifier']` is the same class object | `test_plain_resnet_class_identity` | same |
| I8 | `main.py` SHA-256 unchanged | `test_main_unchanged` | Byte-equality (**PERMANENT**) |
| I9 | `src/algorithms/plain.py` SHA-256 unchanged | `test_plain_py_unchanged` | Byte-equality (**PERMANENT**) |
| I10 | `src/models/plain_resnet.py` SHA-256 unchanged | `test_plain_resnet_py_unchanged` | Byte-equality (**PERMANENT**) |
| I11 | `src/datasets/custom.py` SHA-256 unchanged EXCEPT in P3 (defaulted kwarg) where hash updates in same commit + `test_custom_crop_default_is_224` becomes permanent | `test_custom_dataset_byte_or_behaviour` | Hash check + behavior check |
| I12 | `CustomWeights(weights=w, class_names=c, device=d)` legacy 3-arg call works | `test_customweights_legacy_call_signature` | Live import + load |
| I13 | In env without `transformers`, `from src.models import classification` succeeds AND `classification.DINOv3CustomWeights` raises `ImportError` whose message contains substring `"pip install -r requirements-dinov3.txt"` | `test_dinov3_absent_actionable_error` | Subprocess in `uv`-isolated env |

I8–I10 are PERMANENT byte-equality. I11 is byte-equality except for the
one allowed transform-kwarg commit in P3. Everything else is protected
by behavior / dict / import tests, not byte-hashes (directive #6).

### 5.2. Cross-cutting policy

- Every phase MUST run `pytest tests/regression/ -x` before merging.
- `tests/regression/dryrun_revert.sh` (committed in P0):
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
- CI matrix activates **at P3**: job (a) `pip install -r requirements.txt -r requirements-dev.txt`; job (b) `pip install -r requirements.txt -r requirements-dev.txt -r requirements-dinov3.txt`. No `-e .[test]` (directive #5).

---

<a name="test-strategy"></a>
## 6. Test strategy — addresses scope_item:test-strategy

### 6.1. Pyramid

| Layer | Count target | Distribution |
|---|---|---|
| **Unit** (CPU, <1s, no network) | ~18 | Mixup 6, model 4, algorithm 4, `_extract_state_dict` v1+v2 4 |
| **Integration** (CPU, stub backbone, ~10s each) | ~12 | Linear probe 2, full FT 2, LoRA 1, PiSSA 1, loader v1 2, loader v2 SINCHI raw+PEFT 2, transform-518 dataloader 1, PEFT roundtrip 1 |
| **Regression** (invariants + behavior snapshots) | 13 (I1–I13) + 1 (`test_peft_disabled_behavior_snapshot`) | All in P0 except I13 (P5) and PEFT-disabled snapshot (P7) |
| **E2E** (real HF download, GPU) | 4 — manual, never in CI | Linear probe, full FT, LoRA, PiSSA |

### 6.2. Stub strategy (`tests/conftest.py`)

```python
import torch, torch.nn as nn
from types import SimpleNamespace
import pytest

class _StubConfig:
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size

class _StubBackbone(nn.Module):
    """Tiny ViT-shaped stub. Module names mirror HF DINOv3 ViT so PEFT
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

### 6.3. Fixtures (all under `tests/_artifacts/` or `tests/fixtures/` — NO `/tmp`)

- `tests/fixtures/baseline_predictions.json` (P0)
- `tests/fixtures/resnet18_synth.ckpt` (P1)
- `tests/fixtures/peft_disabled_input.pt`, `peft_disabled_target.pt`, `peft_disabled_golden.pt` (P0, consumed by P7)
- `tests/_artifacts/dinov3_linear_mdcls.ckpt` (P4 → P5 → P7)
- `tests/fixtures/sinchi_raw_synth.pt`, `sinchi_peft_synth.pt` (P8)
- `docs/design/dinov3-integration/sinchi_key_layouts.md` (P8, **tracked docs path**, not transient artifacts — directive #4)

### 6.4. HF auth handling

CI never authenticates with HF. All P4–P8 tests use `mock_hf_automodel`.
Real-HF tests marked `@pytest.mark.requires_hf` and gated on `HF_TOKEN`
env; default `pytest` excludes the marker.
`huggingface_hub` references gate via `pytest.importorskip("huggingface_hub")`.

### 6.5. CI vs local

- **CI baseline (P0+)**: unit + integration (mocked HF) + regression.
- **CI matrix (P3+)**: job (a) default-only deps, job (b) default + dinov3 deps. CUDA/HF/network disabled.
- **Local nightly**: `pytest -m requires_hf` and `pytest -m gpu`. Never CI.

### 6.6. Coverage policy

`--cov-fail-under` is **advisory only** (directive #8). CI may emit
coverage reports but does not fail on threshold.

---

<a name="acceptance-and-validation"></a>
## 7. Acceptance and validation — addresses scope_item:acceptance-and-validation

Mapped to research checklist §8. Every command executable; expected
outputs concrete.

| # | Checklist item | Phase | Command | Expected | Tier |
|---|---|---|---|---|---|
| 1 | ResNet path behavior stable | P0 | `pytest tests/regression/test_resnet_baseline_predictions.py -x` | predictions match `baseline_predictions.json` (`atol=1e-4`) | CI |
| 2 | Linear-probe trains | P4 | `python main.py --config configs/config_dinov3_linear.yaml --dev` | ckpt at `weights/Crop/DINOv3Linear/*.ckpt`; backbone grad-norms == 0 | local CPU (mocked HF in CI) |
| 3 | Full-FT trains | P6 | same, `freeze_backbone: False` | grad-norms > 0; `trainable_params ≈ 86M` (ViT-B) | local GPU |
| 4 | Mixup toggle | P4+P6 | `use_noisy_same_class_mixup: True` | loss curves differ from baseline; no shape errors | local CPU |
| 5 | LoRA + PiSSA | P7 | `peft.enable: True` (both methods) | `trainable_params < 5%` total; PiSSA LoRA-A diverges from gaussian init | local GPU |
| 6 | Load own MD-Cls ckpt | P5 | `DINOv3CustomWeights(weights=<ckpt#2>, class_names=[...])` | `missing == [] AND unexpected == []` | CI (synthetic) + local |
| 7 | Load SINCHI raw | P8 | same + real SINCHI raw `.pt` | strict-False load; remap applied | local (real artifact) |
| 8 | Load SINCHI PEFT | P8 | same + `peft_config={...}` + real PEFT `.pt` | re-wrap succeeds; adapter keys consumed | local (real artifact) |
| 9 | CustomWeights regression | P1 | `pytest tests/regression/test_customweights_num_layers.py` | `num_layers=18` AND `=50` both load with `strict=True` | CI |
| 10 | **Default env import + actionable error (directive #7)** | P0/P5 | `uv run --isolated --with-requirements requirements.txt python -c "from src.models import classification; \ntry: classification.DINOv3CustomWeights\nexcept ImportError as e: assert 'pip install -r requirements-dinov3.txt' in str(e); print('OK')\nelse: raise SystemExit(1)"` | exit 0, stdout `"OK"` | CI |
| 11 | Transform sizes | P3/P4 | `pytest tests/regression/test_custom_crop_default_is_224.py tests/integration/test_dinov3_dataloader_emits_518.py` | ResNet `(B,3,224,224)`, DINOv3 `(B,3,518,518)` | CI |
| 12 | Real SINCHI key evidence | P8 | review `docs/design/dinov3-integration/sinchi_key_layouts.md` | file present; lists top-level + sample sub-keys for both flavors; provenance + reviewer initials | manual gate |
| 13 | PEFT `target_modules` validation | P7 | `pytest tests/unit/test_peft_target_modules_validated.py` | empty → `ValueError`, partial → `UserWarning` listing unmatched targets, full → silent | CI |
| 14 | **PEFT-disabled behavior snapshot (directive #6)** | P7 | `pytest tests/regression/test_peft_disabled_behavior_snapshot.py` | `allclose(logits, golden, atol=1e-5)` AND `isclose(loss, golden_loss, abs_tol=1e-5)` | CI |

### 7.1. Definition of done
- All 13 invariants + all 14 checklist tests pass.
- CI green on both matrix jobs.
- `docs/dinov3.md` reviewed for HF auth / GPU memory / PEFT defaults / license / citation.
- `sinchi_key_layouts.md` populated (or P8 explicitly deferred with documented "experimental" status).

---

<a name="risk-and-rollback"></a>
## 8. Risk and rollback — addresses scope_item:risk-and-rollback

### 8.1. Top-level risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A phase breaks I1 (default-env import) | MEDIUM | CATASTROPHIC | Per-phase regression gate; lazy imports; narrowed try/except; subprocess-isolated env tests (I1, I5, I13). |
| R2 | DINOv3 HF model gated → cryptic auth error | HIGH | MEDIUM | Wrap `from_pretrained` with actionable `RuntimeError`; docs include `huggingface-cli login`; CI never authenticates. |
| R3 | GPU OOM on full-FT default | HIGH | MEDIUM | Sample ships ViT-S + `batch_size: 4` + `precision: 16-mixed`; memory table labels MEASURED vs ESTIMATED. |
| R4 | `peft` API drift | MEDIUM | MEDIUM | Pin `peft>=0.10.0,<0.14.0`; CI matrix tests both bounds. |
| R5 | Loader auto-detect silently misclassifies | MEDIUM | HIGH | Key-prefix detection with strict ordering; fail-loud on missing classifier keys; hidden-size pre-check; LoRA-without-`peft_config` hard raises. |
| R6 | Reverting P6 with P7 present | LOW (with DAG) | MEDIUM | Reverse-DAG enforced; `dryrun_revert.sh` refuses out-of-order; documented order P9→P0. |
| R7 | Lightning checkpoint format drift | LOW | LOW | Pin `pytorch_lightning` floor in `requirements-dinov3.txt`. |
| R8 | Reproducibility | MEDIUM | MEDIUM | `seed_everything` + `Trainer(deterministic=True)`; embedded `dinov3_peft_config` + `model_name` + `image_size` in checkpoint metadata. |
| R9 | DINOv3 license/redistribution | LEGAL | LEGAL | P9 docs; no automatic download; checkpoint metadata records source `model_name`. |
| R10 | Non-CUDA execution | MEDIUM | LOW | Sample configs use `precision: 16-mixed` with "CUDA-only" comment; default fallback `precision: 32`. |
| R11 | **Golden-tensor drift on accidental architecture changes** | LOW | MEDIUM | Frozen golden tensor; regeneration requires explicit script invocation + `CHANGELOG` entry; failing snapshot test blocks PR. |
| R12 | Real SINCHI ckpts unobtainable → P8 blocks indefinitely | MEDIUM | LOW | P8 explicitly deferrable; v1 loader (P5) is the shipped surface; documented "experimental" status. |

### 8.2. Per-phase rollback DAG (reverse order)

```
P9 → revert → pytest tests/regression -x                                                                    → green
P8 → revert → pytest tests/regression tests/integration/test_dinov3_custom_weights_mdckpt.py -x             → green
P7 → revert → pytest tests/regression tests/integration/test_dinov3_finetune_grad_flow.py -x                → green
P6 → revert → pytest tests/regression tests/integration/test_dinov3_linear_probe.py -x                      → green
P5 → revert → pytest tests/regression tests/integration/test_dinov3_linear_probe.py -x                      → green
P4 → revert → pytest tests/regression tests/unit/test_mixup.py -x                                           → green
P3 → revert → pytest tests/regression -x                                                                    → green
P2 → revert → pytest tests/regression -x                                                                    → green
P1 → revert → pytest tests/regression -x → test_resnet18_loads RED (expected); test_resnet50 GREEN
P0 → revert → manual confirmation; baseline removed
```

### 8.3. Single-command emergency revert

```bash
git revert --no-edit \
  $PHASE_9_SHA $PHASE_8_SHA $PHASE_7_SHA $PHASE_6_SHA \
  $PHASE_5_SHA $PHASE_4_SHA $PHASE_3_SHA $PHASE_2_SHA $PHASE_1_SHA
# P0 (tests) stays: better tests, zero DINOv3 code
pytest tests/regression -x
```

---

## 9. Out of scope (explicit non-goals for v1)

- Multi-GPU DDP for DINOv3 fine-tuning. `Trainer(strategy="auto")` only.
- AdaLoRA, IA3, prefix-tuning, or other PEFT methods beyond LoRA + PiSSA.
- Packaging migration (`pyproject.toml [project]` block, `pip install -e .`).
- Auto-download of DINOv3 weights via the CLI. User must run `huggingface-cli login` and/or `huggingface_hub.snapshot_download`.
- Migration of MegaDetector detector code; this plan touches only the classifier pipeline.

---

## 10. Design evolution

| Round | Architects' `NEW` totals | Disagreements | Inquisitor verdict |
|---|---|---|---|
| 1 | (initial) | many | `NEEDS-MORE` with 47 NEW critiques |
| 2 | 25 + 17 + 11 | 3 + 3 + 3 (9 total) | `NEEDS-MORE` with 7 NEW + 8 final-synthesis directives + 9 adjudications (8 MISTAKEN-BY, 1 RESOLVED-TOWARD=architect-3, 0 DEADLOCKED) |
| 3 | 5 + 5 + 4 (all CLOSURE) | 0 + 0 + 0 | `CONVERGED SCOPE_CHECK=PASS COVERED=12/12 NEW=0` |

### Key resolutions

1. **Transform mechanism** (round 2 → 3): architect-3's defaulted `image_size=224` kwarg on the existing `Custom_Crop` class won over architect-1's separate `build_dinov3_eval_transform` shim. Single transform entry point; no shim.
2. **Absent-DINOv3 surface** (A3-N1): unified to `ImportError` with the substring `"pip install -r requirements-dinov3.txt"` via package-level `__getattr__`. All four document references aligned.
3. **Regression gate** (A3-N2): state-dict hashes removed as pass/fail gates; replaced with behavioral snapshots (logits/loss within tolerance).
4. **SINCHI evidence location** (A2-N1): moved from `tests/_artifacts/` to `docs/design/dinov3-integration/sinchi_key_layouts.md` (tracked docs path).
5. **Packaging** (A2-N2): no `pyproject.toml [project]` block; requirements-only installs.
6. **Dependency set** (A1-N1): `requirements-dinov3.txt` lists `transformers` + `peft` + `huggingface_hub` explicitly; absent-env tests cover all three.
7. **Rollback script** (A1-N3): repo-local `mktemp -d` worktree with `trap` cleanup; no fixed sibling-path collision risk.

### Unresolved

None. All 9 round-2 disagreements were resolved (8 MISTAKEN-BY, 1
RESOLVED-TOWARD). The round-3 inquisitor reports 0 NEW critiques and 0
unresolved disagreements; convergence is genuine, not surface-deep.

---

## 11. Appendix — file-tree summary (post-P9)

```
MegaDetector-Classifier/
├── configs/
│   ├── config.yaml                            (commented append in P9)
│   ├── config_dinov3_linear.yaml              (NEW, P9)
│   ├── config_dinov3_finetune.yaml            (NEW, P9)
│   └── config_dinov3_lora.yaml                (NEW, P9)
├── docs/
│   ├── dinov3.md                              (NEW, P9 — full)
│   ├── dinov3_experimental.md                 (NEW, P5 — minimal stub)
│   ├── usage.md                               (modified, P9)
│   └── design/dinov3-integration/
│       ├── final.md                           (this file)
│       ├── sinchi_key_layouts.md              (NEW, P8 — evidence)
│       └── round_{01,02,03}/                  (design artifacts)
├── requirements.txt                           (1 comment line in P3)
├── requirements-dev.txt                       (unchanged or NEW if absent)
├── requirements-dinov3.txt                    (NEW, P3)
├── pyproject.toml                             (NEW in P0, ONLY [tool.pytest.ini_options])
├── scripts/
│   └── rollback_dry_run.sh                    (NEW, P0)
├── src/
│   ├── algorithms/
│   │   ├── __init__.py                        (+1 line in P4)
│   │   └── dinov3_linear.py                   (NEW, P4; extended P6, P7)
│   ├── datasets/
│   │   └── custom.py                          (+1 defaulted kwarg in P3)
│   ├── models/
│   │   ├── __init__.py                        (+1 line in P4)
│   │   ├── dinov3.py                          (NEW, P4)
│   │   └── classification/
│   │       ├── __init__.py                    (+1 guarded import + __getattr__ in P5)
│   │       ├── resnet_base/custom_weights.py  (+1 defaulted kwarg in P1)
│   │       └── dinov3_base/                   (NEW, P5; extended P8)
│   │           ├── __init__.py
│   │           ├── base_classifier.py
│   │           └── custom_weights.py
│   └── utils/
│       └── mixup.py                           (NEW, P2)
├── tests/
│   ├── conftest.py                            (NEW, P0)
│   ├── fixtures/                              (P0+)
│   ├── _artifacts/                            (P4+)
│   ├── unit/                                  (P2+)
│   ├── integration/                           (P4+)
│   ├── regression/                            (P0+)
│   │   └── dryrun_revert.sh
│   └── docs/                                  (P9)
├── README.md                                  (modified, P9)
└── CHANGELOG.md                               (NEW or modified at P1, P5, P7, P9)
```

---

## 12. References

- Research report: `/shared/home/aahernandez/.copilot/session-state/fa5699aa-9c12-4f4d-9834-b337cda7f3d5/research/what-are-the-steps-necessary-to-allow-the-repo-on-.md`
- Round 1 proposals + inquisitor: `round_01/`
- Round 2 proposals + inquisitor: `round_02/`
- Round 3 proposals + inquisitor: `round_03/`
- Source SINCHI repo (reference for mixup + algorithm structure): `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/`
- Scope ledger: `SCOPE_LEDGER.json`
- Coverage log: `COVERAGE_LOG.jsonl`
