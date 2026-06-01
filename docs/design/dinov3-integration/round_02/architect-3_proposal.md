# DINOv3 integration — phased plan (architect-3: risk / backwards-compat / additive-purity, round 2)

> Perspective unchanged from round 1: assume every change can break someone
> mid-experiment. Every byte of new code is opt-in. The load-bearing
> invariant remains: *unchanged default `configs/config.yaml` + unchanged
> `pip install -r requirements.txt` + same seed/data ⇒ bit-identical
> training, checkpoints, and predictions to the pre-change repo.*

> Round-2 evolution focus (from inquisitor F-21..F-31, F-32..F-39, F-40..F-47
> as applicable): (a) loader split into v1 / v2 (adopt architect-1), (b)
> dependency policy made concrete and singular, (c) byte-equality scoped to
> the four files where it is non-fragile, (d) transform injection
> point resolved, (e) SINCHI real-checkpoint gate made mandatory, (f) PEFT
> target-module + TaskType claims runtime-validated, (g) rollback dry-run
> operationalised as a concrete script, (h) try/except guards narrowed.

---

## Executive overview

The 12 ledger scope items are realized as **9 phases** (round 1 had 8 — the
inference loader is now split into v1 / v2, per architect-1 F-27 / F-33). The
phase order is now: scaffolding → bug-fix → mixup → optional-deps →
model+linear-probe → **loader-v1 (MD-Cls .ckpt only)** → full-FT → PEFT →
**loader-v2 (SINCHI raw + SINCHI PEFT)** → config+docs. A minimal "DINOv3
linear-probe + own-checkpoint inference" capability ships at end of phase 5
without ever touching `peft` or SINCHI binaries — that is the partial-rollout
target.

The single dependency policy adopted (F-21, F-26, F-31, F-34) is:
**a separate `requirements-dinov3.txt` file**, with `transformers`, `peft`,
and `huggingface_hub` installed only by that file, AND lazy import at every
module top so that `pip install -r requirements.txt` alone keeps every
import path green. Justification: the repo has no `pyproject.toml` (verified
— only `requirements.txt` and `environment.yaml` exist); introducing PEP 508
extras would require standing up a build system in the same change, which
violates additive-purity. Every doc command that names a DINOv3 capability
explicitly prefaces with `pip install -r requirements-dinov3.txt` (closes
F-31).

Byte-equality regression invariants are restricted to **four files**:
`main.py`, `src/algorithms/plain.py`, `src/models/plain_resnet.py`,
`src/datasets/custom.py` (closes F-24). Other files are protected by
behaviour / import / config-dict tests, which survive whitespace and
comment changes.

The transform-size question (F-30, F-37) is resolved at the
**dataset-config** layer: a new optional `image_size: int = 224` kwarg on the
existing `Classification_Inference_Transform` constructor, defaulted to 224
so the ResNet path is bit-identical. DINOv3 configs set `image_size: 518`.
Test gate: ResNet dataloader emits `(3,224,224)`; DINOv3 dataloader emits
`(3,518,518)`. The change is a defaulted kwarg (additive).

Phase boundaries are now **reverse-DAG revertable**, not
"independently revertable" (closes F-22). The rollback recipe enumerates
the exact `git revert` order and the verification command per step.

---

## Phase index (table)

| # | Phase | Touches existing code? | New runtime deps | Rollback risk | DAG predecessor |
|---|---|---|---|---|---|
| 0 | Test scaffolding + regression baseline | NO (tests/ only; conftest.py creates `tests/_artifacts/`) | none | trivial | — |
| 1 | CustomWeights `num_layers` bug fix | **YES** (1 file, +1 defaulted kwarg) | none | low | 0 |
| 2 | Mixup utility port | NO | none | trivial | 0 |
| 3 | Optional-deps wiring (`requirements-dinov3.txt`) | NO | none in default; transformers/huggingface_hub via dinov3 file | trivial | 0 |
| 4 | Model + linear-probe algorithm + 518-transform kwarg | NO (3 new files + 2 `__init__.py` re-export lines + 1 defaulted kwarg on existing transform) | none in default; transformers lazy | low | 2, 3 |
| 5 | Loader-v1: `DINOv3CustomWeights` for **MD-Cls `.ckpt` only** | NO (new subpackage, narrow `_extract_state_dict`) | none in default | low | 4 |
| 6 | Full fine-tune toggle | NO (extends Phase 4 algorithm) | none additional | low | 4 |
| 7 | PEFT (LoRA / PiSSA) toggle | NO (extends Phase 4 algorithm + Phase 4 hparam save) | `peft` in dinov3 reqs | medium | 6 |
| 8 | Loader-v2: extend `_extract_state_dict` for SINCHI raw + SINCHI PEFT | NO (extends Phase 5 file) | none additional | **medium-high** (real-SINCHI gate) | 5, 7 |
| 9 | Config + docs | YES — comment-only append to `configs/config.yaml`; new docs files | none | trivial | 4, 5, 8 |

Total: 10 commits (0–9). Phases 4–8 are pure adds. Phase 1 and the
comment-only append in 9 are the only edits to already-shipped files; both
back-compat by construction.

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

### Ordering rationale (risk lens, evolved)

1. **Phase 0 first** — without a baseline `python -c "import src"` test and a
   default-config snapshot, "backwards-compatible" is unfalsifiable.
2. **Phase 1 (bug fix) second** — adopted from architect-1 (consensus point);
   it is the *only* edit to behaviour-relevant existing code. Landing it
   alone preserves bisect-ability and demonstrates the Phase-0 regression
   harness on a one-line surface. The pre-fix failing test artifact is
   committed in **Phase 0** as `tests/regression/test_resnet18_loads.py`
   marked `@pytest.mark.xfail(strict=True, reason="bug fixed in Phase 1")`;
   Phase 1's commit flips the marker off (closes F-20 from architect-2 by
   adoption, satisfies F-32).
3. **Phase 2 (mixup)** — pure leaf utility, no callers, smallest test set
   exercising the new tests/ infra under torch/numpy load.
4. **Phase 3 (optional-deps file) before Phase 4** — install target must
   exist before docs and CI matrix point at it.
5. **Phase 4 (model + linear-probe + transform kwarg)** — landed as one
   commit because the 518 transform kwarg only makes sense once a
   DINOv3 algorithm exists to consume it. The default-224 path keeps
   ResNet bit-identical.
6. **Phase 5 (loader-v1 MD-Cls only)** — adopted from architect-1
   (F-27, F-33). Ship the minimal train→save→load→infer loop **before**
   adding full-FT and PEFT, so the MVP is usable without `peft` or SINCHI
   binaries. The loader's three-format auto-detect is *not* in Phase 5;
   v1 raises `NotImplementedError` for the two SINCHI flavours, surfacing
   the boundary explicitly.
7. **Phase 6 (full-FT) → Phase 7 (PEFT)** — in increasing-risk order;
   PEFT depends on Phase 4 in code but is sequenced after 6 so each
   added gradient regime is testable in isolation.
8. **Phase 8 (loader-v2 SINCHI)** — last code phase. Gated on a **real
   SINCHI checkpoint key-inspection step** (closes F-29, F-38). Until at
   least one real raw `.pt` and one real PEFT `.pt` have been `torch.load`-ed
   and their `keys()` recorded in `docs/design/dinov3-integration/sinchi_key_layouts.md`,
   Phase 8 does NOT merge. If real binaries are unavailable, Phase 8 is
   deferred and the loader keeps the v1 surface — a strict partial-rollout
   that is still a net capability gain.
9. **Phase 9 (config + docs)** — last, so docs match shipped behaviour.

### Entry/exit criteria (per phase, unified template)

- **Entry**: (a) predecessor phases merged on `main`; (b) `pytest
  tests/regression -x` green on `main`; (c) phase-specific risk register
  re-read and any new mitigations folded in; (d) rollback SHA recorded.
- **Exit**: (a) phase-specific tests green; (b) regression suite still
  green; (c) **rollback dry-run executed**: `bash
  tests/regression/dryrun_revert.sh <PHASE_SHA>` runs `git worktree add`
  to a sibling dir, applies `git revert --no-edit <PHASE_SHA>`, runs
  `pytest tests/regression -x`, asserts exit 0, removes the worktree
  (closes F-25 — script is checked in during Phase 0, see Test strategy);
  (d) phase entry appended to `COVERAGE_LOG.jsonl`.

### Per-phase deliverables summary

| Phase | New files | Modified files | Test files added |
|---|---|---|---|
| 0 | `tests/conftest.py`, `tests/regression/test_invariants.py`, `tests/regression/dryrun_revert.sh`, `tests/fixtures/baseline_predictions.json` | none | this IS the scaffold |
| 1 | none | `src/models/classification/resnet_base/custom_weights.py` (+1 defaulted kwarg) | `tests/regression/test_customweights_num_layers.py` (xfail flip) |
| 2 | `src/utils/mixup.py` | none | `tests/unit/test_mixup.py` |
| 3 | `requirements-dinov3.txt` | `requirements.txt` (+1 comment line — see Phase 3 §Files MODIFIED), `.github/workflows/*.yml` (only if Phase 0 added one; otherwise N/A) | `tests/regression/test_no_dinov3_imports_in_default_env.py` |
| 4 | `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py` | `src/models/__init__.py` (+1 line), `src/algorithms/__init__.py` (+1 line), `src/datasets/custom.py` (+1 defaulted kwarg on `Classification_Inference_Transform.__init__`: `image_size: int = 224`) | `tests/unit/test_dinov3_model.py`, `tests/integration/test_dinov3_linear_probe.py`, `tests/regression/test_transform_default_is_224.py`, `tests/integration/test_dinov3_dataloader_emits_518.py` |
| 5 | `src/models/classification/dinov3_base/{__init__.py, base_classifier.py, custom_weights.py}` | `src/models/classification/__init__.py` (+1 narrowly-guarded line, see §Inference loader) | `tests/integration/test_dinov3_custom_weights_mdckpt.py`, `tests/unit/test_extract_state_dict_v1.py` |
| 6 | none | `src/algorithms/dinov3_linear.py` (extends `configure_optimizers`) | `tests/unit/test_dinov3_finetune_param_groups.py`, `tests/integration/test_dinov3_finetune_grad_flow.py` |
| 7 | none | `src/algorithms/dinov3_linear.py` (adds `_wrap_with_peft`, `on_save_checkpoint`) | `tests/integration/test_dinov3_peft_lora.py`, `tests/integration/test_dinov3_peft_pissa.py`, `tests/unit/test_peft_target_modules_validated.py` |
| 8 | none | `src/models/classification/dinov3_base/base_classifier.py` (extends `_extract_state_dict`) | `tests/integration/test_dinov3_custom_weights_sinchi_raw.py`, `tests/integration/test_dinov3_custom_weights_sinchi_peft.py`, **plus `docs/design/dinov3-integration/sinchi_key_layouts.md` (evidence file, not a test)** |
| 9 | `configs/config_dinov3_linear.yaml`, `configs/config_dinov3_finetune.yaml`, `configs/config_dinov3_lora.yaml`, `docs/dinov3.md` | `configs/config.yaml` (append commented block only), `README.md` (append capability bullet), `docs/usage.md` (append section), `CHANGELOG.md` | `tests/regression/test_default_config_unchanged.py`, `tests/docs/test_sample_configs_parse.py` |

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

### Goal
Port `noisy_same_class_mixup` and `efficient_batch_same_class_mixup` verbatim
from `Andres_SSD_1/SINCHI_DINOV3/utils/mixup.py` to `src/utils/mixup.py`.
Pure-Python (torch + numpy, already required). No `__init__.py` edits.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 2.1 | Re-export pollution via `src/utils/__init__.py` | LOW | LOW | Do not edit `src/utils/__init__.py`. Callers import explicitly. Confirmed by `grep "from src.utils" src` style elsewhere in repo. |
| 2.2 | Numerical drift from a different numpy version vs SINCHI's pin | LOW | LOW | Mixup is opt-in. Test asserts deterministic-given-seed, not bit-equal-across-environments. |
| 2.3 | Tests use `/tmp` for golden files (forbidden in this environment) | (would-be) LOW | LOW | All test fixtures live under `tests/_artifacts/` per Phase 0 policy (closes architect-2 F-14 by adoption). |

### Files MODIFIED vs CREATED
- **CREATED**: `src/utils/mixup.py`, `tests/unit/test_mixup.py`.
- **MODIFIED**: none.

### Dependency footprint added
None.

### Rollback
`git revert <PHASE_2_SHA>` → file deleted. `python -c "import src.algorithms,
src.models, src.datasets, src.utils"` succeeds.

### DAG
Depends only on Phase 0.

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm (linear probe) — addresses scope_item:model-and-algorithm-phase

### Goal
Land `src/models/dinov3.py` (`DINOv3Classifier`) and
`src/algorithms/dinov3_linear.py` (`DINOv3Linear`, frozen-backbone path
only). Wire into the `__dict__` registries without forcing import on
ResNet users. **Also**: add a defaulted `image_size: int = 224` kwarg to
`Classification_Inference_Transform.__init__` in `src/datasets/custom.py`
so the DINOv3 dataloader can request 518 without affecting the ResNet
default (resolves F-30, F-37).

### Risk register (evolved)
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 4.1 | `from .dinov3 import *` at top of `src/models/__init__.py` triggers `transformers` import for every ResNet user → `ImportError` on `import src` | HIGH (if naive) | HIGH | `src/models/dinov3.py` MUST NOT import `transformers` at module top. `from transformers import AutoModel` lives **inside** `DINOv3Classifier.__init__`. The module-top imports are only `torch`, `torch.nn`. Invariant I1 (Phase 0) enforces in an env without `transformers` installed. Mirrors the verified pattern in `speciesnet_base/__init__.py:1–4`. |
| 4.2 | Same for `dinov3_linear.py` | HIGH (if naive) | HIGH | Module-top imports: `torch`, `pytorch_lightning`, `src.utils.mixup`, `src.models` only. `transformers`/`peft`/`huggingface_hub` live inside method bodies. |
| 4.3 | Default `configs/config.yaml` accidentally triggers DINOv3 instantiation | LOW | HIGH | Phase 4 does NOT modify `configs/config.yaml`. Invariant I4 enforces. |
| 4.4 | HF auth failure (`AutoModel.from_pretrained` on gated model) crashes mid-training with cryptic error | MEDIUM | MEDIUM | Wrap the HF call in `__init__` with try/except catching `HfHubHTTPError` and `OSError`, re-raising `RuntimeError` with explicit text: `"Failed to download {model_name}. If gated: run 'huggingface-cli login'. If offline: set HF_HOME and TRANSFORMERS_OFFLINE=1. Original: {e}"`. Tests **never** invoke real `from_pretrained` — they use the `mock_hf_automodel` fixture (Phase 0). |
| 4.5 | `hidden_size` missing on stub configs in user-extended use | LOW | HIGH | Raise `ValueError("Could not infer embedding_dim from model {name}; pass embedding_dim=... explicitly.")`. Unit test included. |
| 4.6 | `save_hyperparameters` serialises a HF `PretrainedConfig` into the ckpt, creating a transformers version-lock | MEDIUM | MEDIUM | `self.save_hyperparameters(ignore=['conf', 'train_class_counts', 'id_to_labels'])`. Only the string `model_name` is saved. Test: `torch.load(ckpt)` succeeds with `transformers` uninstalled. |
| 4.7 | (resolved) 518 transform breaking ResNet's 224 path | (was HIGH) | (was MEDIUM) | **RESOLVED.** Mechanism: `Classification_Inference_Transform.__init__(image_size: int = 224)` defaulted kwarg; `transforms.Resize((image_size, image_size), interpolation=BICUBIC)`. ResNet config consumes default (224, behaviour bit-identical). DINOv3 config in Phase 9 sets 518. Test pair: `test_transform_default_is_224` (no kwarg ⇒ Resize.size == (224,224)) AND `test_dinov3_dataloader_emits_518` (asserts batch tensor shape `(B,3,518,518)` after wiring the kwarg into the DINOv3 sample config). |
| 4.8 | HuggingFace DINOv3 weights are **gated and carry license obligations** (closes F-40) | N/A | LEGAL | Phase 9 `docs/dinov3.md` documents: (a) HF access request URL, (b) the DINOv3 license terms link, (c) required citation (Oquab et al. 2024), (d) "checkpoints derived from DINOv3 inherit the license — do not redistribute." Phase 4 introduces no automatic download. |

### Files MODIFIED vs CREATED
- **CREATED**: `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py`.
- **MODIFIED**:
  - `src/models/__init__.py`: append `from .dinov3 import *`. **Semantic-identity proof**: existing `models.__dict__['PlainResNetClassifier']` resolves to the same class object pre/post (regression I7).
  - `src/algorithms/__init__.py`: append `from .dinov3_linear import *`. Same proof (I6).
  - `src/datasets/custom.py`: add `image_size: int = 224` kwarg to `Classification_Inference_Transform.__init__`, forward to `transforms.Resize`. **Semantic-identity proof**: callers passing 0–N args (where N is the current arity) get the kwarg defaulted to 224, transformer pipeline is bit-identical for ResNet. Test `test_transform_default_is_224` enforces.

### Dependency footprint added
- Module-top: none.
- Lazy: `transformers` (on `DINOv3Classifier.__init__`).

### Rollback
```
git revert <PHASE_4_SHA>
python -c "import src; import src.algorithms; import src.models"
pytest tests/regression -x        # I1..I12 green
```

### DAG
Depends on Phases 2 (mixup) and 3 (optional-deps file).

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

### Goal
Extend `DINOv3Linear` to support `dinov3.freeze_backbone: False`, emit two
AdamW param groups (feature LR + classifier LR). No new files.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 6.1 | OOM with default `batch_size: 32` on <24GB GPU | HIGH | HIGH | Sample `configs/config_dinov3_finetune.yaml` ships `batch_size: 4, precision: 16-mixed`. Doc table maps model size → minimum measured GPU memory. **Numbers are flagged as ESTIMATES until a smoke benchmark runs** (closes F-28); the doc table includes a "measured / estimate" column. |
| 6.2 | Mid-run freeze toggle inconsistency | LOW | LOW | `freeze_backbone` is a startup-only flag; raise if changed after `setup` finishes. |
| 6.3 | AdamW state allocated for `requires_grad=False` params | LOW | LOW | `trainable = [p for p in self.parameters() if p.requires_grad]` filter. Unit test asserts param-group counts under both flags. |
| 6.4 | Non-CUDA execution (closes F-42) | MEDIUM | LOW | `precision: 16-mixed` is documented as "CUDA-only; on CPU/MPS use `precision: 32`." Sample config has a comment. `DINOv3Linear.configure_optimizers` does not assume CUDA. CI runs on CPU only. |
| 6.5 | Distributed/DDP (closes F-43, partial) | MEDIUM | LOW | Out of scope for v1 — explicit non-goal documented in `docs/dinov3.md`. The Lightning Trainer is left at its default `strategy="auto"`. Follow-up issue tracked. |

### Files MODIFIED vs CREATED
- **CREATED**: none.
- **MODIFIED**: `src/algorithms/dinov3_linear.py` (extends `configure_optimizers`, adds `on_train_start` `trainable_params` log line).
- **NOTE on revertability (F-22)**: Phase 6 modifies the same file as Phases 4 and 7. The plan does NOT claim Phase 6 is independently revertable; it claims **reverse-DAG revertable**: 9 → 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0. Reverting 6 with 7 present leaves an unsatisfied `_wrap_with_peft` symbol — the rollback script (Phase 0) refuses to revert out-of-order and prints the required predecessor.

### Dependency footprint added
None.

### Rollback
`git revert <PHASE_7_SHA> <PHASE_6_SHA>` (mandatory order). Linear-probe
path (Phase 4) remains functional; full-FT configs error with KeyError.

### DAG
Depends on Phase 4.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA/PiSSA) — addresses scope_item:peft-phase

### Goal
Add `_wrap_with_peft` and `on_save_checkpoint` to `DINOv3Linear`. `peft`
lazy-imported only when `dinov3.peft.enable: True`.

### Risk register (evolved)
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 7.1 | `peft` API drift between releases | MEDIUM | MEDIUM | Pin `peft>=0.10.0,<0.14.0` in `requirements-dinov3.txt`. Wrap call in try/except producing actionable upgrade text. CI matrix tests floor + ceiling (closes F-26 via staged adoption: matrix activates only at Phase 7, not Phase 0). |
| 7.2 | `TaskType.FEATURE_EXTRACTION` may be wrong for "backbone + separate head" pattern | LOW | MEDIUM | **Verified per inquisitor industry-comparison F-39 / consensus point**: HF PEFT docs (https://huggingface.co/docs/peft/en/package_reference/peft_types) define `FEATURE_EXTRACTION` as "provides hidden states used as embeddings or features for downstream tasks" — exactly our setup (backbone → CLS → external classifier). Use `TaskType.FEATURE_EXTRACTION`. Unit test asserts wrap succeeds and the external `self.net.classifier` remains a plain `nn.Linear` (not absorbed into PEFT). |
| 7.3 | `target_modules=[q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj]` may not match the loaded backbone's actual module names | **HIGH** | HIGH | Per inquisitor F-05 / industry-comparison: current HF `DINOv3ViT` defines split `q_proj/k_proj/v_proj/o_proj` and `up_proj/down_proj`; `gate_proj` exists only in `DINOv3ViTGatedMLP` variants. **Runtime validation**: after instantiating `DINOv3Classifier`, `_wrap_with_peft` calls `_validate_target_modules(net.feature, target_modules)` which iterates `net.feature.named_modules()`, collects every leaf name suffix, computes the intersection with `target_modules`, and **raises** `ValueError` if the intersection is empty, listing the actual suffixes seen so the user can fix their config. The non-empty intersection (some modules matched, some not) emits a warning with the list of unmatched names. Unit test `test_peft_target_modules_validated`: (a) matching list → silent; (b) all-mismatched list → ValueError with helpful text; (c) partial → warning. |
| 7.4 | `init_lora_weights="pissa_niter_4"` silently falls back on `peft<0.10` | MEDIUM | HIGH (silent quality regression) | Floor-pin `peft>=0.10.0` + post-wrap assertion that LoRA A-matrix differs from a fresh gaussian init by > 1e-4 RMS. |
| 7.5 | `get_peft_model` changes state-dict key namespace; inference loader must handle it | HIGH | HIGH | (a) Phase 7 **embeds** `peft_config_dict` (rank, alpha, dropout, target_modules, init, task_type) into the Lightning checkpoint via `on_save_checkpoint` under the key `dinov3_peft_config` (closes F-06 / F-41). (b) Phase 8 loader-v2 reads this key and re-wraps before `load_state_dict`. (c) Round-trip test landed **in this phase** (save → re-instantiate via embedded config → load → forward → match logits). |
| 7.6 | Target globs accidentally wrap the classifier | LOW | HIGH | After `get_peft_model`, assert `self.net.classifier.weight.requires_grad == True` and that no `classifier.*` key appears under the `base_model.*` PEFT prefix. |
| 7.7 | Missing `peft` install when user enables PEFT | MEDIUM | LOW | Lazy import in `_wrap_with_peft`, called from `__init__` — error fires before `Trainer.fit`. Message includes `pip install -r requirements-dinov3.txt`. |

### Files MODIFIED vs CREATED
- **CREATED**: none.
- **MODIFIED**: `src/algorithms/dinov3_linear.py`. Semantic-identity for linear-probe and full-FT: both branches gated on `if self.use_peft:` — False ⇒ existing behaviour byte-identical (verified by integration test: train one epoch with `peft.enable: False` before and after the commit, hash the resulting `state_dict`, compare).

### Dependency footprint added
- Module-top: none.
- Lazy: `peft`.
- `requirements-dinov3.txt`: gains `peft>=0.10.0,<0.14.0`.

### Rollback
`git revert <PHASE_7_SHA>` — linear-probe and full-FT paths unchanged.
Verify with Phase 6 grad-flow test green.

### DAG
Depends on Phase 6 (sequencing); strictly only Phase 4 in code.

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

**Adopted from architect-1 (consensus, see F-27/F-33)**: split into
**Phase 5 (loader-v1, MD-Cls only)** and **Phase 8 (loader-v2, SINCHI raw +
PEFT)**.

### Phase 5 — Loader v1: MD-Cls `.ckpt` only

#### Goal
Land `src/models/classification/dinov3_base/` with `DINOv3BaseInference` and
`DINOv3CustomWeights`. `_extract_state_dict` handles **only** the
`"state_dict"` + `net.` prefix flavour. SINCHI flavours raise
`NotImplementedError("SINCHI checkpoint flavours land in Phase 8")`.

#### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 5.1 | Naive `from .dinov3_base import *` in `classification/__init__.py` breaks ResNet users without `transformers` | HIGH | HIGH | Use the verified `speciesnet_base/__init__.py` pattern but **narrow the except** (closes F-23): `except ModuleNotFoundError as e: if e.name in ("transformers", "huggingface_hub", "peft"): pass; else: raise`. This catches only optional-dep absence; bugs in our own `dinov3_base` code (e.g. typo in an import) propagate. Test (Phase 0 I1+I13): in an env without `transformers`, `from src.models import classification` succeeds AND `classification.DINOv3CustomWeights` raises `AttributeError` with a clear message (custom `__getattr__` on the package: `def __getattr__(name): if name.startswith("DINOv3"): raise ImportError("DINOv3 inference requires: pip install -r requirements-dinov3.txt")`). Closes F-23 entirely. |
| 5.2 | Corrupt `.ckpt` → unhelpful traceback | MEDIUM | LOW | Wrap `torch.load` in try/except re-raising `RuntimeError(f"Failed to torch.load({path}); is this a valid checkpoint? Original: {e}")`. Test with a 16-byte garbage file. |
| 5.3 | `weights_only=True` future-default in torch ≥ 2.6 (closes F-44 partial) | MEDIUM | MEDIUM | Pass `weights_only=False` explicitly. Docs (Phase 9) include: "Only load checkpoints from trusted sources (`torch.load(..., weights_only=False)` allows arbitrary code execution.)." |
| 5.4 | `strict=False` silently swallows real bugs | MEDIUM | HIGH | After load: count `missing`/`unexpected`; raise (not warn) if `len(missing) > 5` OR any `missing.startswith("classifier.")` OR any `missing.startswith("feature.")` that isn't an obvious LoRA-adapter exception. v1 uses `strict=True` (Phase 5 only loads MD-Cls checkpoints whose layout we control). |
| 5.5 | MD-Cls checkpoint hidden-size vs supplied `model_name` mismatch | MEDIUM | HIGH | Pre-check: `state['classifier.weight'].shape[1] == self.net.embedding_dim`; raise with text identifying both numbers. |

#### Files MODIFIED vs CREATED
- **CREATED**: `src/models/classification/dinov3_base/{__init__.py, base_classifier.py, custom_weights.py}`.
- **MODIFIED**: `src/models/classification/__init__.py` — append one narrowly-guarded import (see Risk 5.1) AND add a package-level `__getattr__` for the actionable AttributeError. Semantic-identity for ResNet callers: all four existing wildcard re-exports unchanged.

#### DAG
Depends on Phase 4.

### Phase 8 — Loader v2: SINCHI raw + SINCHI PEFT

#### Goal
Extend `_extract_state_dict` to handle (a) SINCHI raw `.pt` with
`"model_state_dict"` + `backbone.*` → `feature.*` remap; (b) SINCHI PEFT
`.pt` with `backbone.base_model.model.*` keys, requiring either an
embedded `dinov3_peft_config` (own-saved, MD-Cls flavour from Phase 7) or
an explicit `peft_config=` kwarg.

#### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 8.1 | Auto-detect misclassifies → silent wrong-weights load | MEDIUM | HIGH | Three-way detection by **key-prefix inspection** (not heuristics on filenames). Ordering: (1) if top-level key is `"state_dict"` AND any sub-key starts with `"net."` → MD-Cls v1 path; (2) elif top-level `"model_state_dict"` AND any sub-key contains `"base_model.model."` → SINCHI PEFT; (3) elif top-level `"model_state_dict"` AND any sub-key starts with `"backbone."` → SINCHI raw; (4) else raise `ValueError` listing `list(ckpt.keys())[:10]`. Each branch is unit-tested with a synthetic state-dict. |
| 8.2 | LoRA adapter keys present but no `peft_config` provided → silent shape mismatch | HIGH | HIGH | Hard-raise: `ValueError("Checkpoint contains LoRA adapter keys (matched: <list>) but no peft_config was provided AND no 'dinov3_peft_config' is embedded in the checkpoint. Pass peft_config={r:..., lora_alpha:..., target_modules:[...]} or use a non-PEFT checkpoint.")`. Unit test. |
| 8.3 | Reliance on synthetic SINCHI fixtures alone (closes F-29 / F-38) | (would-be) HIGH | HIGH | **MANDATORY entry gate**: at least one real SINCHI raw `.pt` and one real SINCHI PEFT `.pt` are obtained and their inner key layouts recorded in `docs/design/dinov3-integration/sinchi_key_layouts.md` (committed as evidence) BEFORE Phase 8 merges. If unobtainable, Phase 8 ships with v1 surface only and the SINCHI loader is documented as "experimental / unverified." |
| 8.4 | hidden_size mismatch between checkpoint and supplied `model_name` | MEDIUM | HIGH | Pre-check (same as 5.5), applied to remapped keys. |
| 8.5 | wget download fallback (per `timm_base` parity) hits redirect/auth | LOW | LOW | Docs recommend `huggingface_hub.snapshot_download` for HF-hosted weights; keep wget fallback only for direct-URL `.pt` files. |

#### Files MODIFIED vs CREATED
- **CREATED**: `docs/design/dinov3-integration/sinchi_key_layouts.md` (evidence file).
- **MODIFIED**: `src/models/classification/dinov3_base/base_classifier.py` (`_extract_state_dict` extended).

#### DAG
Depends on Phases 5 and 7.

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

### Goal
Append commented YAML block to `configs/config.yaml`; ship three sample
configs; write `docs/dinov3.md`; append to `docs/usage.md` and `README.md`;
add `CHANGELOG.md` entry.

> **Note on doc timing (closes F-39)**: a minimal "experimental DINOv3
> linear-probe" doc with HF auth + memory warnings ships with **Phase 5**
> (`docs/dinov3.md` v0.1, ~40 lines). Phase 9 is the full expansion.
> Earlier doc work prevents the "users hit HF auth before docs exist"
> gap.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 9.1 | YAML append breaks `yaml.safe_load` parse | LOW | HIGH | Invariant I4 (default-config-unchanged) test. |
| 9.2 | Sample config OOMs on small GPU → bad first impression | HIGH | MEDIUM | Default sample is **ViT-S** (`facebook/dinov3-vits16-pretrain-lvd1689m`). GPU memory table labels rows as **MEASURED** or **ESTIMATED** (closes F-28); estimates are explicitly marked and a Phase-9 task is "replace estimates with measured numbers on RTX-3090 / A100 / V100 when available." |
| 9.3 | Doc example uses gated model without `huggingface-cli login` callout | MEDIUM | LOW | Every code block listing a DINOv3 model name is preceded by `pip install -r requirements-dinov3.txt` AND `huggingface-cli login`. |
| 9.4 | `requirements.txt` accidentally gains heavy deps | (would-be) HIGH | HIGH | Phase 9 modifies `requirements.txt` ONLY to add the comment `# DINOv3 training/inference: pip install -r requirements-dinov3.txt`. Invariant I2 hashes the runtime-deps lines. |
| 9.5 | Discoverability: users may follow only `pip install -r requirements.txt` then fail (closes F-31) | HIGH | MEDIUM | (a) README capability bullet for DINOv3 reads: *"DINOv3 fine-tuning (linear probe / full FT / LoRA / PiSSA). Requires `pip install -r requirements-dinov3.txt` and `huggingface-cli login`."* (b) `docs/usage.md` section opens with the same line. (c) Sample configs comment-header repeats it. |
| 9.6 | License/citation obligations (closes F-40) | N/A | LEGAL | `docs/dinov3.md` includes: License section, Citation section (BibTeX), Redistribution-of-derived-weights section. |
| 9.7 | Error taxonomy not standardised (closes F-46) | MEDIUM | LOW | `docs/dinov3.md` includes a "Troubleshooting" table mapping each `RuntimeError`/`ValueError`/`ImportError` raised by Phase-4–8 code to its cause and fix. |
| 9.8 | Reproducibility (closes F-45) | MEDIUM | MEDIUM | `docs/dinov3.md` "Reproducibility" section documents `seed_everything`, `Trainer(deterministic=True)`, pinned versions in `requirements-dinov3.txt`, and the embedded `dinov3_peft_config` in checkpoints (Phase 7). |

### Files MODIFIED vs CREATED
- **CREATED**: `configs/config_dinov3_linear.yaml`, `configs/config_dinov3_finetune.yaml`, `configs/config_dinov3_lora.yaml`, `docs/dinov3.md`, `CHANGELOG.md` (if absent).
- **MODIFIED**: `configs/config.yaml` (commented append), `README.md` (append), `docs/usage.md` (append), `requirements.txt` (1-line comment).

### Dependency footprint added
None.

### DAG
Depends on Phases 4, 5, 7, 8.

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

### Goal
Fix `src/models/classification/resnet_base/custom_weights.py` where
`num_layers=50` is hard-coded. Add `num_layers: int = 50` defaulted kwarg.
**Sequenced as Phase 1** (adopted from architect-1, satisfies F-32) so it
lands on a fresh regression harness with minimum noise.

### Failing-test-first protocol (closes F-20 by adoption)
Phase 0 ships `tests/regression/test_resnet18_loads.py` marked
`@pytest.mark.xfail(strict=True, reason="bug fixed in Phase 1")`. Phase 1's
single commit (a) deletes the marker and (b) applies the kwarg patch. CI
artifact for Phase 0 shows the test xfailing; CI artifact for Phase 1
shows it passing.

### Risk register
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1.1 | Downstream caller passing 4th positional arg | NEAR-ZERO | LOW | Grep across repo for `CustomWeights(` confirms zero 4-positional callers. |
| 1.2 | ResNet-50 (existing default) regresses due to patch typo | LOW | HIGH | Regression test exercises both ResNet-18 AND ResNet-50 ckpt load with `strict=True`. |
| 1.3 | Bisect ambiguity if folded into a larger phase | (would-be) LOW | MEDIUM | Phase 1 is its own commit. |
| 1.4 | Migration note hidden in final docs (closes F-47) | MEDIUM | LOW | CHANGELOG entry created in Phase 1 (not Phase 9) for this fix specifically: `## v1.x — CustomWeights num_layers bug fix (#PR). Previously hard-coded num_layers=50 prevented loading ResNet-18 user checkpoints; now defaulted kwarg.` |

### Files MODIFIED vs CREATED
- **CREATED**: `tests/regression/test_customweights_num_layers.py` (the xfail-flipped one from Phase 0), `CHANGELOG.md` entry (file may already exist).
- **MODIFIED**: `src/models/classification/resnet_base/custom_weights.py`:
  ```
  -    def __init__(self, weights=None, class_names=None, device="cpu"):
  +    def __init__(self, weights=None, class_names=None, device="cpu", num_layers: int = 50):
  ...
  -                                                   num_cls=self.num_cls, num_layers=50, url=None)
  +                                                   num_cls=self.num_cls, num_layers=num_layers, url=None)
  ```
  Semantic-identity for existing callers: 3-positional call sites get
  `num_layers=50` by default, identical to pre-patch.

### Rollback
`git revert <PHASE_1_SHA>` restores hard-code; regression test fails for
ResNet-18 (correctly indicates bug returned).

### DAG
Depends only on Phase 0.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

### Repo invariants (13) — failing one BLOCKS PR

Byte-equality scoped to four files only (closes F-24).

| # | Invariant | Test name | Mechanism |
|---|---|---|---|
| I1 | `python -c "import src"` succeeds in env with ONLY `requirements.txt` installed (no transformers/peft/huggingface_hub) | `test_default_env_imports_clean` | **Subprocess** with `uv run --isolated --with-requirements requirements.txt python -c "..."` (closes F-17 by adoption from architect-2's gap) |
| I2 | `requirements.txt` runtime deps unchanged (modulo allowed comment lines) | `test_requirements_runtime_lines_hash` | Hash of all non-comment, non-blank lines |
| I3 | `python main.py --config configs/config.yaml --dev` produces ResNet checkpoint whose **predictions** on a 4-image fixture match a frozen `tests/fixtures/baseline_predictions.json` (closes F-03 / F-19: behaviour snapshot, not byte-hash of state_dict) | `test_resnet_baseline_predictions` | argmax labels + confidence ≤ 1e-4; deterministic via `seed_everything(42)` + `Trainer(deterministic=True)` |
| I4 | `yaml.safe_load(open('configs/config.yaml'))` returns the pre-change dict | `test_default_config_yaml_equal` | dict equality, not yaml-dump string equality (closes F-19) |
| I5 | `CustomWeights(weights=..., class_names=[...])` loads ResNet-50 ckpt in env with no DINOv3 deps | `test_customweights_no_dinov3_deps` | Subprocess in isolated env |
| I6 | `algorithms.__dict__['Plain']` is the same class object pre/post | `test_plain_algorithm_class_identity` | id() comparison via subprocess on pre-change SHA |
| I7 | `models.__dict__['PlainResNetClassifier']` is the same class object | `test_plain_resnet_class_identity` | same |
| I8 | `main.py` SHA-256 unchanged | `test_main_unchanged` | Byte-equality (PERMANENT — never updated) |
| I9 | `src/algorithms/plain.py` SHA-256 unchanged | `test_plain_py_unchanged` | Byte-equality (PERMANENT) |
| I10 | `src/models/plain_resnet.py` SHA-256 unchanged | `test_plain_resnet_py_unchanged` | Byte-equality (PERMANENT) |
| I11 | `src/datasets/custom.py` SHA-256 unchanged **EXCEPT** in Phase 4 (transform kwarg) where the hash is updated **in the same commit** and a behaviour test (`test_transform_default_is_224`) is added permanently. After Phase 4, hash is again PERMANENT. | `test_custom_dataset_byte_or_behaviour` | Hash check + behaviour check (the behaviour check is the long-term guarantee) |
| I12 | `CustomWeights(weights=w, class_names=c, device=d)` legacy 3-arg call works | `test_customweights_legacy_call_signature` | Live import + load |
| I13 | In env without `transformers`, `from src.models import classification` succeeds AND `classification.DINOv3CustomWeights` raises `ImportError` with text `"requirements-dinov3.txt"` | `test_dinov3_absent_actionable_error` | Subprocess in isolated env (NEW this round, closes F-23) |

I8–I10 are PERMANENT byte-equality. I11 is byte-equality except for the
one allowed transform-kwarg commit. Comments, formatting, and other
file changes are protected by behaviour tests instead (closes F-24).

### Cross-cutting regression policy

- Every phase MUST run `pytest tests/regression/ -x` before merging.
- Phase 0 ships `tests/regression/dryrun_revert.sh` (closes F-25):
  ```bash
  #!/usr/bin/env bash
  # Usage: dryrun_revert.sh <SHA>
  set -euo pipefail
  SHA="$1"
  WT="$(pwd)/.revert-dryrun-$$"
  git worktree add "$WT" HEAD
  pushd "$WT"
  git revert --no-edit "$SHA"
  pytest tests/regression -x
  popd
  git worktree remove --force "$WT"
  ```
  (no `/tmp` writes; uses repo-local sibling worktree.)
- CI matrix activates **only at Phase 3** (closes F-26 staged-adoption):
  job (a) `pip install -r requirements.txt`; job (b) `pip install -r
  requirements.txt -r requirements-dinov3.txt`. Phase 0 establishes a
  single baseline job first.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Pyramid

| Layer | Count target | Per-phase distribution |
|---|---|---|
| **Unit** (CPU, <1s, no network) | ~18 | Mixup 4, model 4, algorithm 4, extract_state_dict v1+v2 6 |
| **Integration** (CPU, stub backbone, ~10s) | ~12 | Linear probe 2, full FT 2, LoRA 1, PiSSA 1, loader v1 2, loader v2 SINCHI raw+PEFT 2, transform-518 dataloader 1, PEFT roundtrip 1 |
| **Regression** (invariants) | 13 | All in Phase 0; xfail flip in Phase 1; one hash update in Phase 4 |
| **E2E** (real HF download, GPU) | 4 — manual, never in CI | Linear probe, full FT, LoRA, PiSSA |

### Stub strategy (single fixture for Phases 4–8, closes architect-2 F-16 spread)

`tests/conftest.py`:
```python
import torch, torch.nn as nn
from types import SimpleNamespace
import pytest

class _StubConfig:
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size

class _StubBackbone(nn.Module):
    """Tiny 1-layer ViT-shaped stub. Module names mirror HF DINOv3ViT so
    PEFT target_modules tests are meaningful."""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = _StubConfig(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, pixel_values=None, **kw):
        B = pixel_values.shape[0]
        h = self.pool(pixel_values.mean(1, keepdim=True)).flatten(1)
        h = h.expand(B, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=h.unsqueeze(1))

@pytest.fixture
def mock_hf_automodel(monkeypatch):
    """Replace AutoModel.from_pretrained — used in every Phase 4–8 test."""
    def _from_pretrained(*a, **kw):
        return _StubBackbone(hidden_size=64)
    import transformers
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        staticmethod(_from_pretrained))
```

### Fixtures (all under `tests/_artifacts/` and `tests/fixtures/` — NO `/tmp`)

- `tests/fixtures/baseline_predictions.json` (Phase 0)
- `tests/fixtures/resnet18_synth.ckpt` (Phase 1, synthesised in conftest)
- `tests/_artifacts/dinov3_linear_mdcls.ckpt` (written by Phase 4 integration test, consumed by Phase 5 + 7)
- `tests/fixtures/sinchi_raw_synth.pt` (Phase 8 — synthetic)
- `tests/fixtures/sinchi_peft_synth.pt` (Phase 8 — synthetic)
- `docs/design/dinov3-integration/sinchi_key_layouts.md` (Phase 8 — evidence from real binaries)

### HF auth handling (NEW for round 2)

CI **never** authenticates with HuggingFace. All Phase 4–8 tests use
`mock_hf_automodel`. Real-HF tests are marked `@pytest.mark.requires_hf`
and gated:
```python
@pytest.fixture(autouse=True, scope="session")
def _skip_if_no_hf_token():
    if not os.environ.get("HF_TOKEN") and "requires_hf" in pytest.config.getoption("-m", ""):
        pytest.skip("HF_TOKEN not set; run with HF_TOKEN=... pytest -m requires_hf")
```
Default `pytest` invocation excludes `-m requires_hf`. `huggingface_hub` is
an optional dep (in `requirements-dinov3.txt`); tests that reference it
gate via `pytest.importorskip("huggingface_hub")`.

### CI vs local

- **CI (every PR)** — baseline job (Phase 0+): unit + integration (with
  mocked HF) + regression.
- **CI matrix (from Phase 3)** — jobs (a) default-only, (b)
  default + dinov3. CUDA/HF/network disabled in both.
- **Local nightly (developer)** — `pytest -m requires_hf` and `pytest -m
  gpu`. Never in CI.

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

Mapped to research checklist (§8). Every command is executable; expected
outputs are concrete.

| # | Checklist item | Phase | Command | Expected | Tier |
|---|---|---|---|---|---|
| 1 | ResNet path bit-identical (behaviour) | 0 | `pytest tests/regression/test_resnet_baseline_predictions.py -x` | predictions match `baseline_predictions.json` | CI |
| 2 | Linear-probe trains | 4 | `python main.py --config configs/config_dinov3_linear.yaml --dev` | ckpt at `weights/Crop/DINOv3Linear/*.ckpt`; backbone grad-norms == 0 | local CPU (with mocked HF for CI integration test) |
| 3 | Full-FT trains | 6 | same, `freeze_backbone: False` | grad-norms > 0; trainable_params ≈ 86M | local GPU |
| 4 | Mixup toggle | 4 + 6 | `use_noisy_same_class_mixup: True` | loss curves differ; no shape errors | local CPU |
| 5 | LoRA + PiSSA | 7 | `peft.enable: True` (both methods) | trainable_params < 5% total; PiSSA LoRA-A diverges from gaussian init | local GPU |
| 6 | Load own MD-Cls ckpt | 5 | `DINOv3CustomWeights(weights=<ckpt#2>, class_names=[...])` | missing==[], unexpected==[] | CI (with synthetic) + local |
| 7 | Load SINCHI raw | 8 | same + real SINCHI raw `.pt` | strict-False load; remap applied | local (real artifact) |
| 8 | Load SINCHI PEFT | 8 | same + `peft_config={...}` + real PEFT `.pt` | re-wrap succeeds, adapter keys consumed | local (real artifact) |
| 9 | CustomWeights regression | 1 | `pytest tests/regression/test_customweights_num_layers.py` | both num_layers=18 and 50 load with `strict=True` | CI |
| 10 | Default env import + actionable error | 0/5 | `uv run --isolated --with-requirements requirements.txt python -c "from src.models.classification import CustomWeights, DINOv3CustomWeights"` | CustomWeights resolves; DINOv3CustomWeights raises `ImportError("...requirements-dinov3.txt")` | CI |
| 11 | Transform sizes (NEW) | 4 | `pytest tests/regression/test_transform_default_is_224.py tests/integration/test_dinov3_dataloader_emits_518.py` | ResNet batch `(B,3,224,224)`, DINOv3 batch `(B,3,518,518)` | CI |
| 12 | Real SINCHI key inspection (NEW gate) | 8 | review `docs/design/dinov3-integration/sinchi_key_layouts.md` | file present, lists actual top-level + sample sub-keys for both flavours | manual gate |
| 13 | PEFT target_modules validation (NEW) | 7 | `pytest tests/unit/test_peft_target_modules_validated.py` | empty-intersection raises; partial warns; full silent | CI |

### Definition of "done"

All 13 invariants + all 13 checklist tests pass + CI green on both matrix
jobs + `docs/dinov3.md` reviewed for HF auth / GPU memory / PEFT
defaults / license / citation + `sinchi_key_layouts.md` populated (or
Phase 8 explicitly deferred with documented "experimental" status).

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

### Top-level risk register (project-wide)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A phase breaks Invariant I1 (default env import) | MEDIUM (w/o discipline) | CATASTROPHIC | Phase 0 + per-phase regression gate; lazy imports; narrowed try/except in `classification/__init__.py`; subprocess-isolated env test (I1, I5, I13). |
| R2 | DINOv3 HF model gated → cryptic auth error | HIGH | MEDIUM | Wrap `from_pretrained` with actionable RuntimeError; docs include `huggingface-cli login` step; CI never authenticates. |
| R3 | GPU OOM on default full-FT config | HIGH | MEDIUM | Sample ships ViT-S + bs=4 + fp16; memory table labels MEASURED vs ESTIMATED. |
| R4 | `peft` API drift | MEDIUM | MEDIUM | Pin floor+ceiling; CI matrix tests both. |
| R5 | Loader auto-detect silently misclassifies | MEDIUM | HIGH | Key-prefix detection (not heuristics); fail-loud on missing classifier keys; hidden-size pre-check; LoRA-without-peft_config hard raises. |
| R6 | Reverting Phase 6 with 7 present leaves broken file | LOW (with DAG) | MEDIUM | Reverse-DAG enforced; dryrun_revert.sh refuses out-of-order revert; documented order: 9→8→7→6→5→4→3→2→1→0. |
| R7 | Lightning checkpoint format drift | LOW | LOW | Pin Lightning floor (Phase 3, in `requirements-dinov3.txt`). |
| R8 | Reproducibility (RNG/seed/transform) | MEDIUM | MEDIUM | `seed_everything` + `Trainer(deterministic=True)` documented; embedded `dinov3_peft_config` + `model_name` + `image_size` in checkpoint metadata (closes F-41, F-45). |
| R9 | DINOv3 license/redistribution | N/A | LEGAL | Phase 9 docs; no automatic download; checkpoint metadata records source `model_name`. |
| R10 | Non-CUDA execution | MEDIUM | LOW | Sample configs use `precision: 16-mixed` with a "CUDA-only" comment; default fallback is `precision: 32`. |

### Per-phase rollback DAG (operationalised)

Reverse order. Each step: `git revert --no-edit <SHA>` then verification:

```
PHASE_9 → revert → pytest tests/regression -x                            → green
PHASE_8 → revert → pytest tests/regression tests/integration/test_dinov3_custom_weights_mdckpt.py -x  → green (v1 loader still works)
PHASE_7 → revert → pytest tests/regression tests/integration/test_dinov3_finetune_grad_flow.py -x      → green
PHASE_6 → revert → pytest tests/regression tests/integration/test_dinov3_linear_probe.py -x            → green
PHASE_5 → revert → pytest tests/regression tests/integration/test_dinov3_linear_probe.py -x            → green
PHASE_4 → revert → pytest tests/regression tests/unit/test_mixup.py -x                                 → green
PHASE_3 → revert → pytest tests/regression -x                                                          → green
PHASE_2 → revert → pytest tests/regression -x                                                          → green
PHASE_1 → revert → pytest tests/regression -x → test_resnet18_loads RED (expected; bug back); test_resnet50 GREEN
PHASE_0 → revert → manual confirmation; baseline removed
```

### Single-command emergency revert

```bash
git revert --no-edit \
  $PHASE_9_SHA $PHASE_8_SHA $PHASE_7_SHA $PHASE_6_SHA \
  $PHASE_5_SHA $PHASE_4_SHA $PHASE_3_SHA $PHASE_2_SHA $PHASE_1_SHA
# Phase 0 (tests) stays → strict win: better tests, no DINOv3 code
pytest tests/regression -x
```

---

## Rationale (risk/backwards-compat lens, evolved)

Same north star as round 1, refined by inquisitor pressure:

- **"Independently revertable" downgraded to "reverse-DAG revertable"**
  (F-22): more honest, still operationally rigorous via
  `dryrun_revert.sh` + ordered emergency-revert recipe.
- **Loader split into v1/v2** (F-27, F-33): MVP capability ships at end
  of Phase 5, *before* PEFT/SINCHI complexity, matching architect-1.
- **Single dependency policy** (F-21, F-34): one `requirements-dinov3.txt`
  file. No extras_require (no `pyproject.toml` exists), no editing
  `requirements.txt` (preserves I1, I2). Docs always preface DINOv3
  commands with the dinov3-reqs install line.
- **Byte-equality restricted to four immutable files** (F-24);
  everything else uses behaviour / dict / import tests.
- **Transform resolved at the dataset-config layer** (F-30, F-37):
  defaulted `image_size: int = 224` kwarg keeps ResNet bit-identical,
  DINOv3 overrides to 518 via config.
- **PEFT target-module list runtime-validated** (F-05): `ValueError` on
  empty intersection, warning on partial.
- **SINCHI compatibility blocked behind a real-checkpoint inspection
  gate** (F-29, F-38): if no real `.pt` arrives, Phase 8 is deferred and
  the loader stays at v1 — a partial rollout that is still net-positive.
- **Narrowed try/except in classification/__init__.py** (F-23): catches
  only `ModuleNotFoundError` for the three optional deps; bugs in our
  own code propagate; absent class yields an `ImportError` (not silent
  `AttributeError`) with the install command in the message.
- **License/citation/error-taxonomy/reproducibility/non-CUDA all docs
  obligations** (F-40, F-42, F-45, F-46) folded into Phase 9.

The plan still optimises for "ship 5 of 9 phases and hold the rest"
posture: Phases 0–5 + 9 deliver linear-probe DINOv3 with own-checkpoint
inference and zero `peft` dependency. PEFT and SINCHI compatibility are
strictly additive on top.

---

## New ledger items proposed

(None. All 12 scope items addressed.)

---

## Discussion summary

Round 1 produced three coherent but diverging plans. Inquisitor surfaced
47 critiques, of which ~11 targeted my proposal directly (F-21..F-31)
and several cross-architect divergences required convergence. The single
most consequential adoption is the loader split (architect-1's idea),
which makes the MVP genuinely shippable in 5 phases instead of 7.

## Consensus points

- **Loader split into v1 (MD-Cls only) / v2 (SINCHI)** — adopted from
  **architect-1** (F-27, F-33). Strictly better partial-rollout posture
  than my round-1 "loader after PEFT".
- **Failing-test-first protocol for the CustomWeights bug fix** —
  adopted from **architect-2** (F-20). Phase 0 ships the xfail; Phase 1
  flips it.
- **Bug fix as Phase 1 (not Phase 2)** — adopted from **architect-1**
  (F-32). Pre-flight against the regression harness with minimum
  contemporaneous noise.
- **Single shared `mock_hf_automodel` fixture** — adopted from
  **architect-2** (test-strategy section), extended here to include
  PEFT-aware module names so target-module validation tests are
  meaningful.
- **Subprocess / isolated env for "no transformers" tests** —
  adopted from **architect-2** (F-17). My round-1 plan implied
  `sys.modules` patching, which is weaker.
- **Behaviour-snapshot for ResNet regression instead of state_dict
  hash** — adopted from **architect-1**'s F-03 framing (the inquisitor
  pointed this out). My round-1 plan used state_dict hash (fragile).
- **Embed `peft_config` metadata in checkpoints** — consensus across
  architect-1, architect-2, and inquisitor F-06 / F-41. My round-1
  already proposed this; reaffirmed.

## Disagreements

- **Architect-2's "PEFT branch already implemented in Phase 3/4
  skeleton, verified later"** — I continue to disagree (aligns with
  inquisitor F-13). PEFT code + lazy import + checkpoint hook MUST land
  in the PEFT phase, not earlier. My Phase 7 enforces this.
- **Architect-2's coverage threshold (`--cov-fail-under=100`) as a
  blanket gate** — disagree (inquisitor F-18). Coverage thresholds are
  a quality gate, not a DINOv3 acceptance gate. My Phase 9 acceptance
  does not require coverage thresholds, only the 13 behavioural tests.
- **Architect-1's direct edit to `requirements.txt`** — disagree
  (preserves my round-1 position, reinforced by F-34 needing one
  explicit choice). Editing `requirements.txt` to add `transformers`
  would break Invariant I1 for every ResNet-only user on `pip install
  --upgrade`. I keep `requirements-dinov3.txt`.

## Evolution

Changes from round 1 → round 2:

1. Phase count 8 → 9 (loader split).
2. Phase order: bug-fix moves from Phase 2 → Phase 1; mixup moves from
   Phase 1 → Phase 2 (both depend only on Phase 0, but bug-fix-first
   gives the harness its first real exercise).
3. Transform-size question (R1-Phase-4-Risk-4.7) resolved with a
   concrete `image_size: int = 224` kwarg in `custom.py`.
4. PEFT target-module list now runtime-validated (Phase 7 Risk 7.3 +
   `test_peft_target_modules_validated`).
5. `TaskType.FEATURE_EXTRACTION` retained, with an inline citation to
   HF docs verifying it (Phase 7 Risk 7.2).
6. SINCHI real-checkpoint key-inspection gate added as MANDATORY Phase
   8 entry criterion (Risk 8.3 + new Acceptance item 12).
7. Byte-equality invariants narrowed from 4 files → 3 PERMANENT + 1
   PERMANENT-after-Phase-4 (I8–I11). I3 changed from state_dict hash
   to behaviour snapshot. NEW: I13 (actionable error for absent
   DINOv3 deps).
8. Loader split: Phase 5 (MD-Cls v1) + Phase 8 (SINCHI v2).
9. Rollback dry-run operationalised as committed shell script
   (`tests/regression/dryrun_revert.sh`) using `git worktree` — no
   `/tmp`.
10. CI matrix staged: baseline at Phase 0, two-env matrix activates at
    Phase 3.
11. `try/except ImportError: pass` narrowed to `except
    ModuleNotFoundError as e: if e.name in {"transformers",
    "huggingface_hub", "peft"}: pass; else: raise`, with a
    package-level `__getattr__` raising an actionable `ImportError`
    when a DINOv3 class is accessed without deps.
12. Dependency policy declared **singular and explicit**:
    `requirements-dinov3.txt` (no `pyproject.toml` extras —
    `pyproject.toml` does not exist in repo; verified). Every
    user-facing command is prefaced with the install line.
13. Sample-config GPU memory numbers now explicitly labelled MEASURED
    vs ESTIMATED with a Phase-9 task to replace estimates.
14. License/citation (F-40), error taxonomy (F-46), reproducibility
    (F-45), non-CUDA notes (F-42) folded into Phase 9 docs.
15. Phase 5 ships `docs/dinov3.md` v0.1 (minimal warnings) so users
    don't hit HF auth before any docs exist (closes F-39).
16. HF auth test policy declared: `mock_hf_automodel` for all CI tests;
    `@pytest.mark.requires_hf` for nightly; `pytest.importorskip` for
    `huggingface_hub` references.
17. CHANGELOG entry for the bug fix lands in Phase 1 (not Phase 9) per
    F-47.

## NEW This Round

- **Phase 5 / Phase 8 loader split** (was unified Phase 7).
- **Phase 4 transform-kwarg edit to `src/datasets/custom.py`** (was a
  deferred decision).
- **Mandatory `sinchi_key_layouts.md` evidence file gating Phase 8**
  (was synthetic-only fixtures).
- **`test_peft_target_modules_validated` + runtime validator in
  `_wrap_with_peft`** (was a static target_modules list).
- **`tests/regression/dryrun_revert.sh`** (was a manual ritual).
- **I3 reworked from state_dict-hash → predictions snapshot.**
- **I13 (actionable absent-deps error) added.**
- **Single-source dependency policy**: `requirements-dinov3.txt` (was
  ambiguous between extras_require, requirements file, and lazy
  import — now lazy import is the *runtime* mechanism, the file is the
  *install* mechanism, no extras).
- **HF auth test policy** with `requires_hf` marker and
  `importorskip`.
- **License/citation/error-taxonomy/reproducibility/non-CUDA docs
  obligations** explicitly mapped into Phase 9.
- **Behaviour-snapshot regression baseline** (`baseline_predictions.json`)
  replaces the brittle state_dict-hash plan.

STATUS: NEW=11 DISAGREEMENTS=3
