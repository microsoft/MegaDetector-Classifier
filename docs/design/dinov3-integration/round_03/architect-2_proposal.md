# DINOv3 integration — phased plan (architect-2: testing/verification, round 3)

> **Perspective unchanged.** A phase does not exist until it is verifiable. Test
> plans precede code. Round 3 is the convergence round: I delete my round-2
> disagreements (all three were adjudicated MISTAKEN-BY=architect-2 by the
> inquisitor — I re-read the round-2 proposals and confirm they are stale),
> address my two NEW round-2 critiques (A2-N1, A2-N2), and adopt the
> lead-injected binding directives. The 12-anchor skeleton, phase ordering, and
> test pyramid are unchanged from round 2; only the deltas required by the
> directives appear below.

## Executive overview (deltas from round 2)

1. **A2-N1 fix.** The real SINCHI key-evidence file moves out of the transient
   test-artifact root into the tracked design-docs tree:
   `docs/design/dinov3-integration/sinchi_key_layouts.md` (the path the
   inquisitor named explicitly). Phase 8 entry criteria, the Phase-8 markers,
   and the loader-v2 tests all consume the docs path; `tests/_artifacts/` no
   longer holds gating evidence.
2. **A2-N2 fix.** The CI plan stops assuming `[project.optional-dependencies]`
   and `pip install -e .[test]`. The repo has no packaging metadata today
   (`pyproject.toml` currently contains only the pytest ini block from
   Phase 0a). Adding extras would be an unintended packaging migration. I
   replace every CI install line with a requirements-only invocation:
   `pip install -r requirements.txt -r requirements-dev.txt` for the min leg
   and `pip install -r requirements.txt -r requirements-dev.txt -r requirements-dinov3.txt`
   for the dinov3 leg. Phase 0a still ships the pytest ini block; Phase 0b
   ships the CI workflow file ONLY (no extras, no `-e .`).
3. **Single dependency policy (binding).** `requirements-dinov3.txt` is the
   sole optional-deps surface and contains exactly three pinned lines:
   `transformers`, `peft`, `huggingface_hub`. All three are import-local in
   product code; all three are covered by absent-env subprocess tests.
4. **PEFT validator semantics (binding).** `validate_target_modules` returns
   the intersection with `model.named_modules()`. Empty intersection → ERROR
   (`ValueError` listing what was requested vs what is available). Partial
   intersection → WARNING (`warnings.warn(UserWarning)`) naming the unmatched
   requested targets and proceeding with the intersection. `TaskType.FEATURE_EXTRACTION`
   is the configured `LoraConfig.task_type`.
5. **Transform decision (binding).** Adopt architect-3's design: a single
   defaulted `image_size: int = 224` kwarg on the existing `Custom_Crop`
   transform. `DINOv3Linear` requests `image_size=518` via
   `configs/config_dinov3_linear.yaml`. No new `build_dinov3_eval_transform`
   shim. (Credited in §Consensus.)
6. **Regression-snapshot surface (binding).** Behavior snapshots
   (logits/loss with explicit RNG seed and version-header metadata) replace
   any state-dict hashing. No state-dict SHA is a merge gate at any phase.
7. **Absent-DINOv3-class surface (binding, chosen).** Single contract:
   `from src.models.classification import DINOv3CustomWeights` in a min-env
   raises `ImportError("DINOv3CustomWeights requires the DINOv3 extras. "
   "Install: pip install -r requirements-dinov3.txt")`. The symbol is NOT
   silently absent; it is present in the `classification` namespace as a
   `__getattr__`-resolved stub that raises on access. Tests match that
   choice exactly (see §inference-loader-phase).
8. **Coverage threshold (binding).** `--cov-fail-under=100` is NOT a DINOv3
   acceptance gate. Coverage runs in Phase 0b are advisory only and produce
   a CI artifact; no `--cov-fail-under=...` flag appears anywhere in the
   merge-gate command.

All other round-2 decisions (Phase 0a/0b split, loader v1 vs v2 split,
NotImplementedError guards in Phase 4 lifted in Phase 6/7, `uv venv`
absent-env subprocess tests, `basetemp=tests/_pytest_tmp`, three-layer
regression, mixup port determinism via monkeypatched `np.random.beta` and
`torch.randint`, real-key fixture as Phase 8 gate, behavior-snapshot
baseline of `Plain`) stand unchanged.

## Phase index (round-3 deltas only)

| # | Phase | Round-3 delta |
|---|---|---|
| 0a | Pytest harness | Unchanged. `pyproject.toml` gains ONLY the `[tool.pytest.ini_options]` block — NO `[project.*]`, NO `[build-system]`. |
| 0b | CI matrix | Workflow uses `pip install -r requirements.txt -r requirements-dev.txt` (min) and `... -r requirements-dinov3.txt` (full). No `-e .[test]`. |
| 1 | `CustomWeights` bug fix | Unchanged. |
| 2 | Mixup utility port | Unchanged. |
| 3 | DINOv3 model | Transform-kwarg lands here as `image_size: int = 224` (architect-3's design, credited). |
| 4 | DINOv3Linear MVP | Unchanged; `configs/config_dinov3_linear.yaml` sets `image_size: 518`. |
| 5 | Loader-v1 (MD-Cls) | `classification/__init__.py` registers `DINOv3CustomWeights` as a `__getattr__` shim that raises `ImportError` if `transformers`/`peft`/`huggingface_hub` missing; minimal `docs/dinov3-experimental.md` ships here. |
| 6 | Full FT | Unchanged. |
| 7 | PEFT | `validate_target_modules` semantics codified (empty=ERROR, partial=WARNING listing unmatched targets); `TaskType.FEATURE_EXTRACTION` confirmed. |
| 8 | Loader-v2 (SINCHI) | Gating evidence path moves to `docs/design/dinov3-integration/sinchi_key_layouts.md`; tests load key lists from that doc. |
| 9 | Config + docs | `requirements.txt` comment-only addition pointing to `requirements-dinov3.txt`. |

---

<a name="phase-decomposition"></a>
## Phase decomposition — addresses scope_item:phase-decomposition

Unchanged from round 2 except for the two evidence/packaging deltas:

- Phase 0a's `pyproject.toml` patch contains ONLY the `[tool.pytest.ini_options]`
  block (testpaths, addopts with `--basetemp=tests/_pytest_tmp`, markers,
  `tmp_path_retention_policy`). No `[project]`, no
  `[project.optional-dependencies]`, no `[build-system]`. Adopting extras
  would imply the repo is pip-installable, which is a separate decision out
  of scope for DINOv3.
- Phase 8's entry criterion now reads: "`docs/design/dinov3-integration/sinchi_key_layouts.md`
  is committed and signed off by a human reviewer; the file contains the
  key lists of one real SINCHI raw `.pt` and one real SINCHI PEFT `.pt`,
  with provenance (source path, SHA-256, date, reviewer initials)." The
  Phase-8 tests `parse` this Markdown file's fenced code blocks and assert
  the in-memory synthetic dicts use those exact keys.

Dependency graph and parallel-safety are unchanged.

---

<a name="mixup-utility-phase"></a>
## Phase: mixup utility — addresses scope_item:mixup-utility-phase

Unchanged from round 2. All nine unit tests, determinism rules
(monkeypatched `np.random.beta` and `torch.randint`), and the basetemp
policy are unchanged.

---

<a name="model-and-algorithm-phase"></a>
## Phase: model + algorithm — addresses scope_item:model-and-algorithm-phase

### Phase 3 — model

Unchanged from round 2 except:

- **Transform injection** is now exactly architect-3's design: add a single
  defaulted `image_size: int = 224` kwarg to the existing `Custom_Crop`
  constructor in `src/datasets/custom.py`, threaded into
  `Classification_Inference_Transform`. Default preserves ResNet bit-identity.
  `configs/config_dinov3_linear.yaml` sets `image_size: 518`. No new
  `transforms.py` module; no `build_dinov3_eval_transform` shim. The
  integration test `tests/integration/test_dinov3_dataset_transform.py`
  asserts both directions (518 for DINOv3 config, 224 for default config).

### Phase 4 — DINOv3Linear (linear-probe MVP)

Unchanged. `freeze_backbone=False` and `peft.enable=True` still raise
`NotImplementedError`. The MVP `.ckpt` artifact still lands under
`tests/_artifacts/`.

---

<a name="full-finetune-phase"></a>
## Phase: full fine-tune toggle — addresses scope_item:full-finetune-phase

Unchanged from round 2.

---

<a name="peft-phase"></a>
## Phase: PEFT (LoRA / PiSSA) — addresses scope_item:peft-phase

### What lands (round-3 refinements)

- `src/algorithms/_peft_targets.py`:
  ```python
  def validate_target_modules(model, requested):
      """Return intersection of requested LoRA targets with model.named_modules().

      Empty intersection -> ValueError (fail loud).
      Partial intersection -> warnings.warn(UserWarning) listing unmatched
      requested targets; return the matched subset.
      """
      available = {name.split(".")[-1] for name, _ in model.named_modules()}
      matched = [t for t in requested if t in available]
      unmatched = [t for t in requested if t not in available]
      if not matched:
          raise ValueError(
              f"None of {requested!r} match any module in the backbone. "
              f"Available leaf module names ending in '_proj': "
              f"{sorted(n for n in available if n.endswith('_proj'))}. "
              f"See docs/dinov3.md#peft-target-modules."
          )
      if unmatched:
          warnings.warn(
              f"PEFT target_modules {unmatched!r} do not exist in the backbone; "
              f"proceeding with matched subset {matched!r}.",
              UserWarning, stacklevel=2,
          )
      return matched
  ```
- `LoraConfig(..., task_type=TaskType.FEATURE_EXTRACTION, target_modules=matched, ...)`.
- `peft` is imported lazily inside `_wrap_with_peft()` — module-top imports
  remain `torch` only.

### Test plan (additions vs round 2)

| Test name | Assertion |
|---|---|
| `test_validate_target_modules_raises_when_empty_intersection` | All requested targets missing → `ValueError` whose message contains every requested name AND the available `_proj` leaf-name list |
| `test_validate_target_modules_warns_and_returns_matched_when_partial` | 7 requested, 6 present (no `gate_proj`) → `pytest.warns(UserWarning, match="gate_proj")` AND returned list equals the 6 matched names in input order |
| `test_validate_target_modules_silent_when_full_match` | All requested present → no warnings; returned list equals input |
| `test_lora_config_uses_task_type_feature_extraction` | Spy on `LoraConfig` ctor; assert `task_type == TaskType.FEATURE_EXTRACTION` |

Round-2 PEFT roundtrip and trainable-ratio tests are unchanged.

### Absent-`peft` test
Unchanged: `uv venv` subprocess, real absent environment, runs in CI
matrix leg 1.

---

<a name="inference-loader-phase"></a>
## Phase: inference loader — addresses scope_item:inference-loader-phase

### Phase 5 — loader-v1 (MD-Cls only)

Unchanged from round 2 except the **absent-class surface contract**, now
binding:

`src/models/classification/__init__.py` uses module-level `__getattr__` (PEP 562):

```python
# src/models/classification/__init__.py
from .resnet_base import CustomWeights, PlainResNetClassifier  # always available

_DINOV3_INSTALL_HINT = (
    "DINOv3CustomWeights requires DINOv3 optional dependencies. "
    "Install: pip install -r requirements-dinov3.txt"
)

def __getattr__(name):
    if name == "DINOv3CustomWeights":
        try:
            from .dinov3_base import DINOv3CustomWeights as _cls
        except ModuleNotFoundError as e:
            # Only swallow optional-dep MNFEs; bubble everything else.
            if e.name not in {"transformers", "peft", "huggingface_hub"}:
                raise
            raise ImportError(_DINOV3_INSTALL_HINT) from e
        return _cls
    raise AttributeError(name)
```

**Tests must match this choice exactly:**

| Test name | Assertion |
|---|---|
| `test_classification_dinov3_symbol_raises_importerror_in_min_env` | In `uv venv` min env: `from src.models.classification import DINOv3CustomWeights` raises `ImportError` whose message contains `"pip install -r requirements-dinov3.txt"` |
| `test_classification_dinov3_symbol_resolves_in_full_env` | In dinov3-extras env: same import succeeds; `isinstance(obj, type)` |
| `test_classification_init_propagates_non_dep_importerror` | Inject a non-dep `ModuleNotFoundError` (e.g. typo `import foo_bar`) into a shadow copy of `dinov3_base/base_classifier.py`; assert the import RAISES (`ModuleNotFoundError`, not the install-hint `ImportError`) |
| `test_classification_customweights_resolves_in_min_env` | ResNet `CustomWeights` is imported eagerly and works in min env |

All other Phase-5 unit + integration + fail-loud tests are unchanged from
round 2.

### Phase 8 — loader-v2 (SINCHI raw + SINCHI PEFT)

**Hard gate (revised per A2-N1):** PR may NOT merge until
`docs/design/dinov3-integration/sinchi_key_layouts.md` is committed. The
file contains:

- A YAML front-matter / header with provenance: source path, SHA-256,
  inspection date, reviewer initials.
- A fenced ` ```text ` block titled `sinchi_raw_keys` containing the
  output of `list(torch.load(real_sinchi_ckpt, weights_only=False).keys())`
  followed by `list(torch.load(...)['model_state_dict'].keys())[:50]`.
- A fenced ` ```text ` block titled `sinchi_peft_keys` doing the same for
  one real SINCHI PEFT `.pt`.

A small loader `tests/_helpers/sinchi_keys.py::load_sinchi_keys(name)`
parses the fenced blocks by title and returns the key list. The Phase-8
tests synthesize zero-tensor `state_dict`s from those parsed keys; no
binary checkpoint is committed.

| Test name | Assertion |
|---|---|
| `test_sinchi_key_layouts_doc_exists_and_parses` | `docs/design/dinov3-integration/sinchi_key_layouts.md` parses; both named blocks present; reviewer-initials header non-empty |
| `test_extract_state_dict_remaps_backbone_to_feature_for_sinchi_raw` | Uses `load_sinchi_keys("sinchi_raw_keys")`; in-memory dict with EXACT real keys; remap behavior |
| `test_extract_state_dict_preserves_peft_adapter_keys_in_sinchi_peft` | Uses `load_sinchi_keys("sinchi_peft_keys")` |
| `test_custom_weights_loads_sinchi_raw_pt` | Integration |
| `test_custom_weights_loads_sinchi_peft_pt_with_peft_config_kwarg` | Integration |
| `test_custom_weights_loads_md_cls_peft_ckpt_auto_rewraps` | Uses Phase 7 embedded `dinov3_peft_config` |
| `test_custom_weights_raises_when_lora_keys_present_but_no_peft_config` | Fail-loud per architect-3 R7.4 |

`tests/_artifacts/` no longer holds Phase-8 gating evidence; only
Phase-4/Phase-7 ckpt outputs live there (transient build products).

---

<a name="config-and-docs-phase"></a>
## Phase: config + docs — addresses scope_item:config-and-docs-phase

Unchanged from round 2 except:

- `requirements.txt` gains exactly ONE comment line:
  `# For DINOv3 features: pip install -r requirements-dinov3.txt`.
- `requirements-dinov3.txt` (new, lands in Phase 3) contains exactly three
  pinned lines: `transformers`, `peft`, `huggingface_hub`. Pins to be
  decided in Phase 3 by reading the source repo `Andres_SSD_1/SINCHI_DINOV3`'s
  own requirements; recorded in this file's header comment.
- `docs/dinov3.md` documents the install command, the PEFT target-modules
  validation semantics (empty=error, partial=warning), the absent-class
  `ImportError` surface, and the GPU memory table.
- `docs/dinov3-experimental.md` ships in Phase 5 (HF auth + memory caveats).
- No `pyproject.toml` `[project.*]` block is added in any phase.

The config-yaml regression test is unchanged: direct dict equality, not
`yaml.safe_dump` hash.

---

<a name="bug-fix-phase"></a>
## Phase: CustomWeights bug fix — addresses scope_item:bug-fix-phase

Unchanged from round 2.

---

<a name="regression-protection"></a>
## Regression protection — addresses scope_item:regression-protection

Unchanged from round 2 except: **no state-dict SHA appears as a merge
gate at any layer.** Layer 2 (behavior snapshots) is now the only
mechanism for catching numeric drift; the snapshot files are
`logits`/`loss` JSON with version-metadata headers. Layer 3
(byte-equality) remains restricted to four immutable files and is a
guard against accidental edits, not against numeric drift.

CI matrix in Layer 4 uses requirements-only installs:

```yaml
# .github/workflows/tests.yml (Phase 0b)
strategy:
  matrix:
    leg: [min, dinov3]
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with: { python-version: "3.10" }
  - name: Install (min)
    if: matrix.leg == 'min'
    run: |
      python -m pip install --upgrade pip
      pip install -r requirements.txt -r requirements-dev.txt
  - name: Install (dinov3)
    if: matrix.leg == 'dinov3'
    run: |
      python -m pip install --upgrade pip
      pip install -r requirements.txt -r requirements-dev.txt -r requirements-dinov3.txt
  - name: Test (min)
    if: matrix.leg == 'min'
    run: pytest -m "not gpu and not slow and not requires_dinov3_deps"
  - name: Test (dinov3)
    if: matrix.leg == 'dinov3'
    run: pytest -m "not gpu and not slow"
```

`requirements-dev.txt` is a tiny new file containing `pytest`, `pytest-cov`,
`pytest-mock`, `uv` (for the absent-env subprocess test). It is NOT
DINOv3-specific. No `-e .` anywhere.

---

<a name="test-strategy"></a>
## Test strategy — addresses scope_item:test-strategy

### Install commands (round-3 binding)

All test invocations assume the requirements-only install above. The
README and `docs/dinov3.md` document:

```bash
# Min env (default behavior; ResNet path only):
pip install -r requirements.txt -r requirements-dev.txt
pytest -m "not gpu and not slow and not requires_dinov3_deps"

# Full env (DINOv3 features):
pip install -r requirements.txt -r requirements-dev.txt -r requirements-dinov3.txt
pytest -m "not gpu and not slow"
```

There is **no** `pip install -e .[test]`, no `pip install .`, no setup.py.
The repo remains a script-style project until a deliberate packaging
decision is made.

### Absent-package test (binding: covers all three optional deps)

The absent-env subprocess test asserts the min env is missing all three:
`transformers`, `peft`, `huggingface_hub`. Specifically, after `uv venv`
+ `uv pip install -r requirements.txt -r requirements-dev.txt`:

| Test name | Assertion |
|---|---|
| `test_default_env_has_no_optional_dinov3_deps` | Subprocess imports each of the three optional packages and asserts each raises `ModuleNotFoundError`. Guards against accidental transitive pulls. |
| `test_default_env_imports_src_clean` | Subprocess imports `src`, `from src.models import classification`, `from src.algorithms import Plain`; all succeed; `DINOv3CustomWeights` access raises `ImportError` with the install hint |
| `test_default_env_main_help_succeeds` | Subprocess runs `python main.py --help`; exit 0 |

### Coverage (advisory only, per binding directive)

Phase 0b runs `pytest --cov=src --cov-report=term --cov-report=xml`. No
`--cov-fail-under=...` flag. The XML report is uploaded as a CI artifact
for trend visibility. Coverage failure is NOT a merge gate at any phase.

Everything else (test pyramid, fixture strategy, basetemp, determinism
rules, naming convention) is unchanged from round 2.

---

<a name="acceptance-and-validation"></a>
## Acceptance and validation — addresses scope_item:acceptance-and-validation

Unchanged from round 2 except:

- Validation item 7 / 8 ("Load SINCHI raw / PEFT ckpt") now references
  `docs/design/dinov3-integration/sinchi_key_layouts.md` as the evidence
  source.
- Validation item 10 ("Lazy-import (real absent env)") explicitly covers
  `huggingface_hub` in addition to `transformers` and `peft`.

### Acceptance gate ("done") — revised invocation

```bash
# CI gate — both env-matrix legs (no -e .):
pip install -r requirements.txt -r requirements-dev.txt
pytest -m "not gpu and not slow and not requires_dinov3_deps"
# then in the dinov3 matrix leg:
pip install -r requirements.txt -r requirements-dev.txt -r requirements-dinov3.txt
pytest -m "not gpu and not slow"
# manual GPU gate on dev workstation (Phase 8+):
pytest -m gpu tests/gpu/
# Phase 8 hard prereq (path moved per A2-N1):
test -f docs/design/dinov3-integration/sinchi_key_layouts.md
```

All four commands must exit 0 with a signed-off
`sinchi_key_layouts.md`. No coverage threshold appears anywhere in
this gate.

---

<a name="risk-and-rollback"></a>
## Risk and rollback — addresses scope_item:risk-and-rollback

Unchanged from round 2. Round-3 additions to the risk register:

| Risk | Phase | Mitigation | Rollback |
|---|---|---|---|
| Accidental packaging migration via `-e .[test]` | 0b | Round-3 binding: requirements-only installs; no `[project.*]` in `pyproject.toml`; CI grep forbids `pip install -e .` literals | n/a (prevented by review) |
| `huggingface_hub` pulled transitively and masks an absent-env bug | all | `test_default_env_has_no_optional_dinov3_deps` asserts MNFE for all three optional packages | Pin upper bounds in `requirements.txt` consumers if drift seen |
| SINCHI evidence file drifts from real ckpts over time | 8 | Header date + reviewer initials; labelled `sinchi-keys-bump` PR for refresh | Re-run inspection, update doc |
| PEFT partial-match warning silently ignored by users | 7 | Warning is `UserWarning` (always shown by default); `docs/dinov3.md` documents it | n/a |
| Absent-DINOv3 `ImportError` confused with a real bug | 5 | Install-hint string is unique and grep-checked in `docs/dinov3.md` | n/a |

Reverse-DAG rollback order unchanged.

---

## Rationale (testing lens)

The round-3 changes are surface-level relative to round 2:

1. **A2-N1** — gating evidence belongs under `docs/design/`, not under a
   transient artifact tree. The remap-rule spec is a design artifact; it
   should live where design artifacts live and survive `git clean -fdx`.
2. **A2-N2** — adding `-e .[test]` and `[project.optional-dependencies]`
   would change what kind of project this is. Requirements-only installs
   preserve the existing script-style layout and still cover both env
   matrix legs cleanly.
3. **Binding directives** — the inquisitor's directives match my own
   testing-first principles (behavior snapshots over byte hashes,
   no-coverage-as-gate, absent-env tests cover all optional deps, one
   absent-class surface contract). I adopt them as written.

---

## Consensus points

Adopted from **architect-1**:
- Loader split into v1/v2; MVP framing at Gate 1; PR-template failing-test-first
  workflow; `docs/release_gates.md`; sample-config split for discoverability.
- (round 2 disagreement on "HF credentials in CI" was MISTAKEN-BY=architect-2;
  architect-1's round-2 proposal does not require HF in CI.)

Adopted from **architect-3**:
- `requirements-dinov3.txt` as the single optional-deps file (no extras).
- 12-invariant table approach for surface-level regression.
- In-phase PEFT ckpt-roundtrip test (R6.3).
- Fail-loud loader assertions (R7.4, R7.6, R7.7).
- Embed `peft_config` in MD-Cls ckpts via `on_save_checkpoint` (R6.6).
- Sample config defaults to ViT-S + `batch_size=4` + fp16 (R8.4).
- Reverse-DAG rollback (R6).
- **`speciesnet_base`-style guarded import as the proven pattern.**
- **`image_size: int = 224` defaulted kwarg on existing `Custom_Crop`**
  (round 2 I claimed I differed from architect-3 on this; the inquisitor
  ruled MISTAKEN-BY=architect-2 — architect-3 selected exactly this design.
  I credit architect-3 with the integration point and remove my round-2
  alternative.)
- **Dict-equality on config and runtime-dep-set-equality on requirements**
  (round 2 I claimed byte-equality difference; the inquisitor ruled
  MISTAKEN-BY=architect-2 — architect-3 already uses non-byte equality on
  these two files.)

From the **inquisitor's industry comparison**:
- Runtime validation of PEFT `target_modules` against `named_modules()` (F-05),
  with empty=ERROR / partial=WARNING semantics (binding directive).
- `TaskType.FEATURE_EXTRACTION` is the correct task type for backbone-as-feature-extractor
  with an external classifier head.
- DINOv3 ViT attention proj names (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `up_proj`, `down_proj`; `gate_proj` only on gated MLP variants) — the
  target list must tolerate non-gated variants, which the partial-warning
  semantics handles cleanly.

## Disagreements

(None.) All three round-2 disagreements were adjudicated MISTAKEN-BY=architect-2
by the inquisitor. I re-read architect-1's and architect-3's round-2
proposals to confirm:

- "HF credentials in CI" — architect-1's round-2 proposal explicitly states
  no HF token required in CI; tests use mocked HF. Quote in inquisitor's
  audit, F-01 ADDRESSED. My round-2 disagreement was a misread. Deleted.
- "Byte-equality on `configs/config.yaml` and `requirements.txt`" —
  architect-3's round-2 proposal uses dict equality on YAML and runtime
  dep-set equality on requirements. Quote in inquisitor's audit, F-24
  ADDRESSED. My round-2 disagreement was stale. Deleted.
- "Transform decision deferred" — architect-3's round-2 proposal selects
  the defaulted dataset-transform kwarg explicitly. Quote in inquisitor's
  audit, F-30 ADDRESSED. My round-2 disagreement was stale. Deleted; the
  design is now jointly adopted and credited to architect-3 in
  §Consensus.

No current, quoted round-2 conflicts remain.

## Evolution

| Round 2 claim | Round 3 fix | Driving directive |
|---|---|---|
| SINCHI key evidence at `tests/_artifacts/sinchi_keys_real.txt` | Moved to `docs/design/dinov3-integration/sinchi_key_layouts.md`; tests parse fenced blocks from that doc | A2-N1 |
| Phase 0b adds `[project.optional-dependencies]`; CI uses `pip install -e .[test]` | Phase 0b adds the workflow file only; CI uses `pip install -r requirements.txt -r requirements-dev.txt [-r requirements-dinov3.txt]`; no extras, no `-e .`, no `pyproject` `[project.*]` block | A2-N2 |
| Optional-deps absent-env test names only `transformers` and `peft` | Test asserts MNFE for all three: `transformers`, `peft`, `huggingface_hub` | binding directive (dependency policy) |
| PEFT validator: empty → ValueError; partial behavior left at "returned list + warning" | Codified: empty → ValueError (lists requested + available `_proj` names); partial → `UserWarning` listing unmatched targets and proceeding with matched subset; `TaskType.FEATURE_EXTRACTION` confirmed | binding directive (PEFT validation) |
| Transform: 2-line additive `image_size` kwarg, framed as a disagreement with architect-3 | Same design but credited to architect-3 in §Consensus; disagreement deleted as MISTAKEN-BY=architect-2 | binding directive (transform) + inquisitor adjudication |
| Absent-DINOv3-class surface left implicit ("`hasattr(classification, "DINOv3CustomWeights") == False`") | Single binding contract: `classification.DINOv3CustomWeights` raises `ImportError("...pip install -r requirements-dinov3.txt")` in min env; tests match exactly | binding directive (absent-optional-class surface) |
| Behavior snapshot wording allowed both predictions JSON and `state_dict` SHA as parallel mechanisms | Behavior snapshots (logits/loss with explicit seed + version header) only; no `state_dict` SHA is a gate | binding directive (regression snapshots) |
| Coverage thresholds explicitly excluded from DINOv3 gate but `--cov-fail-under=100` mentioned as a Phase-0b option | `--cov-fail-under=...` does not appear in any merge command at any phase | binding directive (coverage threshold) |

## NEW This Round

Items added in round 3 that were not in round 2:

1. **SINCHI key evidence relocated** from `tests/_artifacts/sinchi_keys_real.txt`
   to `docs/design/dinov3-integration/sinchi_key_layouts.md` with parseable
   fenced blocks (`sinchi_raw_keys`, `sinchi_peft_keys`) and a
   `_helpers/sinchi_keys.py` loader; Phase 8 gate command updated.
2. **Requirements-only CI install plan** (no `pyproject.toml`
   `[project.*]` block, no `pip install -e .[test]`); new
   `requirements-dev.txt` (pytest, pytest-cov, pytest-mock, uv);
   workflow YAML in §regression-protection.
3. **`huggingface_hub` added to the optional-deps absent-env test surface**
   and to `requirements-dinov3.txt` as the third pinned line, completing
   the single dependency-policy invariant.
4. **PEFT validator semantics codified** as empty=`ValueError` (with both
   requested and available `_proj` names in the message) and
   partial=`UserWarning` listing unmatched targets; `TaskType.FEATURE_EXTRACTION`
   confirmed in the LoRA config and in a dedicated unit test.
5. **Absent-DINOv3-class binding contract**: `classification.DINOv3CustomWeights`
   raises `ImportError` with install-hint string in min env (via
   module-level `__getattr__`), tested in both min and full envs and
   distinguished from non-dep `ModuleNotFoundError`s.

STATUS: NEW=5 DISAGREEMENTS=0
