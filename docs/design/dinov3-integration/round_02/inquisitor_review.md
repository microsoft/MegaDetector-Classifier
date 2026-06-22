# Inquisitor review — round 2

## Addressed/unaddressed audit of round-1 critiques

### architect-1
- F-01 — ADDRESSED. Quote: "NO HF token required in CI" and all Phase 4 tests use `MockAutoModel`; real HF is marked/non-default.
- F-02 — ADDRESSED. Quote: "training_step calls `noisy_same_class_mixup` only if `self.hparams.use_mixup` is True" and tests cover on/off paths.
- F-03 — ADDRESSED. Quote: "Behavior snapshot (primary, introduced P0)" with predictions JSON and state-dict SHA demoted to a warning.
- F-04 — ADDRESSED. Quote: "Default `requirements.txt` is untouched" and DINOv3 deps move to `requirements-dinov3.txt`.
- F-05 — ADDRESSED. Quote: "_validate_peft_targets(target_modules) ... assert at least one module name ends with one of `target_modules`".
- F-06 — ADDRESSED. Quote: `dinov3_metadata` includes "peft_method, lora_r, lora_alpha, lora_dropout, lora_target_modules" and loader consumes metadata.
- F-07 — ADDRESSED. Quote: `_categorize_keys` returns "backbone/classifier/adapter/unknown" and fails on `unknown`.
- F-08 — ADDRESSED. Quote: tests assert DINOv3 `(B,3,518,518)` and ResNet `(B,3,224,224)`.
- F-09 — ADDRESSED. Quote: "docs/dinov3_experimental.md" ships in Phase 5 with auth, memory, image-size, security, and license notes.

### architect-2
- F-10 — ADDRESSED primary concern. Quote: "Phase 0a ... no coverage threshold, no new CI job" and Phase 0b is detachable. Residual packaging concern is NEW A2-N2 below.
- F-11 — ADDRESSED. Quote: "Adopting architect-1's split: loader-v1 ... loader-v2".
- F-12 — ADDRESSED. Quote: "Full-FT ... real implementation deltas, not test-only".
- F-13 — ADDRESSED. Quote: "implement `_wrap_with_peft()`" in Phase 7 and the guard removal is the failing-test-first proof.
- F-14 — ADDRESSED. Quote: pytest `basetemp=tests/_pytest_tmp`; every `tmp_path` resolves repo-locally.
- F-15 — ADDRESSED conceptually. Quote: Phase 8 hard-gated on `sinchi_keys_real.txt` from real checkpoint inspection. Residual path-policy conflict is NEW A2-N1 below.
- F-16 — ADDRESSED. Quote: model forward tests cover both `(2,3,224,224)` and `(2,3,518,518)` and dataset tests assert both sizes.
- F-17 — ADDRESSED. Quote: absent-package test creates a genuine `uv venv`, not `sys.modules` poisoning.
- F-18 — ADDRESSED. Quote: "Coverage thresholds are NOT a merge gate".
- F-19 — ADDRESSED. Quote: default config comparison is direct dict equality, "NOT a yaml.safe_dump SHA".
- F-20 — ADDRESSED. Quote: PR must include CI artifact link or `git notes` with raw failing pytest output.

### architect-3
- F-21 — ADDRESSED. Quote: "one `requirements-dinov3.txt` file. No extras_require" and docs preface commands with that install line.
- F-22 — ADDRESSED. Quote: "Independently revertable downgraded to reverse-DAG revertable".
- F-23 — ADDRESSED in principle. Quote: catches only `ModuleNotFoundError` for optional deps; bugs propagate. Residual absent-class inconsistency is NEW A3-N1 below.
- F-24 — ADDRESSED. Quote: "Byte-equality restricted to four immutable files" and dict/behavior tests elsewhere.
- F-25 — ADDRESSED. Quote: Phase 0 ships `tests/regression/dryrun_revert.sh` with concrete worktree/revert/test/remove steps.
- F-26 — ADDRESSED. Quote: CI matrix activates "only at Phase 3".
- F-27 — ADDRESSED. Quote: loader split into "Phase 5 (loader-v1, MD-Cls only)" and "Phase 8 (loader-v2, SINCHI raw + PEFT)".
- F-28 — ADDRESSED. Quote: memory table labels rows as "MEASURED" or "ESTIMATED".
- F-29 — ADDRESSED. Quote: real SINCHI layouts recorded in `sinchi_key_layouts.md` before Phase 8 merges.
- F-30 — ADDRESSED. Quote: transform resolved with `image_size: int = 224` defaulted kwarg and DINOv3 overrides to 518.
- F-31 — ADDRESSED. Quote: every user-facing DINOv3 command is prefaced with `pip install -r requirements-dinov3.txt`.

## NEW round-2 critiques (count toward gate)

### architect-1
- A1-N1. `huggingface_hub` is missing from the optional-dependency contract. The proposal tests absence of `transformers` and `peft`, but never names `huggingface_hub`; final plan must test a default environment with none of the three optional packages and ensure any Hub-specific imports are local to code that downloads or authenticates.
- A1-N2. Transform injection is now a new forked design. A1 creates `src/datasets/transforms.py::build_dinov3_eval_transform(image_size)`, while A2/A3 use a defaulted `image_size` kwarg on the existing dataset transform. A separate shim is acceptable only if the final synthesis proves it is the sole path consumed by `Custom_Crop`; otherwise choose the defaulted-kwarg design.
- A1-N3. Rollback dry-run uses a fixed absolute sibling path in the example. In a shared workspace that can collide with another run or leave stale worktrees. Use a checked-in script with a unique repo-local worktree name and cleanup trap.

### architect-2
- A2-N1. The real SINCHI key evidence file is placed under `tests/_artifacts/sinchi_keys_real.txt`, while the same proposal treats `tests/_artifacts` as a transient test-artifact root. Evidence that gates Phase 8 should live under `docs/design/dinov3-integration/sinchi_key_layouts.md` or another tracked docs path, not under the artifact directory.
- A2-N2. The CI commands use `-e .[test]` and Phase 0b adds `[project.optional-dependencies]`; if the repo currently has no packaging metadata, this becomes a packaging migration. Either add a complete package config deliberately or use requirements-only test installs.

### architect-3
- A3-N1. The absent-DINOv3 class contract is internally inconsistent: it says `classification.DINOv3CustomWeights` raises `AttributeError` with a clear message, but the proposed `__getattr__` raises `ImportError`; the acceptance command imports the symbol without catching the expected exception. Pick one surface and test it exactly.
- A3-N2. The PEFT-disabled regression proposes hashing a resulting `state_dict`, reintroducing the brittle hash pattern rejected in F-03. Use behavior-level logits/loss snapshots with version metadata, not a binary state-dict hash.

## Disagreement adjudication (9 total — 3 per architect)

### A1↔A2 disagreements
- A1 claim: A2 still bundles all loader work late. A2 round 2 explicitly adopts loader-v1 after linear probe and loader-v2 after PEFT. Ruling: MISTAKEN-BY=architect-1.
- A1 claim: A2 still pre-implements full-FT/PEFT in the skeleton. A2 round 2 says both are real implementation phases that lift guards. Ruling: MISTAKEN-BY=architect-1.
- A2 claim: A1 still requires HF credentials in CI. A1 round 2 says no HF token is required in CI and uses mocked HF. Ruling: MISTAKEN-BY=architect-2.

### A1↔A3 disagreements
- A1 claim: A3 still uses broad `try/except ImportError: pass`. A3 round 2 narrows to `ModuleNotFoundError` for optional deps only. Ruling: MISTAKEN-BY=architect-1.
- A3 claim: A1 still edits `requirements.txt` for heavy deps. A1 round 2 switches to `requirements-dinov3.txt`. Ruling: MISTAKEN-BY=architect-3.
- Transform mechanism differs but is not listed as a disagreement by the architects. Ruling: RESOLVED-TOWARD=architect-3 for the final synthesis unless A1 can prove its shim is what `Custom_Crop` consumes; the existing-transform defaulted kwarg is the lower-risk integration point.

### A2↔A3 disagreements
- A2 claim: A3 still byte-freezes config and requirements. A3 round 2 uses dict equality for config and runtime-dependency-line hashing for requirements. Ruling: MISTAKEN-BY=architect-2.
- A2 claim: A3 still defers the transform decision. A3 round 2 selects a defaulted dataset transform kwarg. Ruling: MISTAKEN-BY=architect-2.
- A3 claim: A2 still uses a blanket coverage threshold gate. A2 round 2 says coverage is advisory and not a DINOv3 gate. Ruling: MISTAKEN-BY=architect-3.
- A3 claim: A2 still pre-implements PEFT in Phase 3/4. A2 round 2 says PEFT lands in Phase 7. Ruling: MISTAKEN-BY=architect-3.

## Final-synthesis directives for the lead
- Tell round-3 architects to delete stale disagreements and report `DISAGREEMENTS=0` unless they identify a current, quoted round-2 conflict.
- Use one dependency policy: default `requirements.txt` unchanged; `requirements-dinov3.txt` contains `transformers`, `peft`, and `huggingface_hub`; all three are lazy/import-local and covered by absent-env tests.
- Adopt runtime PEFT target validation: empty intersection is an error; partial intersection is a warning listing unmatched targets. Keep `TaskType.FEATURE_EXTRACTION` for backbone-as-feature-extractor.
- Use the dataset-config defaulted `image_size=224` kwarg unless code inspection proves A1's transform shim is wired into the existing dataset path.
- Put real SINCHI key-layout evidence under tracked design docs, not transient test artifacts.
- Avoid packaging migration in the test plan unless final synthesis explicitly adds a complete package config.
- Use behavior snapshots for regressions; do not revive state-dict hashes as pass/fail gates.
- Define the absent optional-class surface exactly: either the symbol is absent from `classification`, or accessing it raises `ImportError` with install instructions. Tests must match that choice.

## Missing perspectives still uncovered
- No ledger item is uncovered. Remaining gaps are detail-level: exact contents of `requirements-dinov3.txt`, exact absent-DINOv3 exception surface, and exact tracked location for real SINCHI key-layout evidence.

## Industry comparison (round 2 additions)
- PEFT `LoraConfig.target_modules` docs state list entries exact-match or match module names that end with the listed strings, and unknown architectures require manually specified targets: https://huggingface.co/docs/peft/main/en/package_reference/lora. This supports runtime `named_modules()` validation.
- PEFT `TaskType.FEATURE_EXTRACTION` is documented as "Feature extraction" providing hidden states usable as downstream embeddings/features: https://huggingface.co/docs/peft/main/en/package_reference/peft_types. This supports wrapping the DINOv3 backbone while keeping an external classifier.
- Transformers DINOv3 ViT source defines attention `k_proj`, `v_proj`, `q_proj`, `o_proj`; MLP `up_proj`, `down_proj`; and `gate_proj` only in `DINOv3ViTGatedMLP`: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py. Therefore the target list is plausible but must tolerate non-gated variants.
- `huggingface_hub` docs show Hub downloads use `from huggingface_hub import snapshot_download`: https://huggingface.co/docs/huggingface_hub/guides/download. Python packaging docs describe extras as optional dependency expansion: https://packaging.python.org/en/latest/specifications/dependency-specifiers/#extras. If the project uses a separate requirements file instead of extras, the equivalent invariant is local imports plus documented install commands.

## Per-scope-item verification

<a name="phase-decomposition"></a>
### Scope item: phase-decomposition
All three architects address this non-vacuously with phase tables, DAGs, and entry/exit criteria. Verified 3/3. Final doc should synthesize: scaffold → bug fix → mixup → optional deps/transform → model+linear → loader-v1 → full-FT → PEFT → loader-v2 → docs.

<a name="mixup-utility-phase"></a>
### Scope item: mixup-utility-phase
All three specify the mixup port, tests, seeding/determinism, and no production callsite until the algorithm phase. Verified 3/3. Final doc should keep mixup as a leaf phase and wire it off-by-default in the algorithm phase.

<a name="model-and-algorithm-phase"></a>
### Scope item: model-and-algorithm-phase
All three specify `DINOv3Classifier`, `DINOv3Linear`, lazy `transformers`, mocked HF tests, 518 input coverage, and frozen-backbone linear-probe scope. Verified 3/3. Final doc must settle transform injection and optional-dep absent-class behavior.

<a name="full-finetune-phase"></a>
### Scope item: full-finetune-phase
All three now treat full fine-tune as a real implementation phase with unfreeze, dual param groups, gradient-flow tests, and memory warnings. Verified 3/3. Final doc should keep it separate from the linear-probe PR.

<a name="peft-phase"></a>
### Scope item: peft-phase
All three now include PEFT implementation, lazy import, LoRA/PiSSA tests, target-module validation, and checkpoint metadata. Verified 3/3. Final doc should adopt empty-error/partial-warning semantics and behavior snapshots.

<a name="inference-loader-phase"></a>
### Scope item: inference-loader-phase
All three split MD-Cls loader-v1 from SINCHI loader-v2 and include fail-loud loading. Verified 3/3. Final doc should require tracked real key-layout evidence before claiming SINCHI compatibility.

<a name="config-and-docs-phase"></a>
### Scope item: config-and-docs-phase
All three provide config/docs timing, sample configs, install docs, HF auth, memory notes, security, and license/citation. Verified 3/3. Final doc should ship minimal docs with loader-v1 and full docs after loader-v2.

<a name="bug-fix-phase"></a>
### Scope item: bug-fix-phase
All three cover the `CustomWeights(num_layers)` fix with failing-test-first evidence and ResNet-18/50 regression tests. Verified 3/3. Final doc should require a standalone bug-fix PR after baseline scaffolding.

<a name="regression-protection"></a>
### Scope item: regression-protection
All three cover regression protection with default-env import checks, behavior snapshots, config equality, and selected critical-file checks. Verified 3/3. Final doc should avoid binary state-dict hashes as hard gates.

<a name="test-strategy"></a>
### Scope item: test-strategy
All three define unit/integration/regression/manual GPU/real-artifact layers, fixtures, CI-vs-local split, and no network/HF token in default CI. Verified 3/3. Final doc must fix A2's packaging command and evidence-file path.

<a name="acceptance-and-validation"></a>
### Scope item: acceptance-and-validation
All three map the research checklist to commands and add transform-size and real-SINCHI evidence gates. Verified 3/3. Final doc should keep CI/local/GPU/real-artifact labels.

<a name="risk-and-rollback"></a>
### Scope item: risk-and-rollback
All three include risk registers and reverse-DAG rollback. Verified 3/3. Final doc should use unique worktree dry-runs and behavior-based verification after reverts.

## Coverage analysis
- Ledger items: 12
- Verified this round: 12
- Uncovered: []
- Round-2 proposal anchor count: all 12 items appear in all 3 architect proposals.
- Disagreement gate: all 9 stated disagreements are stale/mistaken or prescriptively resolved; no DEADLOCKED item remains.

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=12/12 UNCOVERED=[] NEW=7
