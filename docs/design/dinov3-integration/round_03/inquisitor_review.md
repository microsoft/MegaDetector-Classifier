# Inquisitor review — round 3

## Round-2 NEW critique closures (verification)

- **A1-N1 — RESOLVED** (`architect-1_proposal.md#config-and-docs-phase`, `architect-1_proposal.md#test-strategy`). Architect-1 adds `huggingface_hub>=0.20,<1` to `requirements-dinov3.txt`, leaves default `requirements.txt` unchanged, and lists all three optional packages (`transformers`, `peft`, `huggingface_hub`) in absent/default-surface tests.
- **A1-N2 — RESOLVED** (`architect-1_proposal.md#model-and-algorithm-phase`, `architect-1_proposal.md#regression-protection`). Architect-1 drops `build_dinov3_eval_transform`, adopts the defaulted `Custom_Crop(image_size: int = 224)` path, and keeps ResNet behavior protected by the 224 default.
- **A1-N3 — RESOLVED** (`architect-1_proposal.md#risk-and-rollback`). Architect-1 replaces the absolute rollback snippet with a repo-local `scripts/rollback_dry_run.sh` using `mktemp -d` under a repo-local worktree parent and `trap cleanup EXIT INT TERM`.
- **A2-N1 — RESOLVED** (`architect-2_proposal.md#inference-loader-phase`, `architect-2_proposal.md#test-strategy`). Architect-2 moves real SINCHI key-layout evidence to `docs/design/dinov3-integration/sinchi_key_layouts.md`; tests parse fenced blocks from that tracked design document rather than `tests/_artifacts/`.
- **A2-N2 — RESOLVED** (`architect-2_proposal.md#config-and-docs-phase`, `architect-2_proposal.md#test-strategy`). Architect-2 drops `pip install -e .[test]` and `[project.optional-dependencies]`; CI/test setup is requirements-only.
- **A3-N1 — RESOLVED** (`architect-3_proposal.md#config-and-docs-phase`, `architect-3_proposal.md#test-strategy`). Architect-3 chooses one absent-DINOv3 surface: DINOv3 names are present via lazy export, and access/construction raises actionable `ImportError`. Tests assert that exact surface.
- **A3-N2 — RESOLVED** (`architect-3_proposal.md#regression-protection`, `architect-3_proposal.md#test-strategy`). Architect-3 replaces the state-dict hash sentinel with behavioral snapshots: deterministic forward logits plus one-step loss, with no state-dict hash gate.

## Round-2 directive compliance (verification)

1. **Round-2 disagreements deleted — COMPLIANT.** All three proposals end with `DISAGREEMENTS=0`; disagreement sections are removed or reduced to “None,” and no `DEADLOCKED-WITH` remains.
2. **Dependency policy — COMPLIANT.** All proposals keep `requirements.txt` unchanged for default installs and place `transformers`, `peft`, and `huggingface_hub` in `requirements-dinov3.txt`, with lazy imports and absent-env/default-surface tests.
3. **PEFT target validation — COMPLIANT.** All proposals retain `TaskType.FEATURE_EXTRACTION`; empty target intersection is an error, and partial intersection is a warning that names unmatched targets.
4. **Transform injection — COMPLIANT.** All proposals converge on architect-3’s defaulted `Custom_Crop(image_size: int = 224)` approach, with DINOv3 overriding to 518 and no separate transform shim.
5. **SINCHI key evidence path — COMPLIANT.** All proposals place real key-layout evidence under `docs/design/dinov3-integration/sinchi_key_layouts.md`.
6. **No packaging migration — COMPLIANT.** None of the proposals add package metadata or require `-e .[test]`; CI/install plans are requirements-only.
7. **Behavior snapshots, not state-dict hashes — COMPLIANT.** Regression gates use predictions/logits/loss snapshots. State-dict hashes are not retained as merge gates.
8. **Absent-DINOv3 surface — COMPLIANT.** The converged surface is actionable `ImportError` for DINOv3 access without optional dependencies, and tests match that choice.

## NEW critiques (this round)

None.

The architects’ self-reported `NEW This Round` entries are closure-only, not new unresolved flaws:

| Architect | Self-reported NEW items | Inquisitor classification |
|---|---:|---|
| Architect-1 | 5 | CLOSURE: A1-N3 rollback script, A1-N1 `huggingface_hub`, A3-N1 absent-surface standardization, PEFT validation directive, A3-N2 state-dict-hash removal |
| Architect-2 | 5 | CLOSURE: A2-N1 SINCHI evidence relocation, A2-N2 requirements-only installs, A1-N1 dependency triad, PEFT validation directive, A3-N1 absent-surface standardization |
| Architect-3 | 4 | CLOSURE: A3-N1 absent-surface standardization, A3-N2 behavior snapshot, A1-N1 optional-dependency triad, transform/default-kwarg consensus |

No proposal introduces a new deadlock, new architectural branch, or new unresolved implementation hazard beyond the round-2 closure work.

## Consensus assessment

Convergence is genuine rather than surface-deep. The remaining design now has one dependency policy, one transform-injection mechanism, one absent-DINOv3 error surface, one PEFT target-validation rule, one SINCHI evidence location, one requirements-only test-install policy, and one regression strategy based on behavior snapshots. Round-3 changes close round-2 findings and directives instead of opening new design alternatives.

## Per-scope-item verification

<a name="phase-decomposition"></a>
### Scope item: phase-decomposition
All three architects address phase ordering and dependencies. Architect-1 keeps a nine-phase implementation DAG with rollback points; architect-2 adds phase-0 CI/scaffolding and fails-first sequencing; architect-3 preserves the dependency DAG and overlays risk reductions. Treatment quality: complete; no uncovered phase-decomposition gap remains.

<a name="mixup-utility-phase"></a>
### Scope item: mixup-utility-phase
All three architects keep `noisy_same_class_mixup` as an early isolated utility phase with unit tests before DINOv3 model work. Treatment quality: adequate and consistent; no gap remains.

<a name="model-and-algorithm-phase"></a>
### Scope item: model-and-algorithm-phase
All three architects cover the frozen-backbone DINOv3 classifier and `DINOv3Linear` algorithm path, lazy optional imports, test doubles, and integration coverage. Treatment quality: complete; consensus uses defaulted `Custom_Crop` rather than a separate transform shim.

<a name="full-finetune-phase"></a>
### Scope item: full-finetune-phase
All three architects address the full-finetune toggle, unfreezing behavior, dual optimizer parameter groups, and tests that distinguish frozen vs full-backbone parameter updates. Treatment quality: complete; no gap remains.

<a name="peft-phase"></a>
### Scope item: peft-phase
All three architects address LoRA/PiSSA through PEFT wrapping, keep `TaskType.FEATURE_EXTRACTION`, specify empty-target error and partial-target warning behavior, and include checkpoint/behavior tests. Treatment quality: complete; no state-dict-hash regression gate remains.

<a name="inference-loader-phase"></a>
### Scope item: inference-loader-phase
All three architects cover loader support for MD-Classifier checkpoints, SINCHI raw `.pt`, and SINCHI PEFT `.pt`; all require fail-loud key classification and tracked SINCHI key-layout evidence under the design docs. Treatment quality: complete; no `tests/_artifacts/` evidence dependency remains.

<a name="config-and-docs-phase"></a>
### Scope item: config-and-docs-phase
All three architects cover additive config/docs updates, `requirements-dinov3.txt` for optional dependencies, default `requirements.txt` preservation, and README/usage instructions. Treatment quality: complete; no packaging migration remains.

<a name="bug-fix-phase"></a>
### Scope item: bug-fix-phase
All three architects address the `CustomWeights` `num_layers=50` hardcode and require ResNet-18 plus ResNet-50 loader regression coverage. Treatment quality: complete; no gap remains.

<a name="regression-protection"></a>
### Scope item: regression-protection
All three architects protect existing ResNet/default behavior with default-config invariants and behavior snapshots. DINOv3/PEFT regressions use logits/loss snapshots rather than state-dict hashes. Treatment quality: complete; no gap remains.

<a name="test-strategy"></a>
### Scope item: test-strategy
All three architects define unit, integration, regression, subprocess absent-env, and CI/local test layers. Architect-2 gives the most explicit CI matrix; architect-3 gives the most explicit fixture policy; architect-1 aligns with both round-2 directives. Treatment quality: complete.

<a name="acceptance-and-validation"></a>
### Scope item: acceptance-and-validation
All three architects map acceptance to concrete commands and expected outcomes, including default install, DINOv3 install, ResNet non-regression, PEFT behavior, loader roundtrips, and rollback dry-runs. Treatment quality: complete; no gap remains.

<a name="risk-and-rollback"></a>
### Scope item: risk-and-rollback
All three architects provide phase-specific risks and rollback procedures. Architect-1 specifically closes the rollback dry-run critique with a repo-local `mktemp -d` worktree and `trap` cleanup; architect-2 and architect-3 retain reverse-DAG rollback and dry-run checks. Treatment quality: complete; no gap remains.

## Coverage analysis

- Ledger items: 12
- Verified this round: 12
- Uncovered: []

STATUS: CONVERGED  SCOPE_CHECK=PASS  COVERED=12/12 UNCOVERED=[]              NEW=0
