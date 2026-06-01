# Inquisitor review — round 1 (final)

## Proposal critiques (per architect)

### architect-1 (phasing/MVP)

F-01. Phase 2 assumes "HuggingFace credentials available in CI secret" as entry criteria (`architect-1:112-115`). That conflicts with the research invariant that optional deps and HF access must be lazy/opt-in (`research:138-149`, `research:784-790`). CI should be hermetic by default with mocked `AutoModel`; real HF runs belong in a marked GPU/HF job.

F-02. The Phase 2 description says mixup exists but is not wired yet (`architect-1:103-109`, `architect-1:39`). The research algorithm applies mixup inside `training_step` after `encode` and before `classifier` (`research:364-370`). If Phase 2 claims to land `DINOv3Linear`, the phase either implements a disabled-by-default mixup path with tests, or explicitly defers the config key and acceptance item; the current wording hides the boundary.

F-03. The regression hash plan trains a baseline and compares a serialized `state_dict` hash (`architect-1:281-288`). This is brittle across PyTorch/Lightning versions, dataloader order, CPU/GPU kernels, and serialization metadata. It needs version-pinned metadata, deterministic dataloader seeding, and a behavior-level prediction snapshot, not just a binary hash.

F-04. Adding `transformers>=4.40.0` directly to `requirements.txt` in Phase 2 (`architect-1:105-111`) changes the default install surface before DINOv3 is used. That may be acceptable if the repo treats all requirements as runtime, but it is not reconciled with the optional/lazy dependency invariant (`research:746-747`, `research:780`). The final plan must pick one dependency policy.

F-05. The PEFT phase copies SINCHI `target_modules=[q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj]` without a runtime named-module validation step (`architect-1:161-168`). Current HF DINOv3ViT uses split `q_proj/k_proj/v_proj/o_proj` and MLP `up_proj/down_proj`, with `gate_proj` only for gated MLP variants. The implementation must inspect `model.named_modules()` and fail early if none of the configured targets match.

F-06. The PEFT checkpoint story does not require saving `peft_config` metadata into MD-Cls Lightning checkpoints (`architect-1:169-178`, `architect-1:207-224`). Inference then depends on users passing an external JSON blob. The final design should embed model name, image size, PEFT method, LoRA rank/alpha/dropout, and target modules in the checkpoint and let `DINOv3CustomWeights` auto-rewrap.

F-07. The SINCHI loader exit criteria are too permissive around `strict=False` (`architect-1:214-223`). Missing classifier weights, hidden-size mismatch, or LoRA keys without a rewrapped backbone must be hard failures. A non-empty `missing` list is not enough; key classes must be categorized and asserted.

F-08. The 518 transform change is under-tested. Architect-1 only asserts a direct forward on `(1,3,518,518)` (`architect-1:197-200`), but the research report says `Custom_Crop` is reused and transforms must switch to DINOv3's 518 bicubic path (`research:106-109`, `research:245-251`). The final plan needs a dataset-level test proving the DINOv3 path sees 518 while the ResNet path remains 224/unchanged.

F-09. Docs are last, after both shipping gates (`architect-1:232-249`, `architect-1:410-417`). Gate-1 users would hit HF auth, GPU memory, and transform-size issues before docs exist. A minimal "experimental DINOv3 linear probe" doc and auth/memory warning must ship with the first public gate; full docs can still land later.

### architect-2 (testing)

F-10. The testing-first plan correctly adds baseline scaffolding, but it introduces `pyproject.toml`, pytest extras, coverage thresholds, and a new CI job in one Phase 0 (`architect-2:353-392`). That is a large infrastructure change for a repo with no tests. It needs a smaller bootstrap: install/test command proven on current requirements first, then coverage gates only after baseline tests pass reliably.

F-11. Phase 7 puts the entire inference loader last (`architect-2:26-28`, `architect-2:238-276`). This delays the train→load→infer MVP until after full fine-tune and PEFT. The research report treats loader support as part of the user-facing checkpoint story (`research:461-667`, `research:769-780`). The final synthesis should split MD-Cls `.ckpt` loading earlier and reserve SINCHI formats for later.

F-12. Full fine-tune is described as "mostly a test-only phase" because `freeze_backbone=False` already works from the Phase 3/4 skeleton (`architect-2:174-188`). That hides an implementation change inside the linear-probe phase and removes the independent revert point requested by the ledger (`full-finetune-phase`).

F-13. PEFT is similarly described as already coded in the Phase 4 skeleton and only verified in Phase 6 (`architect-2:199-205`). That violates the phase boundary: PEFT code, dependency imports, and checkpoint hooks must land in the PEFT phase, not silently earlier.

F-14. Several tests rely on pytest's default `tmp_path`/artifact handling (`architect-2:153-170`, `architect-2:219-234`). In this environment, `/tmp` writes are forbidden. The plan must define a repo-local artifact root (for example `tests/_artifacts/` plus a configured `--basetemp`) and enforce it.

F-15. The plan says synthetic checkpoints are sufficient and there is "no need to actually train a SINCHI model" (`architect-2:421-427`). The research report explicitly marks SINCHI checkpoint inner keys as inferred and requiring real-checkpoint confirmation (`research:820-824`, `research:844-846`). Synthetic-only tests can validate code paths but cannot validate the auto-detect contract.

F-16. The DINOv3 model unit test forwards `(2,3,224,224)` (`architect-2:128-136`). That may be legal for HF DINOv3ViT, but it does not validate the SINCHI-derived 518 transform requirement or the shared `Custom_Crop` behavior. At least one integration test must assert the configured DINOv3 dataloader emits 518.

F-17. The lazy-import simulation via `sys.modules["transformers"]=None` (`architect-2:262-267`) is weaker than a true absent-package environment. It can miss importlib/spec behavior and transitive imports. Use a subprocess with an import hook or a clean venv/uv environment that genuinely lacks `transformers` and `peft`.

F-18. The acceptance command requires `pytest-cov` and coverage thresholds across `src` (`architect-2:482-492`). That is a new quality gate, not just a DINOv3 validation gate. It is fine if explicitly accepted as Phase 0 infrastructure, but it should not be the only definition of DINOv3 done.

F-19. The config hash assertion uses `yaml.safe_dump(Munch(...)) == <baseline_hash>` (`architect-2:291-296`). YAML dump ordering and Munch conversion can be version-sensitive. Compare normalized dictionaries and explicit default keys instead.

F-20. The "tests fail first" requirement for the `CustomWeights` bug fix (`architect-2:343-345`) is good, but the plan does not state how this works with atomic phase commits. The final plan should require a regression test commit or CI artifact demonstrating pre-fix failure before the fix commit.

### architect-3 (risk/backwards-compat)

F-21. Phase 3 replaces the research report's direct `requirements.txt` edit with `requirements-dinov3.txt`/extras (`architect-3:41-46`, `architect-3:301-324`). This may be the safer policy, but it diverges from the research plan (`research:138-149`, `research:794-814`) and assumes packaging support that may not exist. The final design must explicitly choose and document the install command.

F-22. The claim that every phase is independently revertible is weakened by the admission that Phases 4–6 all edit `src/algorithms/dinov3_linear.py` (`architect-3:516-523`). Reverting Phase 5 while Phase 6 remains is not independently safe. The rollback DAG should state reverse-order rollback for dependent phases, not single-phase independence.

F-23. Guarding `classification.__init__` with `try/except ImportError: pass` (`architect-3:265-279`) can mask real import bugs inside `dinov3_base` and make `DINOv3CustomWeights` silently absent. Prefer top-level importability with lazy imports inside constructors; if a guard is used, catch only missing optional-dependency errors and log/raise actionable state.

F-24. Byte-equality invariants for source files (`architect-3:382-407`) are over-strict where comments, formatting, or import ordering are harmless, and weak where the plan allows updating hashes in the same phase. Use byte-equality only for files that must never change (`main.py`, `plain.py`, `plain_resnet.py`, `custom.py`), and use behavior tests elsewhere.

F-25. The rollback dry-run requirement in a scratch worktree (`architect-3:88-96`) is not operationalized. It needs exact commands, artifact cleanup, and a repo-local working directory policy; otherwise it becomes a manual ritual that teams skip.

F-26. The CI matrix with `requirements.txt` only and `requirements.txt + requirements-dinov3.txt` (`architect-3:409-417`) is useful but larger than the research scope. It should be staged: Phase 0 establishes baseline CI, Phase 3 adds optional-dep matrix once dependency policy is chosen.

F-27. Phase 7 loader is scheduled after PEFT (`architect-3:79-82`, `architect-3:256-294`), delaying MD-Cls `.ckpt` loading even though it only depends on `DINOv3Classifier`. A safer partial rollout is loader-v1 immediately after linear probe, loader-v2 after PEFT/real SINCHI key inspection.

F-28. The plan relies on `facebook/dinov3-vits16-pretrain-lvd1689m` fitting in ~3GB at batch_size=4 fp16 (`architect-3:306-313`) without a cited measurement. Treat GPU-memory numbers as estimates and add a smoke benchmark or conservative docs.

F-29. Acceptance still uses synthetic SINCHI raw/PEFT checkpoints (`architect-3:492-503`) despite the research report's uncertainty around real SINCHI key layouts (`research:820-824`). A real-checkpoint key-inspection gate is mandatory before claiming SINCHI compatibility.

F-30. The DINOv3 transform risk is identified (`architect-3:160-167`) but left as "decision deferred". The final converged plan cannot leave this unresolved: it needs a concrete transform injection point and regression tests proving ResNet transforms are unchanged.

F-31. The requirements-comment-only Phase 8 (`architect-3:301-324`) risks discoverability failure: users following `pip install -r requirements.txt` then the README DINOv3 command may not have deps. If deps are optional, docs and sample commands must use the optional requirements file explicitly.

## Cross-architect divergences

F-32. Phase 0 ordering diverges. Architect-1 starts with the `CustomWeights` bug fix; architects 2 and 3 start with test scaffolding. The converged answer should land minimal regression scaffolding first, then the bug fix with a pre-fix failing test artifact, because the bug fix is the only behavior-changing edit to shipped code.

F-33. Inference-loader phasing diverges. Architect-1 splits MD-Cls loader and SINCHI loader; architect-2 bundles all inference last; architect-3 delays loader until after PEFT. The better convergence target is architect-1's split, amended with architect-3's fail-loud loader checks and a real SINCHI key-inspection gate.

F-34. Dependency policy diverges. Architect-1 and 2 edit `requirements.txt`; architect-3 uses optional requirements/extras and leaves default runtime unchanged. The final document must choose one policy and give exact install commands for default users, DINOv3 linear/full-FT users, and PEFT users.

F-35. Full-fine-tune and PEFT boundaries diverge. Architect-2 pre-implements branches in the linear-probe phase; architects 1 and 3 keep them as later code deltas. The ledger requires separate phases, so the final synthesis should keep separate implementation commits and tests.

F-36. Test strategy diverges from lightweight to full pytest+coverage+CI. A baseline test harness is required, but coverage thresholds and doctest extraction are optional hardening. The final plan should separate "must pass before merge" from "nice-to-have quality gates".

F-37. Transform handling remains unresolved. The research report requires DINOv3 518 bicubic while preserving `Custom_Crop`; only architect-3 flags the shared-transform risk, and nobody provides the final implementation point. This is a blocker for convergence.

F-38. SINCHI compatibility evidence diverges. Architect-1 requires real checkpoint pre-flight for Phase 6, architect-2 says synthetic is enough, architect-3 mostly uses synthetic. Final plan must require real key inspection before supporting SINCHI raw/PEFT in docs.

F-39. Docs timing diverges. Architect-1/2 put docs last; architect-3 has `docs/dinov3.md` plus usage/README in final phase. The final plan should ship minimal gate docs with the first public DINOv3 capability, then expand docs at the end.

## Consensus assessment

Genuine consensus:
- The integration should be additive: new `DINOv3Classifier`, new `DINOv3Linear`, new mixup utility, new `dinov3_base` inference subpackage, and no `main.py` rewrite.
- Existing ResNet/Plain behavior must be protected by regression tests, especially default-config imports and the `CustomWeights` ResNet-18/50 bug fix.
- CI should not require HF auth, real DINOv3 downloads, CUDA, or PEFT unless a marked optional job is selected.
- Heavy dependencies must be lazy-imported so existing ResNet users do not fail at import time.

Shallow or unresolved consensus:
- All architects repeat the SINCHI PEFT `TaskType.FEATURE_EXTRACTION` and target-module list, but only external validation shows the list matches current HF DINOv3ViT names and still needs runtime validation.
- Everyone wants "bit-identical" ResNet protection, but the exact proof varies between fragile state_dict hashes, behavior snapshots, byte hashes, and import checks.
- Everyone maps the 10-item validation checklist, but real SINCHI checkpoint validation and transform-size validation are not consistently executable.
- Everyone claims phase revertibility, but multi-phase edits to `dinov3_linear.py`, delayed loaders, and hidden PEFT/full-FT branches undermine independent rollback.

## Missing perspectives (what nobody considered)

F-40. License/citation/gated-weight policy: DINOv3 weights are gated and may carry model-card/license obligations. Docs should state required HF access, citation, and redistribution limits for checkpoints derived from DINOv3.

F-41. Artifact metadata/model registry: checkpoints need embedded `model_name`, `image_size`, class names or label map, PEFT config, transform normalization, and source format so inference does not rely on sidecar guesses.

F-42. Non-CUDA execution: plans discuss GPU OOM and HF tokens but not CPU-only/MPS behavior, `precision="16-mixed"` on non-CUDA, or graceful fallbacks when users run DINOv3 on a laptop.

F-43. Distributed/multi-GPU: no plan covers Lightning strategy/DDP, gradient checkpointing, accumulation, or seed consistency across workers for full fine-tune/PEFT.

F-44. Security of checkpoint loading: research uses `weights_only=False`; only architect-1/3 partly mention trust. The final plan needs explicit trusted-source docs and a negative test for corrupt/untrusted checkpoint errors.

F-45. Reproducibility: seeds are mentioned, but no plan covers worker seeding, deterministic transforms, pinned torch/torchvision/Lightning versions, or recording RNG/config in checkpoints.

F-46. User-facing error taxonomy: HF auth failure, missing deps, corrupt checkpoints, hidden-size mismatch, missing `peft_config`, no matching LoRA target modules, and non-CUDA OOM need standardized exception messages.

F-47. Release notes and migration: the `CustomWeights` bug fix and DINOv3 optional install path need a concise migration note; burying them in final docs is not enough.

## Industry comparison

External checks change the PEFT and transform assumptions:

- HuggingFace PEFT TaskType docs state: `FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features for downstream tasks.` URL: https://huggingface.co/docs/peft/en/package_reference/peft_types. This supports `TaskType.FEATURE_EXTRACTION` for wrapping only the DINOv3 backbone while keeping a separate classifier head.
- HuggingFace PEFT `LoraConfig.target_modules` docs state that list entries match exact module names or names that end with a listed string, and if the architecture is unknown "you should specify the target modules manually." URL: https://huggingface.co/docs/peft/en/package_reference/lora. This means the final plan must validate target names against `named_modules()` and fail if no DINOv3 modules match.
- Current HuggingFace Transformers DINOv3ViT source defines split attention projections: `self.k_proj`, `self.v_proj`, `self.q_proj`, `self.o_proj`, and MLP projections `up_proj`, `down_proj`, plus `gate_proj` only in `DINOv3ViTGatedMLP`. URL: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py. Therefore the SINCHI list is plausible for current HF ViT revisions, but not universal across DINO/timm/fused-qkv variants.
- HuggingFace DINOv3ViT image processor source currently defaults `size = {"height": 224, "width": 224}` and the config defaults `image_size = 224`; EoMT-DINOv3 docs say DINOv3 can operate on broader resolutions and custom sizes can be supplied. URLs: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/dinov3_vit/image_processing_dinov3_vit.py and https://huggingface.co/docs/transformers/en/model_doc/eomt_dinov3. The research report's 518 transform comes from SINCHI config, not a universal HF default, so the final plan must make image size config-driven and prove the existing ResNet `Custom_Crop` path is unchanged.

## Per-scope-item verification

<a name="phase-decomposition"></a>
### Scope item: phase-decomposition
All three architects provide non-vacuous phase decompositions and dependency discussions. Quality: deep but divergent. Architect-1 optimizes MVP gates; architect-2 optimizes testability; architect-3 optimizes backwards compatibility. Final converged doc should use baseline scaffolding → bug fix → mixup → linear model/algorithm → loader-v1 → gate → full-FT → PEFT → loader-v2/SINCHI → docs, with explicit dependency/revert rules.

<a name="mixup-utility-phase"></a>
### Scope item: mixup-utility-phase
All three architects cover the mixup port and tests. Quality: deep. Gaps: exact RNG source (`np.random.beta` plus torch noise), no `/tmp` fixture policy, and unclear phase where DINOv3Linear wires mixup. Final doc should land pure utility tests first and add algorithm-level mixup toggle tests when `DINOv3Linear` lands.

<a name="model-and-algorithm-phase"></a>
### Scope item: model-and-algorithm-phase
All three architects cover `DINOv3Classifier` and `DINOv3Linear`. Quality: deep but incomplete. Gaps: optional dependency policy, transform injection point, checkpoint hparams serialization, and no-matching-target errors. Final doc should define lazy imports, 518 dataset transform behavior, and clean linear-probe-only scope.

<a name="full-finetune-phase"></a>
### Scope item: full-finetune-phase
All three architects cover full fine-tune. Quality: mixed. Architect-2 makes it mostly test-only, which is vacuous for phase isolation; architects 1 and 3 better preserve a separate implementation/revert point. Final doc should require unfreeze implementation, dual param groups, grad-flow tests, and memory-safe docs.

<a name="peft-phase"></a>
### Scope item: peft-phase
All three architects cover PEFT. Quality: deep but missing validation. Gaps: target-module validation, PEFT version bounds, embedding `peft_config` into checkpoints, and fail-loud inference when adapter keys are present without config. Final doc should include runtime module inspection, metadata save/load, LoRA/PiSSA tests, and lazy import tests.

<a name="inference-loader-phase"></a>
### Scope item: inference-loader-phase
All three architects cover inference loading. Quality: deep but divergent. Architect-1's split loader-v1/loader-v2 is strongest; architect-3's fail-loud risk register is strongest; architect-2's all-at-end phase is too coupled. Final doc should split MD-Cls `.ckpt` loading from SINCHI raw/PEFT and require real SINCHI key inspection before declaring compatibility.

<a name="config-and-docs-phase"></a>
### Scope item: config-and-docs-phase
All three architects cover config/docs. Quality: non-vacuous but unresolved. Gaps: dependency install policy, timing of minimal Gate-1 docs, and sample image size/model memory evidence. Final doc should ship minimal user warnings with first public DINOv3 phase and full config/docs after code stabilizes.

<a name="bug-fix-phase"></a>
### Scope item: bug-fix-phase
All three architects cover the `CustomWeights(num_layers)` bug. Quality: deep. Gaps: exact failing-test-first workflow and synthetic vs real ResNet-18/50 fixtures. Final doc should make this a standalone phase after baseline tests, with ResNet-18 and default ResNet-50 regression tests.

<a name="regression-protection"></a>
### Scope item: regression-protection
All three architects cover regression protection. Quality: deep but over-brittle in places. Gaps: stable deterministic environment, behavior snapshots vs byte hashes, and optional-dep absent environment. Final doc should combine import/default-config invariants, prediction snapshots, selected file hashes, and clear baseline-refresh policy.

<a name="test-strategy"></a>
### Scope item: test-strategy
All three architects cover the test pyramid. Quality: deep. Gaps: repo-local temp/artifact root, true absent-package tests, real SINCHI checkpoint key inspection, and clear CI/local/nightly split. Final doc should specify exact commands, markers, artifact directory, and no network/CUDA in default CI.

<a name="acceptance-and-validation"></a>
### Scope item: acceptance-and-validation
All three architects map the research report's 10 validation items. Quality: non-vacuous. Gaps: transform-size proof, real SINCHI checkpoint proof, and minimal docs before public gates. Final doc should provide executable commands and expected outputs for all 10, marking which are CI, local CPU, local GPU, and real-artifact gates.

<a name="risk-and-rollback"></a>
### Scope item: risk-and-rollback
All three architects cover risk/rollback. Quality: deep. Gaps: independent revert claims for dependent phases, real operational rollback commands, and dependency policy rollback. Final doc should use reverse-DAG rollback, one merge commit per phase, and fail-loud risk tests for optional deps/checkpoints/transforms.

## Coverage analysis
- Ledger items: 12
- Verified this round (all 3 architects + non-vacuous treatment): 12
- Uncovered: []
- Audit note: `COVERAGE_LOG.jsonl` already contained 12 inquisitor entries at review start; I recomputed proposal coverage from the ledger and proposal anchors, then appended the requested round-1 verification entries again.

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=12/12 UNCOVERED=[] NEW=47
