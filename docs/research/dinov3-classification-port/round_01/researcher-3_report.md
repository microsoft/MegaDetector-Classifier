<a name="prior-art-branch"></a>
## Sub-area: prior-art-branch

## Key Findings

1. **[NEW] The prior-art design converged and explicitly prioritized additive integration over replacement.** The final design states it converged after three rounds with `CONVERGED SCOPE_CHECK=PASS COVERED=12/12 NEW=0`, and its load-bearing invariant is unchanged `configs/config.yaml`, unchanged runtime `requirements.txt`, and bit-identical ResNet behavior for the existing pipeline (`docs/dinov3-integration-plan:docs/design/dinov3-integration/final.md:3-5`, `:26-32`). Confidence: HIGH.

2. **[NEW] The design phase plan is P0-P9, but the branch only implements through P5.** P0-P5 cover regression scaffolding, `CustomWeights` ResNet bug fix, mixup utility, optional DINOv3 deps + 518 crop, DINOv3 frozen linear-probe core, and MD-Cls-only loader v1; P6 full fine-tune, P7 PEFT/LoRA/PiSSA, P8 SINCHI raw/PEFT loaders, and P9 configs/docs remain later phases (`final.md:57-72`, `:95-106`). Confidence: HIGH.

3. **[NEW] DINOv3 slots into both the training hierarchy and the PytorchWildlife-style classification hierarchy.** Training adds `src/models/dinov3.py::DINOv3Classifier` and `src/algorithms/dinov3_linear.py::DINOv3Linear`; inference adds `src/models/classification/dinov3_base/{base_classifier,custom_weights}.py` and guarded exports from `src/models/classification/__init__.py` (`final.md:248-280`, `:287-326`, `:890-907`). Confidence: HIGH.

4. **[NEW] Implemented model loading is generic-HF by `model_name`, not a named Amazon/small-plus/base/huge-plus product surface.** `DINOv3Classifier` calls `AutoModel.from_pretrained(model_name)` and infers `embedding_dim` from `config.hidden_size` (`src/models/dinov3.py:40-72`). `DINOv3CustomWeights` defaults to `facebook/dinov3-vits16-pretrain-lvd1689m` and accepts `model_name`/`embedding_dim`, but no registry or alias exists for `dinov3-vits16plus`, `dinov3-vitb16`, or `dinov3-vith16plus`; grep found Amazon only in existing ResNet/TIMM docs/loaders (`src/models/classification/dinov3_base/custom_weights.py:23-30`; `README.md:133-141`). Confidence: HIGH.

5. **[NEW] Implemented training is frozen-backbone linear probing over images; full fine-tune and PEFT are explicit stubs.** `DINOv3Classifier(freeze_backbone=False)` raises `NotImplementedError("Full FT lands in Phase 6")`; `DINOv3Linear(peft.enable=True)` raises `NotImplementedError("PEFT lands in Phase 7")` (`src/models/dinov3.py:24-38`; `src/algorithms/dinov3_linear.py:46-52`, `:108-110`; tests at `tests/integration/test_dinov3_linear_probe.py:62-92`). Confidence: HIGH.

6. **[NEW] Implemented inference returns logits only; embedding extraction/saving is absent.** `DINOv3Classifier.forward` takes CLS token features then immediately returns `self.classifier(feats)`; `DINOv3BaseInference.forward` returns `self.net(x)`. Branch grep for save/extract embeddings outside design/implementation logs found only `src/utils/mixup.py`, not an extraction API (`src/models/dinov3.py:79-83`; `src/models/classification/dinov3_base/base_classifier.py:70-71`). Confidence: HIGH.

7. **[NEW] P5 loader supports only the MD-Cls Lightning checkpoint flavour and intentionally rejects SINCHI flavours until Phase 8.** `_extract_state_dict` handles `ckpt['state_dict']` keys prefixed with `net.`, raises `NotImplementedError("SINCHI checkpoint flavours land in Phase 8")` for SINCHI raw and PEFT patterns, and errors on unknown formats (`src/models/classification/dinov3_base/base_classifier.py:1-5`, `:17-56`; `tests/unit/test_extract_state_dict_v1.py:8-37`). Phase 8 was intended to add real-evidence-driven SINCHI raw/PEFT key remaps, but `sinchi_key_layouts.md` is absent from the branch (`final.md:511-570`). Confidence: HIGH.

8. **[NEW] Verification artifacts show P0-P5 converged under mocked/default tests, not real HF/GPU E2E.** P0: `16 passed, 2 deselected, 1 xfailed`; P1/P2/P3: `26 passed, 2 deselected`; P4: `36 passed, 2 deselected, 1 warning`; P5: `44 passed, 3 deselected, 1 warning`, all with `STATUS: CONVERGED SCOPE_CHECK=PASS` (`docs/implement/dinov3-p0/round_02/verifier-pytest.txt:30`; `docs/implement/dinov3-p1p2p3/round_02/verifier-pytest.txt:37`; `docs/implement/dinov3-p4/round_02/verifier-pytest.txt:52`; `docs/implement/dinov3-p5/round_02/verifier-pytest.txt:60`; corresponding inquisitor reports). `pyproject.toml` excludes `isolated_env`, `requires_hf`, `gpu`, and `requires_dinov3` by default, and the design says real-HF/GPU is local nightly/manual (`pyproject.toml:4-16`; `final.md:733-744`). Confidence: HIGH.

## Prior-art inventory table

| File / group | Purpose | Implemented? | Keep / merge / supersede |
|---|---|---:|---|
| `docs/design/dinov3-integration/final.md` | Converged P0-P9 design, invariants, phase DAG, risk/rollback, acceptance commands. | Design only; P0-P5 later implemented. | **Keep as prior art; supersede as current plan** because new requirements add cached embeddings, named sizes, and auto-check/download that the old plan omits or marks out of scope. |
| `docs/design/dinov3-integration/round_01..03/*`, `SCOPE_LEDGER.json`, `COVERAGE_LOG.jsonl`, `CONVERGED.sentinel` | Design provenance and convergence evidence. | Complete design records. | **Keep/archive**, not needed in production merge except as audit history. |
| `docs/dinov3_experimental.md` | Minimal P5 docs: install, HF login, memory, image size, MVP scope. | Implemented. | **Merge then expand** into README/full docs; it already states login but not auto-check/download or all sizes. |
| `requirements-dinov3.txt` | Optional DINOv3 deps: `transformers`, `peft`, `huggingface_hub`. | Implemented. | **Merge**; revalidate version floor/ceiling with researcher-4 model-access findings. |
| `requirements.txt` comment | Points users to optional DINOv3 requirements without adding runtime deps. | Implemented. | **Merge** if invariant tests still pass. |
| `requirements-dev.txt`, `pyproject.toml` | Test dependencies and pytest marker policy. | Implemented. | **Merge selectively** if absent on current branch; keep marker exclusions explicit. |
| `src/models/dinov3.py` | `DINOv3Classifier`; lazy HF import; frozen feature extractor; linear head. | Implemented P4 MVP. | **Merge and extend** with explicit feature-extraction method / size registry; keep lazy import and auth error handling. |
| `src/algorithms/dinov3_linear.py` | Lightning linear-probe algorithm over frozen DINOv3. | Implemented P4 MVP; P6/P7 stubs. | **Merge with fixes/extensions**; adapt for cached embeddings and all sizes. Fix mixup contract before enabling `use_mixup`. |
| `src/models/classification/dinov3_base/__init__.py` | Re-export inference loaders. | Implemented. | **Merge**. |
| `src/models/classification/dinov3_base/base_classifier.py` | `DINOv3BaseInference`, v1 `_extract_state_dict`. | Implemented for MD-Cls only. | **Merge then extend** for SINCHI/Amazon checkpoint flavours if required. |
| `src/models/classification/dinov3_base/custom_weights.py` | Local `.ckpt` DINOv3 inference loader. | Implemented for own MD-Cls `.ckpt`. | **Merge then extend** for named variants and auto-check/download if product surface requires it. |
| `src/models/classification/__init__.py` | Guarded DINOv3 import + actionable optional-deps `ImportError`. | Implemented. | **Merge**; preserves existing import behavior. |
| `src/datasets/custom.py` | Adds `image_size=224` kwarg and `_build_data_transforms`; DINOv3 uses 518. | Implemented. | **Merge**; core non-breaking transform hook. |
| `src/models/classification/resnet_base/custom_weights.py` | Adds `num_layers` kwarg to custom ResNet loader. | Implemented P1 bug fix. | **Merge only if still needed**; test both ResNet-18 and ResNet-50. |
| `src/utils/mixup.py` | SINCHI same-class embedding mixup utilities. | Implemented utility only. | **Merge if cached-embedding training uses it**; otherwise keep separate. Current `DINOv3Linear.training_step` expects 4 return values while utility returns 2. |
| `src/models/__init__.py`, `src/algorithms/__init__.py` | Re-export DINOv3 model/algorithm. | Implemented. | **Merge**, but ensure no eager HF import. |
| `tests/unit/test_dinov3_model.py` | DINOv3Classifier unit tests with HF stub. | Implemented. | **Merge**. |
| `tests/integration/test_dinov3_linear_probe.py` | Linear-probe forward/training-step and stub assertions. | Implemented. | **Merge and extend** for cached embeddings/all sizes. |
| `tests/integration/test_dinov3_dataloader_emits_518.py` | 518 transform test. | Implemented. | **Merge**. |
| `tests/integration/test_dinov3_custom_weights_mdckpt.py` | P5 MD-Cls checkpoint loader tests. | Implemented. | **Merge**. |
| `tests/regression/test_dinov3_absent_actionable_error.py` | Optional deps absent -> actionable ImportError. | Implemented. | **Merge**. |
| `tests/unit/test_extract_state_dict_v1.py` | MD-Cls extraction and SINCHI-Phase-8 stub tests. | Implemented. | **Merge; update when Phase 8 is replaced by real support.** |
| `tests/regression/test_invariants.py`, `test_resnet_baseline_predictions.py`, `test_custom_crop_default_is_224.py`, `test_customweights_num_layers.py` | Existing pipeline protection. | Implemented. | **Merge and rerun on current main-derived branch.** |
| `docs/implement/dinov3-p0/**` | P0 implementation provenance: invariants, rollback, baselines. | Complete/converged. | **Keep as audit source; merge tests/scripts not raw logs unless docs policy wants provenance.** |
| `docs/implement/dinov3-p1p2p3/**` | P1 bug fix, P2 mixup, P3 optional deps/crop. | Complete/converged. | **Keep as audit source; merge code/tests.** |
| `docs/implement/dinov3-p4/**` | P4 model/algorithm implementation records. | Complete/converged. | **Keep as audit source; merge code/tests.** |
| `docs/implement/dinov3-p5/**` | P5 loader/docs implementation records. | Complete/converged. | **Keep as audit source; merge code/tests/docs stub.** |
| `configs/config_dinov3_{linear,finetune,lora}.yaml`, `docs/dinov3.md`, README DINOv3 section, `sinchi_key_layouts.md` | Planned P8/P9 artifacts. | **Not present** on branch. | **Create in new work**, informed by prior plan and researcher-1/4 findings. |

## Phase plan recovered

- **P0 — Test scaffolding + behavior baseline.** Adds regression invariants, rollback dry run, fixtures, and pytest config; purpose is to prove existing ResNet/default behavior remains stable before DINOv3 lands (`final.md:61`, `:95-97`). Implemented and converged.
- **P1 — `CustomWeights` `num_layers` bug fix.** Adds a defaulted `num_layers` kwarg to ResNet custom weights so ResNet-18 and ResNet-50 loading are both testable without changing legacy call signatures (`final.md:62`; diff shows `num_layers: int = 50`). Implemented and converged.
- **P2 — Mixup utility port.** Adds `src/utils/mixup.py` and tests as a leaf utility before callers (`final.md:63`, `:98`). Implemented and converged; caller mismatch remains if `use_mixup=True`.
- **P3 — Optional deps + `Custom_Crop(image_size)`.** Adds `requirements-dinov3.txt`; keeps runtime deps unchanged except a comment; adds `Custom_Crop(conf, image_size=224)` with DINOv3 using 518 (`final.md:64`, `:220-240`; `src/datasets/custom.py:267-279`). Implemented and converged.
- **P4 — DINOv3 model + linear-probe MVP core.** Adds lazy-import `DINOv3Classifier` and `DINOv3Linear`; full-FT and PEFT intentionally raise `NotImplementedError`; real-HF tests are gated on `HF_TOKEN` (`final.md:248-280`). Implemented and converged.
- **P5 — Loader v1 + minimal docs.** Adds `dinov3_base` inference package, MD-Cls-only checkpoint loader, guarded classification import, and `docs/dinov3_experimental.md`; SINCHI flavours raise until Phase 8 (`final.md:287-330`; `docs/dinov3_experimental.md:42-54`). Implemented and converged.
- **Phase 8 intent.** Add `docs/design/dinov3-integration/sinchi_key_layouts.md` based on at least one real SINCHI raw `.pt` and one real SINCHI PEFT `.pt`; extend `_extract_state_dict` for `model_state_dict` + `backbone.*` and `backbone.base_model.model.*`; add synthetic CI tests plus local real-checkpoint acceptance (`final.md:511-570`). It does **not** exist on the branch; current code raises `NotImplementedError` for those flavours.
- **Beyond P5.** P6 full fine-tune, P7 PEFT/LoRA/PiSSA, and P9 full sample configs/docs/README are planned but not implemented (`final.md:68-72`, `:583-623`; `docs/dinov3_experimental.md:3-5`, `:53-54`).

## Gap analysis vs the 4 requirements

1. **Don't break existing pipeline.** Prior art mostly satisfies this as a foundation: lazy optional deps, unchanged runtime requirements except a comment, guarded classification import, 224 default crop, ResNet identity/baseline tests, and P0-P5 convergence evidence. Gap: those results are from the prior branch state; any port plan must rerun the suite after rebasing/cherry-picking onto the current branch.

2. **Inference loads Amazon model with DINOv3 small-plus/base/huge-plus.** Gap. Prior art exposes generic `model_name` loading and one default `facebook/dinov3-vits16-pretrain-lvd1689m`; it does not define an Amazon DINOv3 loader, a model-size registry, or explicit small-plus/base/huge-plus aliases. Tests use `model_name="stub"`, not real HF/Amazon variants.

3. **Function: inference + save embeddings.** Gap. Prior art has inference-to-logits only. It extracts CLS token features internally but does not expose `extract_embeddings`, `save_embeddings`, file formats, metadata, batching, or resume semantics.

4. **2-stage workflow images → embeddings → finetune; all sizes; auto check+download; README explains login.** Major gap. Prior art trains linear probe directly from images through a frozen backbone, not cached embeddings. Full fine-tune/PEFT are future stubs, all-size coverage is not explicit, auto-download via CLI is explicitly out of scope in the old plan (`final.md:831-837`), and README full docs were deferred to P9 and are absent. `docs/dinov3_experimental.md` does explain `huggingface-cli login`, but not the new auto-check/download workflow.

5. **Port from SINCHI_DINOV3 specifically.** Gap. Prior art planned SINCHI checkpoint loaders in Phase 8 but required real key-layout evidence and deferred if absent. It did not port SINCHI embedding extraction scripts or cached-embedding training entrypoints; those must come from researcher-1's `source-pipeline` inventory.

## Reconciliation recommendation

**Recommendation: rebase/finish the prior-art branch, not supersede it wholesale and not blindly merge it.** The P0-P5 implementation already solves the safest parts of integration: optional dependencies, guarded imports, 518 transforms, frozen DINOv3 model/linear probe, MD-Cls checkpoint loading, and regression protection. Reusing those files reduces duplicated design risk. However, the old design conflicts with the current user requirements where it omits cached embeddings, explicit small-plus/base/huge-plus support, Amazon DINOv3 inference surface, and auto check/download.

Concrete carry-over set:

- **Carry over directly after rebase/test:** `requirements-dinov3.txt`, `src/datasets/custom.py` image-size hook, `src/models/classification/__init__.py` guarded import pattern, `src/models/classification/dinov3_base/__init__.py`, P0/P3/P5 absent-deps and 518-transform tests.
- **Carry over but extend:** `src/models/dinov3.py` (add `extract_features`/embedding API and named-size registry), `src/algorithms/dinov3_linear.py` (support cached embeddings or new embedding dataset path; fix mixup return contract before enabling), `src/models/classification/dinov3_base/base_classifier.py` and `custom_weights.py` (add size aliases, Amazon/auto-download surface, SINCHI loaders as required), `docs/dinov3_experimental.md` (expand into README/docs).
- **Carry over selectively:** `src/utils/mixup.py`, `src/models/__init__.py`, `src/algorithms/__init__.py`, `src/models/classification/resnet_base/custom_weights.py`, `requirements-dev.txt`, `pyproject.toml`; verify against current branch conventions before merging.
- **Carry over test suite and expand:** all `tests/*dinov3*`, `tests/unit/test_extract_state_dict_v1.py`, `tests/regression/test_invariants.py`, ResNet baseline/customweights tests; add new tests for the four current requirements.
- **Use as provenance, not implementation target:** `docs/design/dinov3-integration/final.md`, architect/inquisitor rounds, and `docs/implement/dinov3-p*/**` reports. The new port plan should cite and supersede them.

Operationally, the final port plan should stage work as: (1) rebase/cherry-pick P0-P5 safety net and MVP code; (2) add DINOv3 model access registry from researcher-4; (3) add source-pipeline-compatible embedding extraction/save API from researcher-1; (4) add cached-embedding linear probe and finetune workflow; (5) update README/login/auto-download docs; (6) rerun prior invariant suite plus new all-size/embedding workflow tests.

## Sources

- `git -C /shared/home/aahernandez/MegaDetector-Classifier log --oneline docs/dinov3-integration-plan` — branch chronology through `294561f impl: dinov3-p5 CONVERGED`.
- `git -C /shared/home/aahernandez/MegaDetector-Classifier diff --stat main..docs/dinov3-integration-plan` — 101 files changed, including P0-P5 code/tests/docs.
- `docs/dinov3-integration-plan:docs/design/dinov3-integration/final.md:3-5`, `:57-72`, `:95-106`, `:248-330`, `:511-570`, `:583-623`, `:733-744`, `:831-837`, `:890-907`.
- `docs/dinov3-integration-plan:src/models/dinov3.py:1-83`.
- `docs/dinov3-integration-plan:src/algorithms/dinov3_linear.py:1-110`.
- `docs/dinov3-integration-plan:src/models/classification/dinov3_base/base_classifier.py:1-71`.
- `docs/dinov3-integration-plan:src/models/classification/dinov3_base/custom_weights.py:1-77`.
- `docs/dinov3-integration-plan:docs/dinov3_experimental.md:1-59`.
- `docs/dinov3-integration-plan:tests/unit/test_dinov3_model.py:9-46`; `tests/integration/test_dinov3_linear_probe.py:10-92`; `tests/integration/test_dinov3_custom_weights_mdckpt.py:22-86`; `tests/integration/test_dinov3_dataloader_emits_518.py:8-21`; `tests/regression/test_dinov3_absent_actionable_error.py:25-35`; `tests/unit/test_extract_state_dict_v1.py:8-37`.
- `docs/dinov3-integration-plan:docs/implement/dinov3-p0/round_02/verifier-pytest.txt:30`; `docs/implement/dinov3-p1p2p3/round_02/verifier-pytest.txt:37`; `docs/implement/dinov3-p4/round_02/verifier-pytest.txt:52`; `docs/implement/dinov3-p5/round_02/verifier-pytest.txt:60`.

## Cross-Area Connections

- **source-pipeline (researcher-1):** Prior art does not port SINCHI embedding extraction or cached-embedding training. Researcher-1's function/file inventory should define the exact embedding file format, metadata, and training entrypoints to add on top of P0-P5.
- **target-architecture (researcher-2):** Prior art already proposes the target extension points: `src/models/dinov3.py`, `src/algorithms/dinov3_linear.py`, `src/models/classification/dinov3_base/`, `Custom_Crop(image_size=518)`, and guarded classification imports. Researcher-2 should verify these still match current main-derived architecture.
- **dinov3-model-access (researcher-4):** Prior art has generic HF auth handling but lacks exact small-plus/base/huge-plus repo IDs, embedding dimensions, download checks, and README wording. Researcher-4 should supply the registry and access rules that supersede the old default `facebook/dinov3-vits16-pretrain-lvd1689m`.

## Gaps

1. No explicit Amazon DINOv3 loader or all-size registry.
2. No embedding extraction/saving API or persisted embedding schema.
3. No two-stage cached-embedding training workflow.
4. P6/P7/P8/P9 are unimplemented; Phase 8 SINCHI loaders are only planned and currently raise.
5. Old plan says CLI auto-download is out of scope; current requirement needs auto check+download plus README login instructions.

## Confidence

| Finding | Factual accuracy | Completeness | Implementation correctness |
|---|---|---|---|
| 1 | HIGH — verified in `final.md` and design inquisitor verdict. | HIGH — covers design status and non-break invariant. | N/A — read-only. |
| 2 | HIGH — verified in phase index and source/test presence/absence. | HIGH for P0-P9 status. | N/A — read-only. |
| 3 | HIGH — verified in final file map and source files. | HIGH for requested hierarchy/entrypoints. | N/A — read-only. |
| 4 | HIGH — verified by source and grep. | MEDIUM — exact model IDs/dims deferred to researcher-4. | N/A — read-only. |
| 5 | HIGH — verified by source and tests. | HIGH for P4/P6/P7 status. | N/A — read-only. |
| 6 | HIGH — verified by source and grep. | HIGH for prior branch; source-pipeline may define desired API. | N/A — read-only. |
| 7 | HIGH — verified by source, tests, and final Phase 8 plan. | HIGH for Phase 8 intent. | N/A — read-only. |
| 8 | HIGH — verified in verifier artifacts. | MEDIUM — did not rerun tests because this is read-only research. | N/A — read-only. |

STATUS: NEW=8 GAPS=5
