# Inquisitor review — round 2

## Claim verification (independent spot-checks)

- VERIFIED — Anti-drift inputs read directly. `SCOPE_LEDGER.json` has `convergence_mode: read_only` and four `scope_items`: `source-pipeline`, `target-architecture`, `prior-art-branch`, `dinov3-model-access`. `COVERAGE_LOG.jsonl` already recorded round-1 researcher/inquisitor entries and a round-2 researcher entry for `source-pipeline` before this review.
- VERIFIED — Round reports read directly: all four round-1 researcher reports, the round-1 inquisitor review, and `round_02/researcher-1_report.md`.
- VERIFIED — SOURCE RAW S+ checkpoint. I ran `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/venv/bin/python` with `torch.load(path, map_location="cpu", weights_only=False)` on `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/outputs/baseline_no_augmentations_noisy_same_class_mixup_vits16plus/checkpoints/best_model.pt`. Result: top-level keys include `model_state_dict` and `config`; the state dict has 237 keys with prefix counts `backbone: 235`, `classifier: 2`; first keys begin `backbone.embeddings.*`; final keys include `backbone.norm.*`, `classifier.weight`, `classifier.bias`; `classifier.weight` is `(36, 384)` and `classifier.bias` is `(36,)`; `config.model.name` is `facebook/dinov3-vits16plus-pretrain-lvd1689m`; `config.model.num_classes` is `36`; `config.data.image_size` is `518`.
- VERIFIED — PRIOR-ART loader stubs. `git show docs/dinov3-integration-plan:src/models/classification/dinov3_base/base_classifier.py` shows `_extract_state_dict` supports only `state_dict` keys with `net.` in v1 and raises `NotImplementedError("SINCHI checkpoint flavours land in Phase 8")` for both `model_state_dict` + `backbone.` and keys containing `base_model.model.`.
- VERIFIED — PRIOR-ART module naming. `git show docs/dinov3-integration-plan:src/models/dinov3.py` shows `DINOv3Classifier` stores the HF backbone as `self.feature` and uses it in `forward()`. Therefore SINCHI `backbone.*` keys cannot load unchanged into that prior-art module layout; they need remapping to `feature.*` or a wrapper whose member is named `backbone`.
- VERIFIED — PRIOR-ART dependency floor. `git show docs/dinov3-integration-plan:requirements-dinov3.txt` pins `transformers>=4.43,<5`, `peft>=0.10.0,<0.14.0`, and `huggingface_hub>=0.20,<1`.
- VERIFIED — MODEL-ACCESS Transformers floor. Fetching `https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/dinov3` returned the DINOv3 docs, including “added to Hugging Face Transformers on 2025-08-14” and DINOv3 classes. Fetching the same path under `v4.55.0` returned HTTP 404. This confirms `transformers>=4.56.0` as the research-supported minimum.
- VERIFIED — MODEL-ACCESS repo IDs and gating. HF API responses for `facebook/dinov3-vits16plus-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, and `facebook/dinov3-vith16plus-pretrain-lvd1689m` all report `gated: "manual"`, `library_name: "transformers"`, `pipeline_tag: "image-feature-extraction"`, `license_name: "dinov3-license"`, and `architectures: ["DINOv3ViTModel"]`. The model card technical specifications list S+ embedding dimension 384, B 768, and H+ 1280.
- VERIFIED — TARGET additive family pattern. `src/models/classification/timm_base/base_classifier.py` defines a self-contained `TIMM_BaseClassifierInference` with class attributes, `_load_model`, transform setup, single-image classification, batch classification, and unchanged result schema. `src/models/classification/__init__.py` imports family packages additively. `src/models/classification/speciesnet_base/__init__.py` catches `ImportError`, confirming a guarded optional-dependency precedent.

## Contradictions

- REPEAT-FROM-ROUND-1 — The round-1 inquisitor review says researcher-1/2 were placeholders, researcher-3 was absent, and researcher-4 was a draft. Current round-1 report files are complete and substantive. Resolution: treat the round-1 inquisitor review as a premature snapshot, not as current coverage truth.
- REPEAT-FROM-ROUND-1 — Prior-art `requirements-dinov3.txt` says `transformers>=4.43,<5`, but DINOv3 documentation is present at Transformers v4.56.0 and absent at v4.55.0. Resolution: implementation plan must raise the optional DINOv3 dependency floor to `>=4.56.0`.
- REPEAT-FROM-ROUND-1/2 — SINCHI checkpoints use `backbone.*`, while prior-art `DINOv3Classifier` names the backbone `feature`. Resolution: loader implementation must remap keys or preserve a `backbone` member. This is an implementation-plan decision, not a remaining research gap.
- REPEAT-FROM-ROUND-1 — SINCHI uses image size 518, while HF docs/model cards describe default 224 and larger dimensions only when multiples of patch size 16. Resolution: the port plan must choose and test the preprocessing policy; no additional research is needed to state the tradeoff.
- REPEAT-FROM-ROUND-1 → CLOSED — Source `extract_embeddings_all_models.py` listed B/L/7B, not the requested S+/B/H+ trio. Round-2 checkpoint evidence plus model-access research now establish the requested S+/B/H+ IDs and classifier dimensions. Updating the all-size registry is plan scope.

## Round-1→Round-2 gap status

- CLOSED — Round-1 report completeness gaps are closed by the current round-1 reports and the round-2 empirical source-pipeline report.
- CLOSED — Real SINCHI checkpoint key evidence is closed by round 2 and independently spot-checked on RAW S+.
- CLOSED — Model ID, gating, embedding dimension, CLS-token, HF cache/download, and Transformers-minimum research gaps are covered by researcher-4 and independent web checks.
- CLOSED — Prior-art Phase-8 SINCHI-loader gap is now evidence-backed: the prior-art stub location is known and the raw/PEFT checkpoint layouts are documented. Writing the loader is implementation scope.
- PLAN-SCOPE — Embedding file format to adopt, exact save-embeddings API, cached-embedding training entrypoint, 518-vs-512/224 preprocessing policy, prior-art reuse strategy, and dependency-file edits are design/implementation decisions. They do not block research convergence.

<a name="source-pipeline"></a>
### Sub-area: source-pipeline

Coverage: PASS. Round-1 researcher-1 inventories SINCHI embedding extraction, per-image pickle schema, cached-embedding dataset, linear-probe training branch, checkpoint format, PEFT path, configs, and docs. Round-2 researcher-1 supplies empirical checkpoint and embedding-schema closure. Independent RAW S+ `torch.load` verified the load-bearing checkpoint facts. Residual items are plan-scope implementation choices, not research gaps.

<a name="target-architecture"></a>
### Sub-area: target-architecture

Coverage: PASS. Round-1 researcher-2 covers the current MegaDetector-Classifier inference/training surfaces, additive classifier families, TIMM pattern, SpeciesNet optional-import precedent, registry-based training path, do-not-break surfaces, and test/dependency fragility. Independent views confirmed the TIMM additive family and guarded SpeciesNet import. Residual items are API and test-plan decisions.

<a name="prior-art-branch"></a>
### Sub-area: prior-art-branch

Coverage: PASS. Round-1 researcher-3 covers the `docs/dinov3-integration-plan` branch, P0-P5 implementation state, P6-P9 gaps, reusable files/tests, and reconciliation strategy. Independent `git show` checks confirmed the Phase-8 SINCHI loader stubs, `self.feature` backbone name, and too-low Transformers floor. Residual items are implementation/rebase decisions.

<a name="dinov3-model-access"></a>
### Sub-area: dinov3-model-access

Coverage: PASS. Round-1 researcher-4 covers exact HF repo IDs, S+/B/H+ dimensions, manual gating, license/auth requirements, Transformers minimum, CLS/register-token semantics, auto-download/cache mechanics, and README-ready access steps. Independent web checks confirmed v4.56.0 DINOv3 docs, v4.55.0 absence, all three target repo IDs as manual-gated, and the S+/B/H+ dimension table. No unresolved research gap remains.

## Coverage analysis

- Ledger items: 4
- Verified this round: 4
- Uncovered: []
- New unresolved research gaps this round: 0
- Scope note: coverage was checked against the four ledger `scope_items` and anchors in this review. Repeat plan-scope decisions are intentionally not counted as research gaps under the read-only convergence contract.

STATUS: CONVERGED  SCOPE_CHECK=PASS  COVERED=4/4  UNCOVERED=[]          NEW=0
