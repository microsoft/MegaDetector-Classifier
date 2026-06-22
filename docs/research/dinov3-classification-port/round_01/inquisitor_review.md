# Inquisitor review — round 1

## Claim verification (across researchers)

Round 1 critiques are all tagged NEW.

### VERIFIED
- Anti-drift inputs were read directly. The ledger has four `scope_items[]`: `source-pipeline`, `target-architecture`, `prior-art-branch`, `dinov3-model-access`. Initial `COVERAGE_LOG.jsonl` was empty.
- Reports read directly from `round_01`: researcher-1 and researcher-2 are placeholders; researcher-4 is a draft; `researcher-3_report.md` is absent from the directory listing.
- Source wrapper exists: `DINOv3LinearClassifier` is defined in `SINCHI_DINOV3/models/dinov3_model.py:15`, loads `AutoModel`/`AutoImageProcessor` at lines 47-53, infers `hidden_size` at lines 63-84, freezes the backbone at lines 86-101, and exposes `get_embeddings` at lines 148-173.
- Source embedding extraction exists: `scripts/extract_embeddings.py:47-124` writes one pickle file per image; file names replace path separators and append `.pkl` at lines 32-44; each pickle stores `embedding`, `label`, and `image_path` at lines 108-118. Train/val subdirectories are created at lines 200 and 234.
- Source cached-embedding training exists: `scripts/train.py:138-171` switches datasets when `experiment.use_cached_embeddings` is true; training and validation bypass the backbone and call `model.classifier(embeddings)` at lines 346-368 and 470-474.
- Source checkpoint format is `model_state_dict`: `scripts/train.py:600-611` saves `model.state_dict()` under `model_state_dict`; because the model members are `backbone` and `classifier`, raw keys are expected to start with those prefixes. PEFT resume loads the same key in `scripts/train_peft.py:263-270`.
- Target current architecture has a weak abstract base: `src/models/classification/base_classifier.py:11-29` defines `BaseClassifierInference`, but methods are `pass`, not enforced abstract methods. Existing inference families implement concrete methods in `resnet_base/base_classifier.py:105-193`, `timm_base/base_classifier.py:23-211`, and `speciesnet_base/speciesnet_tf.py`.
- Target current default pipeline is still Lightning/ResNet-oriented: `main.py:107-177` builds `dataset = datasets.__dict__[conf.dataset_name]` and `learner = algorithms.__dict__[conf.algorithm]`, then fits/evaluates with `Trainer`; `configs/config.yaml:24-27` defaults to `PlainResNetClassifier`.
- Target current main does not have active DINOv3 Python modules: `src/models/classification/dinov3_base/` contains only `__pycache__`; `requirements.txt` has no `transformers`, `huggingface-hub`, or `peft` runtime lines.
- Prior-art branch exists and is substantial. `git show docs/dinov3-integration-plan:src/models/dinov3.py` contains `DINOv3Classifier` with lazy `transformers` import and frozen-backbone guard at lines 16-83. `src/algorithms/dinov3_linear.py` contains `DINOv3Linear` at lines 19-110.
- Prior-art branch loader v1 explicitly does not support SINCHI checkpoint flavours: `src/models/classification/dinov3_base/base_classifier.py:42-52` raises `NotImplementedError("SINCHI checkpoint flavours land in Phase 8")` for `model_state_dict` keys with `backbone.` or `base_model.model.`.
- Prior-art design identifies Phase 8 as the SINCHI loader work: `docs/design/dinov3-integration/final.md:511-568` requires real key-layout evidence and remapping for raw and PEFT SINCHI checkpoints. The P5 verifier log reports `44 passed, 3 deselected, 1 warning in 14.98s` at `docs/implement/dinov3-p5/round_02/verifier-pytest.txt:1-60`.
- Official Hugging Face model API verifies these repo ids and gated/license state: `facebook/dinov3-vits16plus-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, and `facebook/dinov3-vith16plus-pretrain-lvd1689m` all return `gated:"manual"`, `pipeline_tag:"image-feature-extraction"`, `library_name:"transformers"`, `license_name:"dinov3-license"`, and `license_link:"https://ai.meta.com/resources/models-and-libraries/dinov3-license"`.
- Official Hugging Face model cards verify embedding dimensions: ViT-S+ = 384, ViT-B = 768, ViT-H+ = 1280; all are patch size 16 with 4 register tokens. The cards also state 224x224 examples and allow larger images only when shapes are multiples of patch size 16; otherwise the model crops to the closest smaller multiple.
- Official Transformers docs verify support starts no earlier than v4.56.0: `https://huggingface.co/docs/transformers/v4.56.0/model_doc/dinov3` says DINOv3 was added to Transformers on 2025-08-14, while `v4.55.0/model_doc/dinov3` returned 404.

### NEW findings / gaps / unverified claims
1. NEW-F01 — Researcher-1 report is a placeholder but claims `NEW=0 GAPS=0`; this is wrong because it does not inventory the source pipeline.
2. NEW-F02 — Researcher-2 report is a placeholder and correctly has `GAPS=1`, so target architecture is not covered by the researcher.
3. NEW-F03 — Researcher-3 report is absent, so prior-art branch coverage by a researcher is missing.
4. NEW-F04 — Researcher-4 report is a draft with `NEW=1 GAPS=1`; model access is not covered by the researcher.
5. NEW-F05 — Source `extract_embeddings_all_models.py:13-19` lists ViT-B, ViT-L, and ViT-7B, while the ledger requires small-plus, base, and huge-plus. The all-model-size mapping for the port is unresolved.
6. NEW-F06 — The source embedding file contract is per-image pickle dicts under `train/` and `val/`; no target report defines whether MegaDetector-Classifier should preserve that format or use a new batched format.
7. NEW-F07 — The required inference-plus-save-embeddings function is not designed. Source extraction is a standalone script, and the target `BaseClassifierInference` contract currently has only `single_image_classification` / `batch_image_classification` stubs.
8. NEW-F08 — The two-stage workflow is not designed for the target. Source training can consume cached embeddings directly, but target current main/prior-art DINOv3Linear does not implement an embedding dataset/training mode.
9. NEW-F09 — Image-size semantics are unresolved: source config uses 518, prior art tests mention 518, but official HF docs/cards use 224 by default and require larger dimensions to be multiples of 16; 518 would be cropped.
10. NEW-F10 — Prior-art branch Phase 5 is insufficient for the requested port because it intentionally rejects SINCHI raw and PEFT checkpoint flavours until Phase 8.
11. NEW-F11 — Prior-art `requirements-dinov3.txt` pins `transformers>=4.43,<5`, but official DINOv3 docs appear at v4.56.0 and v4.55.0 is absent. Minimum version should be raised or otherwise justified.
12. NEW-F12 — Auto-download/access mechanics are not covered: the official API reports manual gating, so implementation needs explicit login/access failure handling and README language.
13. NEW-F13 — Embedding semantics need an explicit decision: source uses the CLS token from `last_hidden_state[:, 0, :]`, while HF examples also expose `pooler_output`; parity with SINCHI likely means preserving CLS unless tested otherwise.
14. NEW-F14 — Real SINCHI checkpoint key evidence was not gathered in this round. Prior art itself requires loading at least one raw and one PEFT SINCHI checkpoint and documenting key layouts before implementing loader v2.
15. NEW-F15 — Automated `scope_check.sh` fails with rc=2 because the ledger has an empty/missing required `.repo_root`; the inquisitor cannot fix the ledger under this role.

## Contradictions
- Researcher-1 body says "Research in progress" throughout, but its final line reports `NEW=0 GAPS=0`. Resolution: treat as not covered; NEW-F01.
- Researcher-2 body says "Pending" throughout and final line reports `GAPS=1`. Resolution: not covered by researcher; NEW-F02.
- Researcher-3 was required by the prompt but no file exists in `round_01`. Resolution: prior-art branch research is missing; NEW-F03.
- Researcher-4 body says "Draft initialized" and final line reports `NEW=1 GAPS=1`. Resolution: model-access research is not converged; NEW-F04.
- Source all-model extraction list conflicts with the ledger's requested variants. Source list: B/L/7B; ledger: S+/B/H+. Resolution: port requirements need a canonical variant table; NEW-F05.
- Source and prior-art target disagree with official image-size evidence. Source/prior art use 518; official HF docs/cards show 224 examples and multiples-of-16 behavior. Resolution: image-size behavior must be specified and tested; NEW-F09.
- Prior-art dependency floor conflicts with official docs availability. Prior art says `transformers>=4.43`; official DINOv3 docs are present at v4.56.0 and absent at v4.55.0. Resolution: raise/verify dependency floor; NEW-F11.

## Source quality
- Researcher-1: weak. It has an anchor and status line but no substantive source-pipeline evidence.
- Researcher-2: weak. It has an anchor and status line but no substantive target-architecture evidence.
- Researcher-3: absent. No report file to review.
- Researcher-4: weak. It acknowledges a gap but provides no verified model-access evidence.
- Direct verification sources: strong for local code and prior-art branch because claims were checked against files/branch via `view`, `rg`, and `git show`; strong for repo ids/gating/license because checked against Hugging Face API/model pages; medium for image-size defaults because gated raw config files returned 401, so evidence comes from official docs/model cards rather than model `config.json`.

<a name="source-pipeline"></a>
### Sub-area: source-pipeline

Coverage assessment: verified directly from `SINCHI_DINOV3`. The source includes `DINOv3LinearClassifier.get_embeddings`, per-image pickle embedding extraction, cached-embedding datasets, direct classifier-on-embedding training/validation, raw checkpoint saving under `model_state_dict`, PEFT training, and config-driven 518 image size.

Gaps: NEW-F01, NEW-F05, NEW-F06, NEW-F08, NEW-F09, NEW-F13, NEW-F14. The researcher report did not cover the source, and the port plan still needs a canonical model variant table, explicit embedding-file compatibility choice, cached-training target design, image-size decision, embedding token semantics, and real checkpoint key evidence.

<a name="target-architecture"></a>
### Sub-area: target-architecture

Coverage assessment: verified directly from `MegaDetector-Classifier` main. The existing pipeline is a Lightning/ResNet-style trainer with dynamic algorithm/model selection, existing classifier inference families, no active DINOv3 module on main, and no DINOv3 dependencies in default requirements.

Gaps: NEW-F02, NEW-F07, NEW-F08, NEW-F12. The researcher report did not cover this. The plan must define additive/lazy integration points, avoid changing default requirements/runtime imports, add or explicitly avoid a `save_embeddings` inference surface, and prove existing ResNet/TIMM/SpeciesNet paths remain unchanged.

<a name="prior-art-branch"></a>
### Sub-area: prior-art-branch

Coverage assessment: verified with `git show docs/dinov3-integration-plan:<path>`. The branch has a converged phased design, DINOv3 model and linear-probe algorithm, loader v1, optional-deps plan, tests, and a P5 verifier log showing 44 selected tests passed.

Gaps: NEW-F03, NEW-F10, NEW-F11, NEW-F14, NEW-F15. No researcher-3 report exists. Prior art should be reused, but Phase 5 rejects the SINCHI formats needed for this task, Phase 8 is only a design target, and its dependency floor is too low unless separately proven.

<a name="dinov3-model-access"></a>
### Sub-area: dinov3-model-access

Coverage assessment: verified directly from Hugging Face model pages/API/docs. Required repo ids exist: `facebook/dinov3-vits16plus-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, `facebook/dinov3-vith16plus-pretrain-lvd1689m`. Dimensions from official model cards are 384, 768, and 1280 respectively. All three are manual-gated with DINOv3 license metadata. Transformers DINOv3 docs exist at v4.56.0 and not at v4.55.0.

Gaps: NEW-F04, NEW-F09, NEW-F11, NEW-F12. Researcher-4 did not complete the model-access work. The plan must document manual access/license, set or justify a correct Transformers floor, and resolve default/preprocessing image size versus SINCHI's 518 setting.

## Coverage analysis
- Ledger items: 4
- Verified this round: 4
- Uncovered: []
- Automated scope_check.sh: FAIL (rc=2, `ledger missing required field .repo_root`)

STATUS: NEEDS-MORE SCOPE_CHECK=FAIL  COVERED=4/4  UNCOVERED=[]            NEW=15
