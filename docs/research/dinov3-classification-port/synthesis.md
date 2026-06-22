# Research synthesis — DINOv3 classification port into MegaDetector-Classifier

**Convergence:** Round 2 — `CONVERGED SCOPE_CHECK=PASS COVERED=4/4 NEW=0` (inquisitor) + `scope_check.sh` exit 0.
**Method:** 4 researchers (round 1) + 1 narrow empirical researcher (round 2) + 2 inquisitor passes, all claims independently spot-checked (checkpoint `torch.load`, HF docs fetch, `git show`, target file reads).

---

## Executive summary

The goal is to port SINCHI_DINOV3's two-stage **DINOv3 embedding extraction → linear-probe classification** workflow (all model sizes) into MegaDetector-Classifier (a PytorchWildlife-style library) without breaking its existing pipeline. Three facts dominate the plan:

1. **There is substantial prior art.** Branch `docs/dinov3-integration-plan` already implemented phases P0–P5 of a converged design: optional-deps scaffolding, a frozen-backbone `DINOv3Classifier`, a `DINOv3Linear` Lightning algorithm, a `dinov3_base` inference family, regression tests, and `requirements-dinov3.txt`. It deliberately **deferred** the SINCHI checkpoint loaders, cached-embedding workflow, named model sizes, and embedding-save API. The port is mostly **finishing the prior art**, not greenfield.
2. **The team's trained Amazon checkpoints exist on disk for every requested size** (S+/B/H+, plus S/L and a PEFT-H+), and their on-disk key layout is now empirically documented — closing the prior art's "Phase 8" evidence gap.
3. **DINOv3 weights are HuggingFace-gated (manual approval) and need `transformers>=4.56.0`.** README must document the license-acceptance + token-login flow.

---

## Findings by sub-area

### 1. source-pipeline (SINCHI_DINOV3) — what to port

- **Two-stage workflow** (`EXPERIMENT_FEATURES.md:13-28`): `extract_embeddings.py` precomputes embeddings → set `experiment.use_cached_embeddings: true` → `train.py` trains only the linear head on cached embeddings (backbone skipped).
- **Embedding extraction** (`scripts/extract_embeddings.py`): config-driven (`--config`, `--output_dir`, `--split`). Reads `data.train_csv`/`val_csv` (+`root_dir`); CSV columns `Photo_ID` (path relative to root) and `Label` (class name); deterministic val transform (bicubic resize to `image_size`=518, ToTensor, ImageNet normalize); CLS-token embedding via `model.get_embeddings()`.
- **Embedding file format (verified on disk):** one **pickle per image**, `dict{embedding: np.ndarray float32 (dim,), label: int, image_path: str-relative}`. Layout `<output_dir>/{train,val}/<image_path with / and \ → _>.pkl`; all-models wrapper adds a `<base>/<model-id>/` level. `EmbeddingDataset` re-derives the same safe filename from the CSV and **trusts the CSV label** (ignores the stored `label`).
- **Model wrapper** (`models/dinov3_model.py::DINOv3LinearClassifier`): `self.backbone = AutoModel.from_pretrained(name, attn_implementation="eager")` + `self.classifier = nn.Linear(dim, num_classes)`; `get_embeddings()` returns `last_hidden_state[:,0,:]` (CLS); embedding dim auto-detected from `config.hidden_size`; frozen backbone for linear probing.
- **Checkpoint format** (`scripts/train.py:588-612`): `torch.save({epoch, model_state_dict, optimizer_state_dict, metrics, config, scheduler_state_dict})`. `config.model.name` carries the exact HF id; `config.model.num_classes`=36; `config.data.image_size`=518.
- **PEFT path** (`train_peft.py`): LoRA/PiSSA; forces `use_cached_embeddings=false` (needs backbone forward). Optional for the 4 requirements.
- **Known source bug to NOT copy:** `validate.py` hardcodes `facebook/dinov3-vit7b16-...` when reconstructing the model instead of reading `config.model.name` — the port must read the name from the checkpoint.
- **Source variant mismatch:** `extract_embeddings_all_models.py`/`sweep_models.yaml` iterate B/L/7B, **not** the requested S+/B/H+. The port supplies an explicit S+/B/H+ registry.

### 2. target-architecture (MegaDetector-Classifier) — where it slots in, what must not break

- **Inference hierarchy:** `BaseClassifierInference` (stub base) → per-family packages `resnet_base/`, `timm_base/`, `speciesnet_base/`, and an empty `dinov3_base/`. The **TIMM family is the closest template** (`TIMM_BaseClassifierInference`: class attrs `BACKBONE/MODEL_NAME/IMAGE_SIZE`, `_load_model` with `wget` hub-cache download, `results_generation`, `single_image_classification`, `batch_image_classification` over `ImageFolder`/`DetectionCrops`).
- **Amazon precedent:** `resnet_base/amazon.py::AI4GAmazonRainforest` (36 `CLASS_NAMES`, Zenodo URL, softmax result dicts). The DINOv3 Amazon class should mirror this surface.
- **Optional-dependency precedent:** `speciesnet_base/__init__.py` wraps `from .speciesnet_tf import *` in `try/except ImportError` so `import classification` never fails when a heavy dep is absent.
- **`.npz` embedding-save precedent already exists:** `src/algorithms/plain.py` test/predict paths already save `feats`, `logits`, `file_ids` to `.npz` — a native target idiom for the save-embeddings feature.
- **Training path:** `main.py` (Typer) resolves YAML strings through registries `datasets.__dict__[name]`, `algorithms.__dict__[name]`, `models.__dict__[name]`. Default binds `Plain`/`Custom_Crop`/`PlainResNetClassifier`.
- **Do-not-break inventory:** default `configs/config.yaml`; the three registries (existing names must keep resolving); `Custom_Crop`/`Plain` contracts; existing `.npz` keys; public imports (`PlainResNetClassifier`, `AI4GAmazonRainforest`, `AI4GOpossum`, `AI4GSnapshotSerengeti`, `CustomWeights`, `DeepfauneClassifier`, `DFNE`, optional `SpeciesNetTFInference`); `Classification_Inference_Transform` default behavior; importing `classification` must not require `transformers`.
- **Deps:** runtime `requirements.txt` has **no** `transformers`/`peft`. Mixed Python floors across files (README 3.9+, conda 3.8.18, uv.lock ≥3.10). `transformers`/`peft` must stay **optional**.

### 3. prior-art-branch (`docs/dinov3-integration-plan`) — reconciliation

- **Design** (`docs/design/dinov3-integration/final.md`, converged) planned P0–P9. **Implemented: P0–P5.** Pending: P6 (full FT), P7 (PEFT), **P8 (SINCHI raw/PEFT loaders)**, P9 (configs/docs).
- **Implemented & reusable:** `requirements-dinov3.txt`; guarded import in `classification/__init__.py` (raises actionable `ImportError`); `Custom_Crop(conf, image_size=224)` + `_build_data_transforms` hook (DINOv3 uses 518); `src/models/dinov3.py::DINOv3Classifier` (frozen; backbone named **`self.feature`**); `src/algorithms/dinov3_linear.py::DINOv3Linear` (image-based linear probe; PEFT/full-FT are `NotImplementedError` stubs); `dinov3_base/{base_classifier,custom_weights}.py` (loader **v1 = MD-Cls `state_dict`+`net.` only**); tests (unit/integration/regression incl. invariants + ResNet baselines).
- **The deferred loader is the crux:** `_extract_state_dict` already contains **stubbed branches that raise `NotImplementedError("SINCHI checkpoint flavours land in Phase 8")`** for `model_state_dict`+`backbone.` (raw) and for `base_model.model.` (PEFT). The port implements exactly these two branches.
- **Two corrections required:** (a) `requirements-dinov3.txt` pins `transformers>=4.43` but DINOv3 needs **`>=4.56.0`**; (b) prior art names the backbone `feature` while SINCHI checkpoints use `backbone.*` → the loader must remap `backbone.`→`feature.`.
- **Recommendation:** **rebase/finish, don't supersede.** Cherry-pick P0–P5 onto a feature branch, then add the SINCHI loaders, named-size registry, save-embeddings API, cached-embedding training, and docs.

### 4. dinov3-model-access — exact ids, dims, gating, login

| Size label | HF repo id | dim | patch | default img | gated |
|---|---|---:|---:|---:|---|
| small-plus | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | 16 | 224 | manual |
| base | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 16 | 224 | manual |
| huge-plus | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 | 16 | 224 | manual |
| (also on disk) small=384, large=1024 | `…vits16…`, `…vitl16…` | | | | manual |

- **Gating:** all `facebook/dinov3-*` repos are `gated: "manual"` under the Meta **DINOv3 License**. Access = create HF account → open model page → submit gate form → wait for approval → create read token → `hf auth login` (or legacy `huggingface-cli login`, or `HF_TOKEN`).
- **transformers floor:** DINOv3 docs exist at v4.56.0 (added 2025-08-14); 404 at v4.55.0 → **`transformers>=4.56.0`**.
- **Embeddings:** CLS token is `last_hidden_state[:,0,:]` (token 0) even though DINOv3 has 4 register tokens (order `[CLS, registers, patches]`). `pooler_output` == CLS after final norm. For patch-mean, start at `1 + num_register_tokens`.
- **Auto-download/check:** `from_pretrained` auto-downloads + caches (`~/.cache/huggingface/hub`, override `HF_HOME`/`HF_HUB_CACHE`); `local_files_only=True` for offline; `huggingface_hub.try_to_load_from_cache()` checks cache without HTTP; `snapshot_download()` for explicit preflight; catch `GatedRepoError`/`LocalEntryNotFoundError`/`RepositoryNotFoundError` for actionable errors. ViT-7B is 6 shards (not needed for S+/B/H+).

---

## Empirical checkpoint evidence (round 2, independently spot-checked)

All `best_model.pt` under `outputs/baseline_no_augmentations_noisy_same_class_mixup_<size>/checkpoints/`; loaded with the SINCHI venv (`torch 2.9.1`).

| Flavour | Top-level | state_dict keys | classifier | config |
|---|---|---|---|---|
| **RAW** (S+/B/H+/S/L) | `model_state_dict`(+optim/sched/metrics/config/epoch) | `backbone.*` (`embeddings.{cls_token,register_tokens,mask_token,patch_embeddings}`, `layer.N.*`, `norm.*`) + `classifier.weight`/`classifier.bias` | `(36, dim)` S+=384, B=768, H+=1280, S=384, L=1024 | `model.name`=exact HF id, `num_classes`=36, `image_size`=518, `peft`=null |
| **PEFT** (H+ LoRA) | same | `backbone.base_model.model.*` with `.lora_A.default.weight`/`.lora_B.default.weight` + `.base_layer.*` + `classifier.*` | `(36, 1280)` | + `peft`={lora, r=8, alpha=8, target_modules=[q/k/v/o/up/down/gate_proj]} |

Embeddings on disk confirm `dict{embedding: float32 (dim,), label: int, image_path: str}`; per-model dirs carry correct dims (S+=384, B=768, H+=1280, …).

---

## Plan-scope decisions (resolved in the port plan, not research gaps)

1. **Embedding file format:** keep SINCHI per-image `.pkl{embedding,label,image_path}` for drop-in compatibility with existing on-disk embeddings, with an optional batched `.npz` shard mode for efficiency.
2. **Save-embeddings API:** add a dedicated method (e.g. `extract_embeddings`/`batch_image_embeddings`) and an `extract_embeddings.py` entrypoint — do not overload existing `batch_image_classification` result dicts.
3. **Cached-embedding training:** add a new `EmbeddingDataset` + `DINOv3LinearEmbedding` algorithm (or a `use_cached_embeddings` branch in `DINOv3Linear`) that trains `classifier(embeddings)` and bypasses the backbone.
4. **Image size:** use **518** for parity with the trained SINCHI checkpoints (document the DINOv3 patch-16 note; 512 is the clean multiple but training used 518).
5. **Prior-art reuse:** rebase/finish P0–P5; implement the two stubbed SINCHI loader branches (`backbone.`→`feature.` remap; PEFT via `LoraConfig` from embedded `config.peft`).
6. **Deps:** bump `requirements-dinov3.txt` to `transformers>=4.56.0`; keep all DINOv3 deps optional.

## Open questions (low risk, for implementation)
- Does the team want PEFT-H+ inference supported in v1, or raw S+/B/H+ only? (Raw covers requirement 2; PEFT is an extra.)
- Confirm a single authenticated test download once a maintainer accepts the license (no agent could authenticate during research).
- Patch-mean vs CLS: SINCHI uses CLS — keep CLS for parity unless a future ablation says otherwise.

## Sources
All claims cited inline in `round_01/researcher-{1,2,3,4}_report.md`, `round_02/researcher-1_report.md`, and verified in `round_02/inquisitor_review.md` (checkpoint `torch.load`, HF docs/API fetches, `git show docs/dinov3-integration-plan:*`, target file reads).
