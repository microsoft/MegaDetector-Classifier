# DINOv3 Classification Port Plan

Porting SINCHI_DINOV3's **DINOv3 embedding extraction + linear‑probe classification (all model sizes)** into **MegaDetector‑Classifier**, without breaking the existing pipeline.

- **Source (port FROM):** `/shared/home/aahernandez/Andres_SSD_1/SINCHI_DINOV3/` (not a git repo)
- **Target (port INTO):** `/shared/home/aahernandez/MegaDetector-Classifier/` (PytorchWildlife‑style library)
- **Prior art:** branch `docs/dinov3-integration-plan` already implements phases P0–P5 of a converged design.
- **Backing research:** `docs/research/dinov3-classification-port/` (converged round 2; all checkpoint/model‑access facts independently verified).

---

## 1. TL;DR

The port is **mostly finishing existing prior art**, plus a SINCHI‑specific delta:

1. **Reuse** the prior‑art branch P0–P5 (optional‑deps scaffolding, frozen `DINOv3Classifier`, `DINOv3Linear`, the `dinov3_base` family skeleton, regression/invariant tests).
2. **Implement the deferred "Phase 8" SINCHI checkpoint loaders** — the prior art already has the stubbed `_extract_state_dict` branches; we now have the real on‑disk key layouts.
3. **Add the named product surface** (`DINOv3AmazonRainforest`, model‑size registry S+/B/H+) → requirement 2.
4. **Add inference‑with‑embedding‑save** → requirement 3.
5. **Add the two‑stage images→embeddings→finetune workflow** → requirement 4.
6. **Keep everything additive and optional** so the existing ResNet/TIMM/SpeciesNet pipeline is untouched → requirement 1.

---

## 2. Current state (verified)

### 2.1 Target inference architecture (`src/models/classification/`)
- `base_classifier.py::BaseClassifierInference` — thin `nn.Module` base (stub methods).
- Per‑family packages: `resnet_base/`, `timm_base/`, `speciesnet_base/`, and an **empty `dinov3_base/`** (only stale `__pycache__` on `main`).
- **TIMM family is the closest template:** `TIMM_BaseClassifierInference` defines `BACKBONE/MODEL_NAME/IMAGE_SIZE`, `_load_model` (wget hub‑cache download), `results_generation`, `single_image_classification`, `batch_image_classification` over `ImageFolder`/`DetectionCrops`.
- **Amazon precedent:** `resnet_base/amazon.py::AI4GAmazonRainforest` (36 `CLASS_NAMES`, Zenodo URL, softmax result dicts).
- **Optional‑dep precedent:** `speciesnet_base/__init__.py` wraps its import in `try/except ImportError`.
- **`.npz` save precedent:** `src/algorithms/plain.py` already saves `feats`/`logits`/`file_ids` to `.npz` in test/predict.

### 2.2 Target training path
- `main.py` (Typer) resolves YAML strings through `datasets.__dict__[name]`, `algorithms.__dict__[name]`, `models.__dict__[name]`. Default = `Plain` / `Custom_Crop` / `PlainResNetClassifier`.
- `main.py` runs ResNet‑oriented preprocessing (`batch_detection_cropping`, split) before dataset construction — **not appropriate for embedding‑based training** (see §9.3).

### 2.3 Prior‑art branch (`docs/dinov3-integration-plan`) — P0–P5 implemented
| File | Role | Reuse |
|---|---|---|
| `requirements-dinov3.txt` | optional deps `transformers/peft/huggingface_hub` | **reuse + fix floor** (→ `transformers>=4.56.0`) |
| `src/models/classification/__init__.py` | guarded DINOv3 import (actionable `ImportError`) | **reuse + split gate** (see §11) |
| `src/datasets/custom.py` | `Custom_Crop(conf, image_size=224)` + `_build_data_transforms` | reuse (DINOv3 uses 518) |
| `src/models/dinov3.py::DINOv3Classifier` | frozen backbone (**stored as `self.feature`**) + `self.classifier`; lazy `transformers` import | **reuse + extend** (add `get_embeddings`) |
| `src/algorithms/dinov3_linear.py::DINOv3Linear` | image‑based linear probe (PEFT/full‑FT are `NotImplementedError`) | reuse as the *image* path; add cached‑embedding path |
| `dinov3_base/base_classifier.py` | `DINOv3BaseInference` + `_extract_state_dict` (**v1 = MD‑Cls `state_dict`+`net.` only**) | **reuse + implement SINCHI branches** |
| `dinov3_base/custom_weights.py::DINOv3CustomWeights` | local `.ckpt` loader | reuse + extend (read config metadata) |
| `tests/**` (unit/integration/regression) | invariants + ResNet baselines + dinov3 stubs | reuse + extend |

**Two corrections required in prior art:** (1) `requirements-dinov3.txt` pins `transformers>=4.43` but DINOv3 needs **`>=4.56.0`**; (2) the guarded import requires `peft` to export *any* DINOv3 class — RAW inference must not require `peft` (§11).

### 2.4 The team's trained Amazon checkpoints (on disk, verified via `torch.load`)
Located at `…/SINCHI_DINOV3/outputs/baseline_no_augmentations_noisy_same_class_mixup_<size>/checkpoints/best_model.pt`. Sizes present: **S+, B, H+** (required) plus S, L, and a PEFT‑H+.

| Flavour | Top‑level keys | State‑dict keys | classifier | Embedded config |
|---|---|---|---|---|
| **SINCHI RAW** (S+/B/H+/S/L) | `epoch, model_state_dict, optimizer_state_dict, metrics, config, scheduler_state_dict` | `backbone.*` (`embeddings.{cls_token,register_tokens,mask_token,patch_embeddings}`, `layer.N.*`, `norm.*`) + `classifier.weight`/`classifier.bias` | `(36, dim)` — S+=384, B=768, H+=1280 (S=384, L=1024) | `model.name`=exact HF id, `num_classes`=36, `data.image_size`=518, `peft`=null |
| **SINCHI PEFT** (H+ LoRA) | same | `backbone.base_model.model.*` with `.base_layer.*`, `.lora_A.default.weight`, `.lora_B.default.weight` + `classifier.*` | `(36, 1280)` | + `peft`={lora, r=8, alpha=8, target_modules=[q/k/v/o/up/down/gate_proj]} |

### 2.5 DINOv3 model access (verified against HF)
| Size label | HF repo id | dim | patch | gated |
|---|---|---:|---:|---|
| small‑plus | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | 16 | manual |
| base | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 16 | manual |
| huge‑plus | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 | 16 | manual |

- All `facebook/dinov3-*` repos are **gated (manual approval)** under the Meta DINOv3 License.
- **`transformers>=4.56.0`** (DINOv3 docs added at v4.56.0; absent at v4.55.0).
- CLS embedding = `last_hidden_state[:,0,:]` (token 0; DINOv3 has 4 register tokens, order `[CLS, registers, patches]`).
- `from_pretrained` auto‑downloads + caches (`~/.cache/huggingface/hub`, override `HF_HOME`/`HF_HUB_CACHE`). Preflight via `huggingface_hub.try_to_load_from_cache` / `snapshot_download`; catch `GatedRepoError`.

---

## 3. Guiding principles

1. **Additive only.** New files and registry symbols; never rename/alter existing names, default `configs/config.yaml`, or the three `__dict__` registries.
2. **Optional dependencies.** `transformers`/`peft`/`huggingface_hub` stay optional behind guarded imports; `import src…` must succeed without them.
3. **Parity over convenience.** Match SINCHI's exact preprocessing and CLS‑token semantics; validate accuracy with golden tests before declaring done.
4. **Metadata is authority.** Loaders read the model id, num_classes, image_size, and class order from the checkpoint/manifest — never hardcode (avoid SINCHI's `validate.py` 7B‑hardcode bug).
5. **License‑safe distribution.** Do not redistribute full gated DINOv3 backbone weights; ship **classifier‑head‑only** checkpoints and pull the gated backbone from HF after login.

---

## 4. Requirement → component map

| Req | What | Primary components | Phase |
|---|---|---|---|
| 1 | Don't break existing pipeline | additive files, optional/guarded imports, invariant + ResNet‑baseline tests | P0, all |
| 2 | Inference loads team Amazon model w/ S+/B/H+ | `_extract_state_dict` SINCHI branches, `DINOv3AmazonRainforest(model_size=…)`, `DINOV3_MODEL_IDS`, `DINOv3AmazonManifest` | P2, P3 |
| 3 | Function: inference **and** save embeddings | `batch_image_classification_with_embeddings(…, save_embeddings_dir, return_embeddings)` + `extract_embeddings()` + `scripts/extract_embeddings.py` | P4 |
| 4 | 2‑stage images→embeddings→finetune; all sizes; auto check+download; README login | `DINOv3EmbeddingDataset`, cached mode in `DINOv3Linear`, `scripts/train_dinov3.py`, `ensure_model_available()`, `configs/dinov3_linear.yaml`, `docs/dinov3.md` | P1, P4, P5 |

---

## 5. Target file layout (added / modified)

```
src/
  models/
    dinov3.py                         # REUSE+EXTEND: DINOv3Classifier + get_embeddings()
    dinov3_registry.py                # NEW: DINOV3_MODEL_IDS, DINOV3_DIMS, ensure_model_available()
    classification/
      __init__.py                     # MODIFY: split guarded import (RAW vs PEFT)
      dinov3_base/
        __init__.py                   # REUSE+EXTEND exports
        base_classifier.py            # REUSE+EXTEND: full inference API + _extract_state_dict (4 flavours)
        custom_weights.py             # REUSE+EXTEND: read config metadata, classifier_only flavour
        amazon.py                     # NEW: DINOv3AmazonRainforest(model_size=...) + AMAZON manifest
        transforms.py                 # NEW: DINOv3Transform (bicubic 518 parity)
  datasets/
    dinov3_embedding.py               # NEW: DINOv3EmbeddingDataset (+ DataModule) ported from SINCHI
  algorithms/
    dinov3_linear.py                  # REUSE+EXTEND: cached-embedding training mode
scripts/
  extract_embeddings.py               # NEW: stage-1 entrypoint
  train_dinov3.py                     # NEW: stage-2 entrypoint (bypasses main.py)
configs/
  dinov3_linear.yaml                  # NEW: stage-2 config (does not touch config.yaml)
docs/
  dinov3.md                           # NEW: install/login/two-stage workflow/size table
requirements-dinov3.txt               # REUSE+FIX: transformers>=4.56.0
tests/
  unit/        integration/   regression/   # REUSE+EXTEND (synthetic fixtures + gated integration)
```

No existing file is renamed or has its public behavior changed; `classification/__init__.py` is edited only to **add** guarded DINOv3 exports.

---

## 6. Checkpoint schema & loader (`_extract_state_dict`)

The loader must recognize **four** flavours by structural signature and use embedded metadata as authority. The prior‑art `DINOv3Classifier` names its backbone `self.feature`, so SINCHI `backbone.*` keys remap to `feature.*`.

| Flavour | Signature | Action |
|---|---|---|
| **A. MD‑Cls own** (prior art v1) | `ckpt['state_dict']` has `net.*` | strip `net.` → `feature.*`/`classifier.*` (already implemented) |
| **B. SINCHI RAW** | `ckpt['model_state_dict']` has `backbone.*` (not `base_model.model.`) + `classifier.*` | remap `backbone.`→`feature.`, keep `classifier.*`; backbone id ← `config.model.name`; num_classes ← `config.model.num_classes` or `classifier.weight.shape[0]` |
| **C. SINCHI PEFT** | `ckpt['model_state_dict']` has `backbone.base_model.model.*` + lora keys | build HF backbone from `config.model.name`, wrap via `LoraConfig(**config.peft)`, remap `backbone.`→`feature.`, load; optionally `merge_and_unload()` for inference. **Requires `peft`.** |
| **D. classifier‑only** (NEW — stage‑2 output & distribution format) | top‑level has `classifier.weight/bias` + a `meta` block, **no backbone tensors** | instantiate the HF backbone from `meta.model_name` (downloaded from HF), load only `classifier.*` |

**`classifier_only` metadata block (authoritative):**
```python
{
  "format": "dinov3_classifier_only", "version": 1,
  "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
  "model_size": "base", "embedding_dim": 768, "num_classes": 36,
  "image_size": 518, "feature_type": "cls",
  "class_names": [...36 ordered names...],
  "transform": "dinov3_bicubic_518_imagenet", "transform_version": 1,
  "classifier.weight": <Tensor (36,768)>, "classifier.bias": <Tensor (36,)>
}
```

**Load‑time validation (all flavours):** assert `classifier.weight.shape[1] == feature.config.hidden_size`; assert requested `model_size`/`model_name` matches checkpoint; assert `num_classes == classifier.weight.shape[0]`; carry `class_names` ordering. Mismatch → explicit `ValueError`.

> **Rationale (rubber‑duck):** stage‑2 cached training never touches the backbone, so it must emit `classifier_only` (flavour D), not a full state dict. Distributing flavour D (a few hundred KB head) is also license‑safe; the gated backbone is fetched from HF per the user's login.

---

## 7. Model‑size registry, access & auto‑download

`src/models/dinov3_registry.py`:
```python
DINOV3_MODEL_IDS = {
    "small":      "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base":       "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large":      "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge_plus":  "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}
DINOV3_DIMS = {"small":384,"small_plus":384,"base":768,"large":1024,"huge_plus":1280}

def ensure_model_available(model_id: str, token=True) -> str:
    """Return local snapshot path; download if absent; actionable error if gated/unauth."""
    # 1) try_to_load_from_cache(config.json) → if cached, return
    # 2) snapshot_download(model_id, allow_patterns=[config.json, *.safetensors,
    #    preprocessor_config.json], token=token)
    # 3) except GatedRepoError → raise RuntimeError(
    #      f"{model_id} is gated. Visit https://huggingface.co/{model_id}, accept the
    #       DINOv3 license, then run `hf auth login` (or set HF_TOKEN).")
    #    except LocalEntryNotFoundError/RepositoryNotFoundError → actionable messages
```
- "All sizes callable" = any registry key resolves; `from_pretrained` auto‑downloads, `ensure_model_available` gives clear gated/offline errors.
- (ViT‑7B is 6 safetensors shards — include shard patterns only if 7B support is later added; not needed for S+/B/H+.)

**`DINOv3AmazonManifest`** (`amazon.py`) — versioned source of truth for the team's heads:
```python
DINOV3_AMAZON = {
  "small_plus": {"model_name": "...vits16plus...", "embedding_dim": 384,
                 "head_url": "<host>/amazon_dinov3_small_plus_head_v1.pt",
                 "sha256": "...", "flavor": "classifier_only", "revision": "<hf_rev>"},
  "base":       {... "embedding_dim": 768 ...},
  "huge_plus":  {... "embedding_dim": 1280 ...},
}
CLASS_NAMES = [ ...36 Amazon classes (same order as SINCHI)... ]
```
Per size: head URL/path, sha256, HF id + revision, dim, num_classes, class order, flavor. Verify sha256 before load.

---

## 8. Inference design (`dinov3_base/`)

**Full API parity** lives in `DINOv3BaseInference` (mirrors TIMM/ResNet so existing callers work unchanged):
- `__init__(weights, class_names, device, model_size|model_name, transform=None)`
- `_load_model(...)` → builds backbone via `ensure_model_available` + loads head via `_extract_state_dict`
- `results_generation(logits, img_ids, id_strip)` → same dict schema (`img_id, prediction, class_id, confidence, all_confidences`)
- `single_image_classification(...)`, `batch_image_classification(data_path=None, det_results=None, …)` over `ImageFolder`/`DetectionCrops`
- **`batch_image_classification_with_embeddings(data_path=None, det_results=None, save_embeddings_dir=None, return_embeddings=False, …)`** → one forward pass returns predictions **and** optionally saves/returns CLS embeddings (**requirement 3**)
- `extract_embeddings(...)` → embedding‑only convenience wrapper used by stage‑1
- All inference wrapped in `self.eval()` + `torch.inference_mode()` (avoid OOM/grad on H+)

`DINOv3AmazonRainforest(DINOv3BaseInference)` (`amazon.py`): `IMAGE_SIZE=518`, 36 `CLASS_NAMES`, `__init__(model_size="base", weights=None, device="cpu")` → resolves manifest, downloads/verifies the head, builds the backbone for `model_size`, loads. **This is requirement 2.**

---

## 9. Two‑stage workflow

### 9.1 Stage 1 — extract embeddings (requirement 3/4)
`scripts/extract_embeddings.py --model-size {small_plus,base,huge_plus} --csv <ann.csv> --root <img_root> --out <dir> [--split train|val|both]`:
- Builds the chosen DINOv3 backbone (auto‑download), applies `DINOv3Transform` (bicubic 518), computes CLS embeddings.
- Writes **SINCHI‑compatible** per‑image pickle `dict{embedding: np.float32 (dim,), label: int, image_path: str-relative}` under `<out>/{train,val}/<safe_name>.pkl`. `label=-1` for unlabeled/predict extraction.
- **Collision handling:** the SINCHI safe‑name (`/`,`\`→`_`) can collide across nested dirs; write a `manifest.csv` (`image_path,pkl_path,label`) and detect/raise on collisions, or preserve directory structure under `<out>/<split>/`.

### 9.2 Stage 2 — train linear head on cached embeddings (requirement 4)
- `DINOv3EmbeddingDataset` (ported from SINCHI `data/embedding_dataset.py`) reads the cached `.pkl` per CSV row, returns `(embedding, label)`.
- **Train a standalone `nn.Linear(dim, num_classes)`** — *do not* instantiate the heavy `DINOv3Classifier`/backbone for cached training (rubber‑duck Q3).
- `scripts/train_dinov3.py --config configs/dinov3_linear.yaml` runs the loop (AdamW, cosine, label‑smoothing CE — matching SINCHI), then **saves a `classifier_only` (flavour D) checkpoint** with full metadata (§6).
- Reuse `src/algorithms/dinov3_linear.py::DINOv3Linear` by adding a `use_cached_embeddings` branch whose `training_step` consumes embeddings directly; or keep a dedicated lightweight loop. Either way the output is flavour D.

### 9.3 `main.py` is bypassed for stage 2
Stage‑2 training uses `scripts/train_dinov3.py` directly and does **not** route through `main.py` (which runs ResNet crop/split side effects). `main.py`, `Custom_Crop`, and `Plain` remain untouched. (Optional later: a guarded `main.py` branch that skips cropping for embedding datasets — explicitly out of scope for v1.)

### 9.4 Round‑trip
Stage‑2 `classifier_only` checkpoint → loaded by §6 flavour D in `DINOv3BaseInference`/`DINOv3AmazonRainforest` for inference. The team's existing SINCHI **RAW**/**PEFT** checkpoints load via flavours B/C. All paths share the same backbone‑from‑HF + head‑on‑top assembly.

---

## 10. Transform parity (accuracy‑critical)

`dinov3_base/transforms.py::DINOv3Transform` replicates SINCHI exactly: **bicubic** resize to `(518, 518)`, `ToTensor`, ImageNet normalize (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`), RGB convert. Do **not** reuse the generic `Classification_Inference_Transform` default (bilinear). Keep **518** for parity with the trained checkpoints; document the patch‑16 caveat (518 ≠ 16·k; HF crops internally — match training, do not "fix" to 512 unless a parity test proves it safe).

**Golden parity test:** for S+/B/H+, compare embeddings/logits from SINCHI vs the ported code on a small fixed image set; assert max‑abs diff < tol.

---

## 11. Dependencies & optional‑dep gating

- `requirements-dinov3.txt` → `transformers>=4.56.0`, `huggingface_hub>=0.20,<1`, `peft>=0.10.0,<0.14.0`.
- **Split the guarded import** (fix prior‑art bug where everything required `peft`):
  - **RAW/embedding/inference classes** (`DINOv3Classifier`, `DINOv3BaseInference`, `DINOv3CustomWeights`, `DINOv3AmazonRainforest`) gate on `transformers` + `huggingface_hub` only.
  - **PEFT loading** (flavour C) gates additionally on `peft`, with an actionable `ImportError` if absent.
- Runtime `requirements.txt` stays unchanged (no `transformers`); add only a comment pointing to `requirements-dinov3.txt`.

---

## 12. Non‑breaking guarantees & test plan

**Invariants (must stay green):**
- `import src`, `import src.models`, `import src.models.classification` succeed with **no** DINOv3 deps installed.
- Existing public names import unchanged: `PlainResNetClassifier`, `AI4GAmazonRainforest`, `AI4GOpossum`, `AI4GSnapshotSerengeti`, `CustomWeights`, `DeepfauneClassifier`, `DFNE`, optional `SpeciesNetTFInference`.
- Default `python main.py` path, `configs/config.yaml`, and the three registries unchanged. ResNet baseline predictions bit‑identical.
- `Classification_Inference_Transform` default behavior unchanged (DINOv3 uses its own transform).

**Tests:**
- **Unit (default CI, synthetic):** `_extract_state_dict` flavours A–D via tiny fabricated state dicts (no HF, no real ckpt); registry lookups; manifest sha256 guard; dim‑mismatch `ValueError`.
- **Regression:** prior‑art invariant + ResNet baseline tests, re‑run on this branch.
- **Integration (gated, opt‑in):** real S+/B/H+ checkpoint load + 1‑image inference; golden transform parity. Skipped unless `DINOV3_RUN_INTEGRATION=1`, `DINOV3_AMAZON_CKPT_DIR=…`, and a valid HF token are present.
- Mark slow/gated tests with pytest markers; keep them out of default collection.

---

## 13. Phased execution

| Phase | Goal | Key deliverables | Acceptance |
|---|---|---|---|
| **P0 — Baseline** | Reuse prior art on a fresh branch off `main` | cherry‑pick/rebase P0–P5 files; branch `feat/dinov3-port` | invariant + ResNet baseline tests green; `import src` clean w/o deps |
| **P1 — Deps & access** | Registry + auto‑download + docs floor | `dinov3_registry.py`, `ensure_model_available`, `requirements-dinov3.txt`→4.56, split guarded import, `docs/dinov3.md` skeleton | gated‑error message correct; deps optional; unit tests for registry/gate |
| **P2 — SINCHI loaders** | Implement flavours B/C/D | `_extract_state_dict` branches + metadata validation | synthetic unit tests A–D pass; gated integration loads real RAW S+/B/H+ |
| **P3 — Inference surface** | Amazon product + full API parity | `DINOv3BaseInference` full API, `DINOv3AmazonRainforest(model_size)`, `DINOv3AmazonManifest`, `DINOv3Transform` | **req 2**: load+classify with S+/B/H+; API parity tests; golden parity |
| **P4 — Embedding save** | Inference + save embeddings | `batch_image_classification_with_embeddings`, `extract_embeddings`, `scripts/extract_embeddings.py` | **req 3**: predictions + `.pkl` from one pass; format matches SINCHI |
| **P5 — Two‑stage train** | Cached‑embedding finetune | `DINOv3EmbeddingDataset`, cached mode in `DINOv3Linear`, `scripts/train_dinov3.py`, `configs/dinov3_linear.yaml` | **req 4**: extract→train→load round‑trip on a small set; outputs flavour D |
| **P6 — (optional)** | PEFT‑H+ inference; full‑FT/PEFT train | flavour C `merge_and_unload`; prior‑art P6/P7 | PEFT‑H+ checkpoint loads + classifies |
| **P7 — Docs & E2E** | Finalize | `docs/dinov3.md` complete + README section; end‑to‑end run on one real Amazon ckpt + one authenticated download | docs reviewed; E2E passes |

**Ordering rationale:** the checkpoint schema (§6) and transform (§10) are defined **before** loaders (P2) and training (P5), so loaders aren't built against an undefined head format.

---

## 14. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `backbone.`→`feature.` remap misses buffer/non‑persistent keys under `strict=True` | Validate key sets in P2 synthetic tests; load with `strict=True` against a real ckpt in gated integration; fall back to `strict=False` with an explicit diff log only if a documented non‑persistent buffer mismatch is found |
| 518 vs patch‑16 hurts accuracy | Exact SINCHI transform + golden parity test (P3); keep 518 unless parity proves otherwise |
| Redistributing gated backbone weights violates license | Distribute **classifier‑only** heads (flavour D); pull backbone from HF after user login |
| Existing pipeline breakage via eager import | Split guarded imports (§11); invariant tests (P0) |
| CI fails on gated/local resources | Gated integration tests opt‑in via env vars; synthetic fixtures in default CI |
| H+ OOM | `torch.inference_mode()` + eval; batch‑size knob |
| Class‑name order drift | Carry `class_names` in manifest + `classifier_only` metadata; assert order on load |

---

## 15. README / `docs/dinov3.md` outline
1. Install: `pip install -r requirements-dinov3.txt` (note `transformers>=4.56.0`).
2. **Access the gated DINOv3 weights:** create HF account → open each model page (S+/B/H+) → submit the DINOv3 license gate → wait for approval → create a read token → `hf auth login` (or legacy `huggingface-cli login`, or `export HF_TOKEN=…`).
3. Model‑size table (§2.5).
4. Inference: load `DINOv3AmazonRainforest(model_size=…)`; `single_image_classification` / `batch_image_classification` / `batch_image_classification_with_embeddings`.
5. Two‑stage workflow: `scripts/extract_embeddings.py` → `scripts/train_dinov3.py --config configs/dinov3_linear.yaml`.
6. Troubleshooting: `GatedRepoError`, offline (`HF_HUB_OFFLINE`/`local_files_only`), cache location.

---

## 16. Open questions (low risk)
1. **PEFT‑H+ inference in v1?** Raw S+/B/H+ satisfies requirement 2; flavour C (PEFT) can land in P6 unless needed now.
2. **Head hosting** for `DINOv3AmazonManifest` (Zenodo, like the ResNet Amazon model? internal store?) + revision pinning.
3. Confirm a single authenticated gated download once a maintainer accepts the license (not testable during research).
4. CLS vs patch‑mean: keep CLS for SINCHI parity unless a future ablation says otherwise.

---

*Research backing this plan: `docs/research/dinov3-classification-port/synthesis.md` and round reports. Design critique incorporated from a rubber‑duck review (checkpoint‑flavour coherence, license‑safe head distribution, full API parity, inference‑with‑save method, transform parity, split optional‑dep gating, gated CI).*
