<a name="dinov3-model-access"></a>
## Sub-area: dinov3-model-access

## Key Findings

1. **[NEW] Target label mapping is confirmed.** The three requested HF identifiers are: small-plus → `facebook/dinov3-vits16plus-pretrain-lvd1689m`, base → `facebook/dinov3-vitb16-pretrain-lvd1689m`, huge-plus → `facebook/dinov3-vith16plus-pretrain-lvd1689m`. The SINCHI huge-plus config spelling matches the official HF/Meta spelling. Sources: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009, https://github.com/facebookresearch/dinov3#pretrained-backbones-via-hugging-face-transformers

2. **[NEW] The official HF DINOv3 collection contains 12 backbone repos plus a CHMv2 head.** ViT LVD-1689M repos are `vits16`, `vits16plus`, `vitb16`, `vitl16`, `vith16plus`, `vit7b16`; SAT-493M repos are `vitl16`, `vit7b16`; ConvNeXt repos are tiny/small/base/large; `facebook/dinov3-vitl16-chmv2-dpt-head` is a downstream depth head, not a plain embedding backbone. Source: https://huggingface.co/api/collections/facebook/dinov3-68924841bd6b561778e31009

3. **[NEW] Do not use `vitsplus` or `vitl16plus` as HF repo ids.** The v4.56.0 Transformers docs contain a likely typo `facebook/dinov3-vitsplus-pretrain-lvd1689m` in one quantization snippet, but the official collection and Meta README use `facebook/dinov3-vits16plus-pretrain-lvd1689m`. HF API search under `facebook/` returned no `dinov3-vitl16plus` repo in the official collection. Sources: https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/dinov3, https://github.com/facebookresearch/dinov3#pretrained-backbones-via-hugging-face-transformers

4. **[NEW] Target architecture facts are cross-verified.** DINOv3 ViT uses patch size 16, image-size config default 224, and 4 register/storage tokens. Embedding dims are S/S+=384, B=768, L=1024, H+=1280, 7B=4096. Param counts are exact from HF safetensors metadata; dims/patch/image/registers are verified from the HF model card and Meta `backbones.py`. Sources: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m#technical-specifications, https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/hub/backbones.py, per-model HF API URLs in Sources.

5. **[NEW] All official HF DINOv3 backbone repos are gated with manual approval.** Per-model HF API metadata reports `gated: "manual"`, `license: "other"`, `license_name: "dinov3-license"`, and `license_link` to the Meta DINOv3 License. Direct unauthenticated access to gated `config.json` returns: `Access to model ... is restricted. You must have access to it and be authenticated to access it. Please log in.` Sources: https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m?blobs=true, https://huggingface.co/docs/hub/models-gated, https://ai.meta.com/resources/models-and-libraries/dinov3-license/

6. **[NEW] README must document both browser approval and CLI authentication.** HF docs state gated-model access requests must be made from the browser and script downloads require a user token. Current HF CLI docs use `hf auth login`; older environments may expose `huggingface-cli login`. A read or fine-grained read token is sufficient for downloads after approval. Sources: https://huggingface.co/docs/hub/models-gated#download-files, https://huggingface.co/docs/hub/security-tokens, https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login

7. **[NEW] Minimum Transformers version is `transformers>=4.56.0`.** Meta README says released Transformers packages support DINOv3 starting with 4.56.0; the v4.56.0 release notes include “Dino v3”; v4.56.0 docs expose `DINOv3ViTModel`, `DINOv3ConvNextModel`, `DINOv3ViTConfig`, and `DINOv3ViTImageProcessorFast`. Sources: https://github.com/facebookresearch/dinov3#pretrained-backbones-via-hugging-face-transformers, https://github.com/huggingface/transformers/releases/tag/v4.56.0, https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/dinov3

8. **[NEW] Use `AutoModel`/`AutoImageProcessor`; do not depend on a generic exported `DINOv3Model` name.** HF model API metadata uses `architectures: ["DINOv3ViTModel"]`, `model_type: "dinov3_vit"`, `pipeline_tag: "image-feature-extraction"`, and `library_name: "transformers"`. ViT and ConvNeXt have separate model types. Source: https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m?blobs=true, https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/dinov3

9. **[NEW] Auto-download is already built into `from_pretrained`.** `AutoImageProcessor.from_pretrained(repo_id)` and `AutoModel.from_pretrained(repo_id)` download missing files and cache them; `local_files_only=True` prevents network access; `force_download=True` refreshes; `token=True` or a logged-in CLI token can be used for gated repos. Sources: https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained, https://huggingface.co/docs/hub/models-gated#download-files

10. **[NEW] HF cache location and local checks are well-defined.** Default Hub cache is `~/.cache/huggingface/hub`, overridden by `HF_HOME` or `HF_HUB_CACHE`; repo cache layout is `refs/`, `blobs/`, `snapshots/`; `try_to_load_from_cache()` checks local cache without HTTP; `snapshot_download()` can prefetch repo snapshots with `allow_patterns`, `local_files_only`, and `token`. Sources: https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache, https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cache#huggingface_hub.try_to_load_from_cache, https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download

11. **[NEW] CLS extraction remains token 0 despite register tokens.** HF docs and source show token order `[CLS, register tokens, patch tokens]`; `outputs.last_hidden_state[:, 0, :]` is CLS and `outputs.pooler_output` is `sequence_output[:, 0, :]` after final norm. Patch averaging should start at `1 + model.config.num_register_tokens`, not at token 1. Sources: https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/dinov3#notes, https://raw.githubusercontent.com/huggingface/transformers/v4.56.0/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py, https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

## Model variant table

Weights download size is the sum of `model*.safetensors` file sizes from HF API metadata; config/processor files add only KB. All rows below are manual-gated on HF.

| Size label | HF repo id | Embedding dim | Patch | Image size | Params | Download size | Gated? |
|---|---|---:|---:|---:|---:|---:|---|
| small | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 16 | 224 | 21,596,544 | 86.4 MB / 0.08 GiB | yes, manual |
| small-plus **target** | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | 16 | 224 | 28,692,864 | 114.8 MB / 0.11 GiB | yes, manual |
| base **target** | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 16 | 224 | 85,660,416 | 342.7 MB / 0.32 GiB | yes, manual |
| large | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 | 16 | 224 | 303,129,600 | 1.21 GB / 1.13 GiB | yes, manual |
| huge-plus **target / SINCHI** | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 | 16 | 224 | 840,592,640 | 3.36 GB / 3.13 GiB | yes, manual |
| 7B | `facebook/dinov3-vit7b16-pretrain-lvd1689m` | 4096 | 16 | 224 | 6,716,035,072 | 26.86 GB / 25.02 GiB, 6 shards | yes, manual |
| SAT large | `facebook/dinov3-vitl16-pretrain-sat493m` | 1024 | 16 | 224 | 303,129,600 | 1.21 GB / 1.13 GiB | yes, manual |
| SAT 7B | `facebook/dinov3-vit7b16-pretrain-sat493m` | 4096 | 16 | 224 | 6,716,035,072 | 26.86 GB / 25.02 GiB, 6 shards | yes, manual |

Other official HF collection entries for completeness: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, `facebook/dinov3-convnext-small-pretrain-lvd1689m`, `facebook/dinov3-convnext-base-pretrain-lvd1689m`, `facebook/dinov3-convnext-large-pretrain-lvd1689m`, and `facebook/dinov3-vitl16-chmv2-dpt-head`.

Recommended callable-name registry for ViT backbones:

```python
DINOV3_MODEL_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge_plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "sat_large": "facebook/dinov3-vitl16-pretrain-sat493m",
    "sat_giant": "facebook/dinov3-vit7b16-pretrain-sat493m",
}
```

## Access steps (README-ready)

1. Create or sign in to a Hugging Face account: https://huggingface.co/join
2. Open the DINOv3 collection or the exact model page you plan to use, for example:
   - https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
   - https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
   - https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m
3. Submit the gated-access form. The DINOv3 gate asks for identity/affiliation fields and includes a checkbox accepting the license/privacy terms; HF API metadata sets the button text to `Submit`. Wait until HF shows that access is approved. The repos are `gated: manual`, so approval may not be instant.
4. Create a token at https://huggingface.co/settings/tokens. Use a `read` token or a fine-grained token with read access to the DINOv3 repo(s).
5. Log in on the machine that will run MegaDetector-Classifier:

```bash
# Current HF CLI:
hf auth login
hf auth whoami

# Non-interactive option:
export HF_TOKEN=hf_...
hf auth login --token "$HF_TOKEN"

# Legacy CLI, if your installed huggingface_hub exposes it:
huggingface-cli login
```

6. Test access before downloading large weights:

```bash
MODEL_ID=facebook/dinov3-vitb16-pretrain-lvd1689m
hf download "$MODEL_ID" config.json
```

7. If the test fails with `GatedRepo` or `Access to model ... is restricted`, revisit the model page while logged in, confirm approval, and rerun `hf auth login` or pass `token=`/`HF_TOKEN` to the loader.

## transformers + deps requirements

- Minimum: `transformers>=4.56.0`.
- Required runtime stack for HF loading: `torch`, `torchvision`, `safetensors`, `huggingface_hub` (normally pulled by `transformers`, but the CLI/docs should name it because users need `hf auth login`).
- Optional: `accelerate` if using `device_map="auto"`; `torchao` only for the optional int4 quantization example in HF docs; FlashAttention packages only if selecting a flash attention implementation.
- `AutoModel.from_pretrained(model_id, attn_implementation="eager")` is valid. HF docs list `eager`, `sdpa`, and flash implementations; default uses SDPA when available. Keep `eager` if the source wrapper needs attention outputs or wants the manual attention path.
- Use `AutoImageProcessor.from_pretrained(model_id)` instead of hard-coding transforms. The model config default image size is 224, but the HF model card states ViT models can accept larger images whose dimensions are multiples of patch size 16; otherwise the model crops to the closest smaller multiple.

## Auto-download & availability-check recipe

Default simple path: let Transformers download and cache automatically.

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained(model_id, token=True)
model = AutoModel.from_pretrained(
    model_id,
    token=True,
    attn_implementation="eager",
)
```

Local-only check path:

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
model = AutoModel.from_pretrained(model_id, local_files_only=True)
```

Prefetch path for explicit download and clearer errors:

```python
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

try:
    snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ],
        token=True,
        local_files_only=False,
    )
except GatedRepoError as exc:
    raise RuntimeError(
        f"DINOv3 model {model_id} is gated. Visit https://huggingface.co/{model_id}, "
        "submit/accept the DINOv3 license gate, wait for approval, then run `hf auth login` "
        "or set HF_TOKEN."
    ) from exc
except RepositoryNotFoundError as exc:
    raise RuntimeError(f"Unknown DINOv3 repo id: {model_id}") from exc
except LocalEntryNotFoundError as exc:
    raise RuntimeError(
        f"DINOv3 model {model_id} is not in the local HF cache and downloads are disabled. "
        "Remove local_files_only=True or prefetch the model while online."
    ) from exc
except HfHubHTTPError as exc:
    raise RuntimeError(f"Could not download DINOv3 model {model_id}: {exc}") from exc
```

Cache-inspection path without network:

```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
    cached = try_to_load_from_cache(repo_id=model_id, filename=filename)
    if isinstance(cached, str):
        print(filename, "cached at", cached)
    elif cached is _CACHED_NO_EXIST:
        print(filename, "is known absent for this revision")
    else:
        print(filename, "not cached")
```

Implementation notes:
- `from_pretrained` is enough for normal auto-download/cache behavior.
- `snapshot_download` is useful for an explicit `--download-model` command or startup preflight.
- `try_to_load_from_cache` proves local cache presence only; it does not prove that the user has online access.
- For all-model-size support, include sharded weight patterns because ViT-7B uses six `model-0000x-of-00006.safetensors` files.
- Surface the HF cache path in errors: default `${HF_HOME:-~/.cache/huggingface}/hub`, overridable with `HF_HUB_CACHE`.

## Sources

- HF DINOv3 collection page: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009
- HF DINOv3 collection API: https://huggingface.co/api/collections/facebook/dinov3-68924841bd6b561778e31009
- Target model APIs: https://huggingface.co/api/models/facebook/dinov3-vits16plus-pretrain-lvd1689m?blobs=true, https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m?blobs=true, https://huggingface.co/api/models/facebook/dinov3-vith16plus-pretrain-lvd1689m?blobs=true
- Base model card with shared DINOv3 technical specs: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- Meta DINOv3 README: https://github.com/facebookresearch/dinov3#pretrained-backbones-via-hugging-face-transformers
- Meta DINOv3 backbone source: https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/hub/backbones.py
- Meta DINOv3 License: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- HF gated model docs: https://huggingface.co/docs/hub/models-gated
- HF token docs: https://huggingface.co/docs/hub/security-tokens
- HF CLI auth docs: https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login
- Transformers DINOv3 docs v4.56.0: https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/dinov3
- Transformers v4.56.0 release notes: https://github.com/huggingface/transformers/releases/tag/v4.56.0
- Transformers `from_pretrained` docs: https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
- HF Hub cache guide: https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-cache
- HF Hub `try_to_load_from_cache`: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cache#huggingface_hub.try_to_load_from_cache
- HF Hub `snapshot_download`: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download
- HF DINOv3 ViT source: https://raw.githubusercontent.com/huggingface/transformers/v4.56.0/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py

## Cross-Area Connections

- **source-pipeline / researcher-1:** The SINCHI wrapper's `AutoModel.from_pretrained(..., attn_implementation="eager")` + `AutoImageProcessor` pattern is compatible with HF DINOv3. If SINCHI extracts `last_hidden_state[:, 0, :]`, that is equivalent to HF `pooler_output` for ViT DINOv3 after final norm; preserve this for parity unless the port intentionally adds CLS+patch-mean features.
- **target-architecture / researcher-2:** Requirements should add `transformers>=4.56.0` and ensure the runtime has `torch`, `torchvision`, `safetensors`, and `huggingface_hub`; `accelerate` is needed only if the port uses `device_map="auto"`. Model registry should be additive and should include all ViT aliases above.
- **prior-art-branch / researcher-3:** Any existing `dinov3_base` or `src/models/dinov3.py` implementation should be checked for the exact `vits16plus` spelling, gated-repo error handling, and ViT-7B sharded-file support if it claims “all sizes”.

## Gaps

No blocking gaps. I did not authenticate to HF, so I could not read gated `config.json` contents directly; this is mitigated by HF model pages/API, Meta source, and Transformers docs. Authenticated implementation testing should still verify one small gated download after a maintainer accepts the license.

## Confidence

| Finding | Confidence | Basis |
|---:|---|---|
| 1 | HIGH | Official HF collection + Meta README list exact ids. |
| 2 | HIGH | HF collection API returned all collection entries. |
| 3 | MEDIUM-HIGH | Official docs/collection contradict the typo; unauthenticated API search found no `vitl16plus` repo. |
| 4 | HIGH | HF model card + Meta source + HF safetensors metadata agree. |
| 5 | HIGH | Per-model HF API reports manual gating/license; unauthenticated gated-file fetch returned 401. |
| 6 | HIGH | HF gated-model/token/CLI docs explicitly describe browser request + token login. |
| 7 | HIGH | Meta README and Transformers v4.56.0 release/docs agree. |
| 8 | HIGH | HF model API and docs identify `DINOv3ViTModel`/`dinov3_vit`. |
| 9 | HIGH | Transformers `from_pretrained` docs define download/cache/local/token behavior. |
| 10 | HIGH | HF Hub cache and API docs define cache layout and helpers. |
| 11 | HIGH | HF docs and source code show token order and pooler computation. |

STATUS: NEW=11 GAPS=0
