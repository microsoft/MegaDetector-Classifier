# Reviewer subtask: DINOv3 registry behavioral correctness

Scope: `src/models/dinov3_registry.py` against `DINOv3_classification_port_plan.md` §7 and §12. Read-only review; no source edits.

## Findings

### SUB-REG-001 — `ensure_model_available` does not resolve or validate registry size keys before Hub access

- Evidence:
  - `resolve_model_id()` provides the registry-key validation and maps size keys to HF repo ids at `src/models/dinov3_registry.py:30-40`.
  - `ensure_model_available()` accepts `model_id` but passes that raw value directly into `huggingface_hub.try_to_load_from_cache()` and `snapshot_download()` at `src/models/dinov3_registry.py:43-47` and `src/models/dinov3_registry.py:73-88`.
  - The plan says "All sizes callable" means any registry key resolves at `DINOv3_classification_port_plan.md:184`, and §12 lists registry lookups in the default unit-test plan at `DINOv3_classification_port_plan.md:265-266`.

Behavioral impact: direct calls such as `ensure_model_available("base")` or `ensure_model_available("small_plus")` try to access Hub repos named `base` / `small_plus` instead of the DINOv3 repos. Malformed inputs can also leak lower-level Hub validation errors before the registry's actionable "valid sizes" message is used.

Proposed behavioral fix: normalize at function entry, e.g. `model_id = resolve_model_id(model_id.strip())`, preserving full HF repo ids. For values that are neither known sizes nor valid HF repo ids, raise the existing actionable `ValueError` with valid size names before any Hugging Face call. Add unit coverage for `ensure_model_available("base", local_files_only=True)` using a fake hub module to assert the resolved `facebook/dinov3-vitb16-pretrain-lvd1689m` repo id is passed through.

### SUB-REG-002 — local/offline cache path can report success with only `config.json` cached, and returned paths are not consistently normalized

- Evidence:
  - The plan's preflight says `try_to_load_from_cache(config.json) -> if cached, return`, otherwise `snapshot_download(... allow_patterns=[config.json, *.safetensors, preprocessor_config.json])` at `DINOv3_classification_port_plan.md:174-178`.
  - The implementation checks only cached `config.json` and returns its parent immediately when `local_files_only=True` at `src/models/dinov3_registry.py:72-75`.
  - The same implementation only normalizes the cache-hit branch with `Path(...).parent` and returns `snapshot_download()`'s raw string directly at `src/models/dinov3_registry.py:77-88`.
  - §12 requires gated/offline DINOv3 paths to be opt-in and actionable, with default tests covering registry lookups and dim-mismatch errors at `DINOv3_classification_port_plan.md:257-268`.

Behavioral impact: in offline/local-only mode, a partial HF cache containing `config.json` but no `model.safetensors` or shard files returns a snapshot directory as if the model is available. The later `from_pretrained` call then fails outside this preflight with a less targeted error. Callers also receive differently normalized path strings depending on whether the path came from the cache fast path or `snapshot_download()`.

Proposed behavioral fix: make the cache fast path validate the expected weight files before returning, or route local-only checks through `snapshot_download(..., local_files_only=True, allow_patterns=...)` so `LocalEntryNotFoundError` is raised for incomplete caches. Return `str(Path(snapshot_path).expanduser().resolve())` for both cache and download paths. If the cache is incomplete in local-only mode, raise a `RuntimeError` that names the missing safetensors/preprocessor files and tells the user to disable `local_files_only` or pre-download after accepting the license.

## Checked with no finding

- `DINOV3_MODEL_IDS` and `DINOV3_DIMS` match the plan's §7 entries for `small`, `small_plus`, `base`, `large`, and `huge_plus` (`src/models/dinov3_registry.py:6-20`; `DINOv3_classification_port_plan.md:165-172`).
- Optional dependency gating is localized to `ensure_model_available()`; importing the registry itself does not import `transformers`, `huggingface_hub`, or `peft` (`src/models/dinov3_registry.py:1-5`, `src/models/dinov3_registry.py:48-54`; `DINOv3_classification_port_plan.md:247-253`).
