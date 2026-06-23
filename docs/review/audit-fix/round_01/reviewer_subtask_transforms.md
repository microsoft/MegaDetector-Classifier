# Reviewer subtask — DINOv3 transform parity

Round: 1  
Role: read-only behavioral reviewer  
Scope: `src/models/classification/dinov3_base/transforms.py` and owned call sites vs `DINOv3_classification_port_plan.md` §10/§12.

## Inputs read

- `~/.copilot/skills/_shared/iterative-anti-drift.md`
- `docs/review/audit-fix/SCOPE_LEDGER.json`
- `docs/review/audit-fix/COVERAGE_LOG.jsonl`
- `DINOv3_classification_port_plan.md` §10/§12
- Owned call sites in:
  - `src/models/classification/dinov3_base/base_classifier.py`
  - `src/models/classification/dinov3_base/amazon.py`
  - `scripts/extract_embeddings.py`
  - `src/models/classification/dinov3_base/__init__.py`
  - `src/models/classification/__init__.py`

## Findings

<a name="SUB-TRF-001"></a>
### SUB-TRF-001 — `DINOv3Transform` does not enforce RGB conversion itself

**Behavioral impact:** Plan §10 assigns RGB conversion to `DINOv3Transform` itself: `DINOv3Transform` must do bicubic 518 resize, `ToTensor`, ImageNet normalize, and RGB convert. Current owned call sites usually pre-convert images to RGB before calling the transform, but the public transform is re-exported and fails when used directly on grayscale or RGBA PIL inputs. That violates the transform contract and makes transform parity dependent on every caller remembering to convert first.

**Evidence:**

- `DINOv3_classification_port_plan.md:241` requires `DINOv3Transform` to perform `RGB convert` as part of the parity transform.
- `src/models/classification/dinov3_base/transforms.py:19-26` composes only `Resize`, `ToTensor`, and `Normalize`; there is no `RGB` conversion step.
- `src/models/classification/dinov3_base/transforms.py:28-29` directly calls `self.trans(img)`, so mode conversion cannot happen elsewhere inside the transform.
- `src/models/classification/dinov3_base/__init__.py:4,10` and `src/models/classification/__init__.py:19-23` expose `DINOv3Transform` as a public DINOv3 symbol, so direct use is expected.
- Verification command run from repo root:
  - `uv run python - <<'PY' ... DINOv3Transform()(Image.new(mode, ...)) ... PY`
  - Result: `RGB (3, 518, 518)`, `L RuntimeError output with shape [1, 518, 518] doesn't match the broadcast shape [3, 518, 518]`, `RGBA RuntimeError The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0`.

**Proposed behavioral fix:**

Move RGB conversion into `DINOv3Transform.__call__` before the composed transform. For PIL images, call `img.convert("RGB")`; for NumPy array inputs, convert through `Image.fromarray(...).convert("RGB")` if that input type is intended to be supported. Keep the existing caller-side `convert("RGB")` calls; they become harmless idempotent safeguards. Add unit coverage for RGB, grayscale, and RGBA inputs producing `(3, 518, 518)` with ImageNet normalization.

## Non-findings / checked behavior

- **Bicubic 518 resize/crop semantics:** `src/models/classification/dinov3_base/transforms.py:20-23` uses `transforms.Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC)` with no `CenterCrop`; owned call sites instantiate `target_size=518` through `base_classifier.py:176-180`, `amazon.py:96-108`, and `scripts/extract_embeddings.py:150`. This matches plan §10's direct bicubic resize-to-518 requirement.
- **ImageNet normalization:** `src/models/classification/dinov3_base/transforms.py:14-15,24-25` uses the plan §10 ImageNet mean/std after `ToTensor`.
- **Batch tensor behavior in owned call sites:** call sites apply `DINOv3Transform` per image and let `DataLoader` stack tensors before model inference/training (`base_classifier.py:267-270`, `base_classifier.py:333-337`, `scripts/extract_embeddings.py:160-174`). I did not find an owned call site passing an already-batched tensor through `DINOv3Transform`.
- **Direct tensor input:** direct `DINOv3Transform(torch.zeros(...))` currently raises because `ToTensor` expects PIL/ndarray after `Resize`. That is not breaking the inspected owned call sites, but a future parity wrapper against the Hugging Face image processor should either document per-image-only input or add an explicit batched/list API.

STATUS: DONE
